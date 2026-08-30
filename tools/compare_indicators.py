#!/usr/bin/env python3
"""
compare_indicators.py

Compare CalcServer's RSI and MACD against independent Python implementations,
over the exact bars CalcServer used.

Asks CalcServer for get_chart_data (bars plus indicator series), recomputes each
indicator locally in the standard variants, and reports which variant the server
matches and where it diverges.

The point is not to prove the server right. It is to establish which convention
it implements, so that collapsing the two Scala implementations into one does not
silently change what the chart shows. Re-run after any indicator change: the same
matching variant and a near-zero maximum difference means nothing moved.

Usage:
    python tools/compare_indicators.py --symbol NVDA --timeframe 5m
    python tools/compare_indicators.py --symbol NVDA --timeframe 1m --bars 500
    python tools/compare_indicators.py --symbol NVDA --rsi-period 14 --macd 12,26,9

Requirements: pyzmq (already used by the other tools here).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.service_locator import ServiceLocator


# --------------------------------------------------------------------------
# Indicator implementations, written from the textbook definitions rather than
# from the Scala, so that agreement means something.
# --------------------------------------------------------------------------

def _rsi_from(avg_gain, avg_loss):
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def rsi_wilder(closes, period):
    """Standard RSI: seeded from a simple average of the first `period` moves,
    then smoothed with alpha = 1/period."""
    if len(closes) <= period:
        return []
    gains, losses = [], []
    for prev, cur in zip(closes, closes[1:]):
        d = cur - prev
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    out = [(period, _rsi_from(avg_g, avg_l))]
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        out.append((i + 1, _rsi_from(avg_g, avg_l)))
    return out


def rsi_simple(closes, period):
    """The other common form: a plain rolling mean of gains and losses."""
    if len(closes) <= period:
        return []
    gains, losses = [], []
    for prev, cur in zip(closes, closes[1:]):
        d = cur - prev
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    out = []
    for i in range(period - 1, len(gains)):
        g = sum(gains[i - period + 1: i + 1]) / period
        l = sum(losses[i - period + 1: i + 1]) / period
        out.append((i + 1, _rsi_from(g, l)))
    return out


def true_ranges(bars, synthetic_first=False):
    """TR = max(H-L, |H-prevClose|, |L-prevClose|).

    Returns (bar_index, tr) pairs. TR is defined against the *previous* close,
    so the first bar of the window has none. Two conventions exist:

      synthetic_first=False  skip it. TR starts at bar 1. This is what TA-Lib
                             does, and what CalcServer does.
      synthetic_first=True   substitute H-L for bar 0. TradingView's ta.tr
                             does this when handle_na is set.

    The choice moves the seed window by one bar and so shifts the whole series
    slightly; the difference decays by (period-1)/period per bar thereafter.
    """
    out = []
    if synthetic_first:
        out.append((0, bars[0]["h"] - bars[0]["l"]))
    for i, (prev, cur) in enumerate(zip(bars, bars[1:]), start=1):
        out.append((i, max(cur["h"] - cur["l"],
                           abs(cur["h"] - prev["c"]),
                           abs(cur["l"] - prev["c"]))))
    return out


def atr_wilder(bars, period, synthetic_first=False):
    """Traditional ATR (Wilder 1978): seed with a simple average of the first
    `period` true ranges, then ATR = (prevATR*(period-1) + TR) / period."""
    tr = true_ranges(bars, synthetic_first)
    if len(tr) < period:
        return []
    prev = sum(v for _, v in tr[:period]) / period
    out = [(tr[period - 1][0], prev)]
    for idx, v in tr[period:]:
        prev = (prev * (period - 1) + v) / period
        out.append((idx, prev))
    return out


def atr_sliding_mean(bars, period, synthetic_first=False):
    """ATR as a plain rolling mean of the last `period` true ranges."""
    tr = true_ranges(bars, synthetic_first)
    if len(tr) < period:
        return []
    return [(tr[i][0], sum(v for _, v in tr[i - period + 1: i + 1]) / period)
            for i in range(period - 1, len(tr))]


def ema_seed_first(values, period):
    """EMA seeded from the first value. Same curve as the SMA-seeded form on the
    right, materially different over the first few `period` bars."""
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    prev = values[0]
    out = [(0, prev)]
    for i in range(1, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out.append((i, prev))
    return out


def ema_seed_sma(values, period):
    """Textbook EMA: seeded from an SMA of the first `period` values."""
    if len(values) < period:
        return []
    alpha = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out = [(period - 1, prev)]
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out.append((i, prev))
    return out


def macd(closes, fast, slow, signal, ema_fn):
    """Returns (bar_index, macd, signal, histogram) using the given EMA convention."""
    fast_ema = dict(ema_fn(closes, fast))
    slow_ema = dict(ema_fn(closes, slow))
    idx = sorted(set(fast_ema) & set(slow_ema))
    if not idx:
        return []
    macd_line = [fast_ema[i] - slow_ema[i] for i in idx]
    out = []
    for pos, sig in ema_fn(macd_line, signal):
        i = idx[pos]
        m = fast_ema[i] - slow_ema[i]
        out.append((i, m, sig, m - sig))
    return out


# --------------------------------------------------------------------------

def epoch_ms(iso):
    """Bar timestamps arrive as ISO-8601 instants; indicator points as epoch ms."""
    s = iso.replace("Z", "+00:00")
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)


def compare(label, server, local, times):
    """server/local are [(bar_index, value)]. Aligned on bar index."""
    lookup = dict(local)
    diffs = [(i, abs(v - lookup[i]), v, lookup[i]) for i, v in server if i in lookup]
    if not diffs:
        print(f"    {label:34s}  no overlapping points")
        return
    i, worst, sv, lv = max(diffs, key=lambda d: d[1])
    mean = sum(d[1] for d in diffs) / len(diffs)
    verdict = "MATCH  " if worst < 1e-6 else "close  " if worst < 0.01 else "DIFFERS"
    print(f"    {label:34s}  {verdict}  max {worst:11.6f}  mean {mean:11.6f}"
          f"   worst at bar {i} ({times[i]}): server {sv:.4f} local {lv:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Compare CalcServer indicators against local implementations")
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--bars", type=int, default=300)
    ap.add_argument("--session", default=None, help="regular | extended")
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--macd", default="12,26,9", help="fast,slow,signal")
    ap.add_argument("--timeout", type=int, default=20000, help="ms")
    args = ap.parse_args()

    fast, slow, signal = (int(x) for x in args.macd.split(","))

    print("Discovering CalcServer...")
    calc = ServiceLocator.wait_for_service(service_name=ServiceLocator.CALC, timeout_sec=30)

    request = {
        "op": "get_chart_data",
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "barCount": args.bars,
        "atrPeriod": args.atr_period,
        "rsiPeriod": args.rsi_period,
        "macdFast": fast,
        "macdSlow": slow,
        "macdSignal": signal,
    }
    if args.session:
        request["session"] = args.session

    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, args.timeout)
    dealer.connect(calc.router)

    print(f"Requesting {args.symbol.upper()} {args.timeframe}, {args.bars} bars, "
          f"ATR({args.atr_period}), RSI({args.rsi_period}), MACD({fast},{slow},{signal})")
    dealer.send_string(json.dumps(request))

    if not dealer.poll(timeout=args.timeout):
        print("No reply from CalcServer. Is it running?")
        return 1

    reply = json.loads(dealer.recv_string())
    if not reply.get("ok", False):
        print(f"Request failed: {reply.get('error')}")
        return 1

    data = reply.get("data", {})
    if isinstance(data.get("data"), dict):
        data = data["data"]

    bars = data.get("bars") or []
    if not bars:
        print("No bars returned, nothing to compare.")
        return 1

    closes = [b["c"] for b in bars]
    times = [b["t"] for b in bars]
    index_of = {epoch_ms(b["t"]): i for i, b in enumerate(bars)}
    print(f"Server returned {len(bars)} bars, {times[0]} .. {times[-1]}\n")

    def align(points, key):
        """Map server points onto bar indices by timestamp, not by position."""
        out, unmatched = [], 0
        for p in points:
            i = index_of.get(p["timestamp"])
            if i is None:
                unmatched += 1
            else:
                out.append((i, p[key]))
        if unmatched:
            print(f"    ({unmatched} server points had no matching bar timestamp)")
        return out

    server_atr = data.get("atr") or []
    if server_atr:
        print(f"  ATR({args.atr_period}) — {len(server_atr)} server points")
        aligned = align(server_atr, "atr")
        compare("Wilder, TR from bar 1", aligned,
                atr_wilder(bars, args.atr_period), times)
        compare("Wilder, synthetic TR[0]", aligned,
                atr_wilder(bars, args.atr_period, synthetic_first=True), times)
        compare("sliding rolling mean", aligned,
                atr_sliding_mean(bars, args.atr_period), times)
        print()
    else:
        print("  ATR — server returned none\n")

    server_rsi = data.get("rsi") or []
    if server_rsi:
        print(f"  RSI({args.rsi_period}) — {len(server_rsi)} server points")
        aligned = align(server_rsi, "rsi")
        compare("Wilder smoothing (standard)", aligned, rsi_wilder(closes, args.rsi_period), times)
        compare("simple rolling mean", aligned, rsi_simple(closes, args.rsi_period), times)
        print()
    else:
        print("  RSI — server returned none\n")

    server_macd = data.get("macd") or []
    if server_macd:
        print(f"  MACD({fast},{slow},{signal}) — {len(server_macd)} server points")
        for name, key, pos in (("macd line", "macd", 1),
                               ("signal line", "signal", 2),
                               ("histogram", "histogram", 3)):
            aligned = align(server_macd, key)
            for variant, ema_fn in (("EMA seeded from SMA", ema_seed_sma),
                                    ("EMA seeded from first close", ema_seed_first)):
                local = [(t[0], t[pos]) for t in macd(closes, fast, slow, signal, ema_fn)]
                compare(f"{name} / {variant}", aligned, local, times)
            print()
    else:
        print("  MACD — server returned none")

    print("Reading the result:")
    print("  MATCH    the server implements that convention exactly")
    print("  close    same convention, accumulated floating-point drift only")
    print("  DIFFERS  a different convention, or different seeding of the same one")
    print()
    print("  If neither variant matches, compare the earliest points: seeding")
    print("  differences are largest at the left edge and converge to the right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
