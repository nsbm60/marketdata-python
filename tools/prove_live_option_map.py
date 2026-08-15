#!/usr/bin/env python3
"""
Prove whether CalcServer liveOptionPrices[osi] matches MDS for the same OSI.

1) Read map slots via Calc control op: debug_live_option_prices
2) Sample MDS ZMQ quotes/trades for those OSIs
3) Verdict per OSI: MAP_OK | MAP_STALE_OR_WRONG | NO_MDS_QUOTE | MISSING_SLOT

Requires CalcServer rebuilt/restarted with LiveOptionPriceDebugHandler registered.

Usage:
  ./tools/venv/bin/python tools/prove_live_option_map.py
  ./tools/venv/bin/python tools/prove_live_option_map.py \\
      --symbols NVDA260810P00220000,NVDA260810P00222500,NVDA260810P00225000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import zmq

# Hierarchical topic from OSI (same rules as TopicBuilder.formatStrike / optionSuffix)
def osi_to_quote_topic(osi: str) -> str:
    osi = osi.strip().upper()
    # UNDERLYING + YYMMDD + C/P + 8-digit strike
    i = 0
    while i < len(osi) and osi[i].isalpha():
        i += 1
    und = osi[:i]
    yymmdd = osi[i : i + 6]
    side = osi[i + 6]
    strike_milli = int(osi[i + 7 : i + 15])
    strike = strike_milli / 1000.0
    cents = int(round(strike * 100))
    dollars, rem = divmod(cents, 100)
    yy, mm, dd = yymmdd[0:2], yymmdd[2:4], yymmdd[4:6]
    year = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)
    expiry = f"{year:04d}-{mm}-{dd}"
    return f"md.option.quote.{und}.{expiry}.{side}.{dollars}_{rem:02d}"


def osi_to_trade_topic(osi: str) -> str:
    return osi_to_quote_topic(osi).replace(".quote.", ".trade.", 1)


def calc_control(endpoint: str, payload: dict, timeout_ms: int = 10000) -> dict:
    ctx = zmq.Context.instance()
    d = ctx.socket(zmq.DEALER)
    d.setsockopt(zmq.LINGER, 0)
    d.setsockopt(zmq.RCVTIMEO, timeout_ms)
    d.setsockopt(zmq.SNDTIMEO, 5000)
    d.connect(endpoint)
    try:
        d.send_string(json.dumps(payload))
        # Calc may reply single or multipart
        parts = d.recv_multipart()
        raw = parts[-1].decode()
        return json.loads(raw)
    finally:
        d.close()


def sample_mds(osis: list[str], seconds: float = 5.0) -> dict[str, dict[str, Any]]:
    """Return per-OSI last quote/trade from MDS PUB."""
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:6006")
    for osi in osis:
        sub.setsockopt_string(zmq.SUBSCRIBE, osi_to_quote_topic(osi))
        sub.setsockopt_string(zmq.SUBSCRIBE, osi_to_trade_topic(osi))
    sub.RCVTIMEO = 200
    out: dict[str, dict[str, Any]] = {o: {} for o in osis}
    end = time.time() + seconds
    while time.time() < end:
        try:
            topic = sub.recv_string()
            payload = sub.recv_string()
            o = json.loads(payload)
            sym = (o.get("symbol") or "").upper()
            if sym not in out:
                continue
            if ".quote." in topic:
                out[sym]["bid"] = o.get("bid")
                out[sym]["ask"] = o.get("ask")
                out[sym]["quote_topic"] = topic
                out[sym]["quote_t"] = time.time()
            elif ".trade." in topic:
                out[sym]["last"] = o.get("price")
                out[sym]["trade_topic"] = topic
                out[sym]["trade_t"] = time.time()
        except zmq.Again:
            continue
    sub.close()
    return out


def near(a, b, eps=0.08) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= eps
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove liveOptionPrices map vs MDS")
    ap.add_argument(
        "--symbols",
        default="NVDA260810P00220000,NVDA260810P00222500,NVDA260810P00225000",
        help="Comma-separated OSI list",
    )
    ap.add_argument("--calc-control", default="tcp://127.0.0.1:6010", help="Calc ROUTER endpoint")
    ap.add_argument("--mds-seconds", type=float, default=6.0)
    args = ap.parse_args()
    osis = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print("=== 1) MAP DUMP (liveOptionPrices via debug_live_option_prices) ===")
    print(f"calc control: {args.calc_control}")
    try:
        resp = calc_control(
            args.calc_control,
            {"op": "debug_live_option_prices", "symbols": osis},
        )
    except Exception as e:
        print(f"FAIL calling Calc: {e}")
        print("Restart CalcServer after building LiveOptionPriceDebugHandler.")
        return 1

    if not resp.get("ok"):
        print(f"FAIL op: {resp}")
        print("If unsupported op: rebuild + restart CalcServer.")
        return 1

    data = resp.get("data") or {}
    slots = {s["osi"]: s for s in data.get("slots") or []}
    print(f"asOf={data.get('asOf')}")
    for osi in osis:
        s = slots.get(osi, {})
        print(
            f"  MAP[{osi}] present={s.get('present')} tracked={s.get('tracked')} "
            f"last={s.get('last')} bid={s.get('bid')} ask={s.get('ask')} "
            f"quoteTs={s.get('quoteTs')} tradeTs={s.get('tradeTs')}"
        )

    print(f"\n=== 2) MDS ZMQ sample ({args.mds_seconds}s) ===")
    mds = sample_mds(osis, args.mds_seconds)
    for osi in osis:
        m = mds.get(osi) or {}
        print(
            f"  MDS[{osi}] bid={m.get('bid')} ask={m.get('ask')} last={m.get('last')} "
            f"qtopic={m.get('quote_topic')} ttopic={m.get('trade_topic')}"
        )

    print("\n=== 3) VERDICT (map slot vs MDS for SAME osi) ===")
    print("Criterion: MAP correct iff bid/ask/last agree with MDS for that OSI when MDS has data.")
    any_bad = False
    for osi in osis:
        s = slots.get(osi) or {}
        m = mds.get(osi) or {}
        if not s.get("present"):
            print(f"  {osi}: MISSING_SLOT (map has no entry)")
            any_bad = True
            continue
        parts = []
        # book
        if m.get("bid") is not None:
            if near(s.get("bid"), m.get("bid")) and near(s.get("ask"), m.get("ask")):
                parts.append("book=OK")
            else:
                parts.append(f"book=WRONG map={s.get('bid')}/{s.get('ask')} mds={m.get('bid')}/{m.get('ask')}")
                any_bad = True
        else:
            parts.append("book=NO_MDS_QUOTE (cannot prove book from stream this window)")
        # last
        if m.get("last") is not None:
            if near(s.get("last"), m.get("last")):
                parts.append("last=OK")
            else:
                parts.append(f"last=WRONG map={s.get('last')} mds={m.get('last')}")
                any_bad = True
        else:
            parts.append("last=NO_MDS_TRADE")
        print(f"  {osi}: " + "; ".join(parts))

    print("\n=== 4) HOW TO READ ===")
    print("  book=OK + last=OK  → map slot for that OSI is correct vs MDS (at sample time)")
    print("  book=WRONG         → liveOptionPrices[osi] bid/ask ≠ MDS for that osi (map content broken)")
    print("  NO_MDS_QUOTE       → cannot convict map book; MDS published no quote for that OSI")
    print("  MISSING_SLOT       → map has no entry for that key")
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
