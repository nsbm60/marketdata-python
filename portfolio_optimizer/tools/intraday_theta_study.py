#!/usr/bin/env python3
"""
Intraday theta study — first cut.

Question: does a delta-neutral short strangle on NVDA weeklies, entered
mid-morning and closed mid-afternoon (flat by close, never held overnight),
harvest enough theta to survive round-trip spread — and is the result a
consistent theta harvest or a path-dominated gamble wearing a theta costume?

This first cut runs ONE configuration family (enter 10:00, exit 15:30, static,
0.30-delta legs) so we can see whether the number is even in an interesting
range before building the full delta x DTE x entry-time sweep and the
re-hedged comparison.

Expiry selection (IMPORTANT): NVDA weeklies expire on FRIDAYS only. We do NOT
compute a target expiry by counting N days forward — that lands on non-Friday
dates that have no contracts. Instead we look up the expiries that actually
exist for each entry day and pick the one whose trading-day distance falls in
[DTE_MIN, DTE_MAX], nearest DTE_TARGET. The REALIZED DTE varies by weekday
(Mon~4, Tue~3, Wed~2, Thu often none, Fri~5) and is RECORDED per trade so the
distribution can be segmented by it.

Design decisions baked in (see conversation for the reasoning):
  - NO 0DTE, never rides into expiry (DTE_MIN >= 2).
  - 0.30-delta short strangle, closest-available strikes each side, matched
    one-for-one. "Neutral" = closest-strikes neutral; actual entry net delta
    is RECORDED per trade (residual delta is a real source of directional P&L).
  - Conservative spread: SELL the bid on both legs at entry, BUY BACK the ask
    on both legs at exit. Full round-trip spread, taker assumption.
  - Static: enter neutral, hold untouched, close. Measures what a passive
    flat-by-close strangle actually earns, gamma bleed included. The re-hedged
    version is the SECOND iteration, once static shows something worth decomposing.
  - Greek sanity guards: stored Greeks are from a provider with KNOWN quality
    problems (no Greeks on expiration days, garbage premarket IV). We avoid
    expiry entirely and enter at 10:00, but still validate entry Greeks per day
    and skip days where they look bad.

USAGE:
    From the repo root:  PYTHONPATH=. python portfolio_optimizer/tools/intraday_theta_study.py

Standalone analysis artifact (like the Massive eval scripts), not part of the
optimizer service. Reads snapshots, simulates, prints a distribution.
"""

import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from statistics import mean, median, pstdev

from ml.shared.clickhouse import get_ch_client

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────

# ClickHouse connection is resolved via service discovery (see get_client),
# the same way the rest of the repo finds services.
CH_TABLE = "option_snapshot"

UNDERLYING = "NVDA"

# Study window (inclusive).
START_DATE = date(2025, 11, 1)
END_DATE = date(2026, 5, 1)

# Expiry selection by trading-day distance. We pick a REAL (Friday) expiry whose
# distance is in [DTE_MIN, DTE_MAX], nearest DTE_TARGET. Realized DTE recorded.
DTE_TARGET = 3
DTE_MIN = 2                  # >= 2 keeps us off 0DTE/1DTE expiry chaos
DTE_MAX = 5
DTE_BASIS = "trading"        # "trading" or "calendar"

TARGET_DELTA = 0.30          # each leg; closest available strike is selected
ENTRY_TIME_ET = time(10, 0)
EXIT_TIME_ET = time(15, 30)

TIME_TOLERANCE_MIN = 10      # snapshot must be within this many minutes of target

# Greek sanity guards for the entry snapshot.
IV_MIN, IV_MAX = 0.05, 3.0
MIN_BID = 0.05
MIN_OPEN_INTEREST = 0        # set > 0 to require liquidity

ET = timezone(timedelta(hours=-4))  # NOTE: fixed EDT offset; DST boundary not handled (see caveat)

CONTRACT_MULTIPLIER = 100


# ─────────────────────────────────────────────────────────────
# OCC symbol parsing
# ─────────────────────────────────────────────────────────────
# Standard OCC/OSI symbol: ROOT + YYMMDD + {C|P} + strike*1000 (8 digits).
# Example: NVDA260417P00180000 -> NVDA, 2026-04-17, Put, 180.000

@dataclass
class OptionContract:
    symbol: str
    root: str
    expiry: date
    right: str       # "C" or "P"
    strike: float


def parse_occ(symbol: str) -> OptionContract:
    i = 0
    while i < len(symbol) and not symbol[i].isdigit():
        i += 1
    root = symbol[:i]
    yymmdd = symbol[i:i + 6]
    right = symbol[i + 6]
    strike_raw = symbol[i + 7:]
    expiry = datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike_raw) / 1000.0
    if right not in ("C", "P"):
        raise ValueError(f"bad right in {symbol!r}: {right!r}")
    return OptionContract(symbol, root, expiry, right, strike)


# ─────────────────────────────────────────────────────────────
# ClickHouse access
# ─────────────────────────────────────────────────────────────

def get_client():
    """ClickHouse client resolved through service discovery (repo-canonical helper)."""
    return get_ch_client()


def fq_table() -> str:
    return CH_TABLE


def trading_days(start: date, end: date) -> list:
    """Business days (Mon-Fri). Does NOT account for market holidays — acceptable
    for a first cut; a holiday only mislabels realized DTE by one, never causes a
    miss now that expiry is selected from real contracts."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def trading_day_distance(d0: date, d1: date) -> int:
    """Business days from d0 (exclusive) to d1 (inclusive). Holiday caveat as above."""
    if d1 <= d0:
        return 0
    count = 0
    cur = d0
    while cur < d1:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            count += 1
    return count


def calendar_distance(d0: date, d1: date) -> int:
    return (d1 - d0).days


def available_expiries(client, entry_day: date) -> list:
    """Distinct expiries (as dates) present for the underlying on entry_day,
    parsed from the symbols. The US RTH trading day sits within one UTC date,
    so toDate(timestamp) == entry_day selects that session."""
    q = f"""
        SELECT DISTINCT symbol
        FROM {fq_table()}
        WHERE underlying = %(u)s
          AND toDate(timestamp) = %(d)s
        """
    rows = client.query(
        q, parameters={"u": UNDERLYING, "d": entry_day.strftime("%Y-%m-%d")}
    ).named_results()
    expiries = set()
    for r in rows:
        try:
            expiries.add(parse_occ(r["symbol"]).expiry)
        except Exception:
            continue
    return sorted(expiries)


def select_expiry(entry_day: date, expiries: list):
    """Pick the real expiry whose distance is in [DTE_MIN, DTE_MAX], nearest
    DTE_TARGET. Returns (expiry, realized_dte) or (None, None) if none qualify
    (e.g. Thursdays, where this Friday is 1 DTE and next Friday is 6)."""
    dist = trading_day_distance if DTE_BASIS == "trading" else calendar_distance
    candidates = []
    for e in expiries:
        n = dist(entry_day, e)
        if DTE_MIN <= n <= DTE_MAX:
            candidates.append((abs(n - DTE_TARGET), n, e))
    if not candidates:
        return None, None
    candidates.sort()
    _, realized_dte, expiry = candidates[0]
    return expiry, realized_dte


def snapshot_near(client, entry_day: date, target_t: time, expiry: date) -> list:
    """Option rows for `expiry` whose timestamp on `entry_day` is closest to
    target_t (within TIME_TOLERANCE_MIN), nearest row per symbol."""
    target_dt = datetime.combine(entry_day, target_t, tzinfo=ET).astimezone(timezone.utc)
    lo = (target_dt - timedelta(minutes=TIME_TOLERANCE_MIN)).strftime("%Y-%m-%d %H:%M:%S")
    hi = (target_dt + timedelta(minutes=TIME_TOLERANCE_MIN)).strftime("%Y-%m-%d %H:%M:%S")

    q = f"""
        SELECT symbol, timestamp, bid, ask, last_price,
               delta, gamma, theta, vega, iv, open_interest
        FROM {fq_table()}
        WHERE underlying = %(u)s
          AND timestamp BETWEEN %(lo)s AND %(hi)s
        """
    rows = client.query(q, parameters={"u": UNDERLYING, "lo": lo, "hi": hi}).named_results()

    best = {}
    exp_str = expiry.strftime("%y%m%d")
    for r in rows:
        sym = r["symbol"]
        if exp_str not in sym:
            continue
        try:
            c = parse_occ(sym)
        except Exception:
            continue
        if c.expiry != expiry:
            continue
        ts = r["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        d = abs((ts - target_dt).total_seconds())
        if sym not in best or d < best[sym][0]:
            best[sym] = (d, r, c)
    return [(r, c) for (_, r, c) in best.values()]


# ─────────────────────────────────────────────────────────────
# Strike selection + sanity
# ─────────────────────────────────────────────────────────────

def _f(x):
    return None if x is None else float(x)


def entry_greeks_ok(rows) -> bool:
    usable = [r for (r, _) in rows
              if _f(r["bid"]) is not None and _f(r["bid"]) >= MIN_BID
              and r["delta"] is not None and r["iv"] is not None]
    if len(usable) < 6:
        return False
    ivs = [_f(r["iv"]) for r in usable]
    if min(ivs) < IV_MIN or max(ivs) > IV_MAX:
        return False
    return True


def pick_leg(rows, right: str, target_delta: float):
    best = None
    for (r, c) in rows:
        if c.right != right:
            continue
        d = r["delta"]
        bid = _f(r["bid"])
        if d is None or bid is None or bid < MIN_BID:
            continue
        if MIN_OPEN_INTEREST > 0 and (r["open_interest"] or 0) < MIN_OPEN_INTEREST:
            continue
        score = abs(abs(float(d)) - target_delta)
        if best is None or score < best[0]:
            best = (score, r, c)
    return None if best is None else (best[1], best[2])


# ─────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_day: date
    expiry: date
    dte: int
    put_strike: float
    call_strike: float
    entry_net_delta: float
    entry_credit: float
    exit_debit: float
    pnl: float
    expected_theta: float


def simulate_day(client, entry_day: date) -> Trade:
    expiries = available_expiries(client, entry_day)
    if not expiries:
        return None
    expiry, realized_dte = select_expiry(entry_day, expiries)
    if expiry is None:
        return None  # no expiry in the DTE band (e.g. Thursdays)

    entry_rows = snapshot_near(client, entry_day, ENTRY_TIME_ET, expiry)
    if not entry_rows or not entry_greeks_ok(entry_rows):
        return None

    put = pick_leg(entry_rows, "P", TARGET_DELTA)
    call = pick_leg(entry_rows, "C", TARGET_DELTA)
    if put is None or call is None:
        return None
    put_row, put_c = put
    call_row, call_c = call

    put_bid = _f(put_row["bid"])
    call_bid = _f(call_row["bid"])
    if put_bid is None or call_bid is None:
        return None
    entry_credit = (put_bid + call_bid) * CONTRACT_MULTIPLIER

    entry_net_delta = (float(put_row["delta"]) + float(call_row["delta"])) * CONTRACT_MULTIPLIER

    hold_hours = (datetime.combine(entry_day, EXIT_TIME_ET)
                  - datetime.combine(entry_day, ENTRY_TIME_ET)).total_seconds() / 3600.0
    theta_sum = (float(put_row["theta"]) + float(call_row["theta"]))
    expected_theta = -theta_sum * CONTRACT_MULTIPLIER * (hold_hours / 24.0)

    exit_rows = snapshot_near(client, entry_day, EXIT_TIME_ET, expiry)
    exit_by_sym = {r["symbol"]: r for (r, _) in exit_rows}
    pr = exit_by_sym.get(put_row["symbol"])
    cr = exit_by_sym.get(call_row["symbol"])
    if pr is None or cr is None:
        return None
    put_ask = _f(pr["ask"])
    call_ask = _f(cr["ask"])
    if put_ask is None or call_ask is None:
        return None
    exit_debit = (put_ask + call_ask) * CONTRACT_MULTIPLIER

    pnl = entry_credit - exit_debit
    return Trade(
        entry_day=entry_day, expiry=expiry, dte=realized_dte,
        put_strike=put_c.strike, call_strike=call_c.strike,
        entry_net_delta=entry_net_delta,
        entry_credit=entry_credit, exit_debit=exit_debit,
        pnl=pnl, expected_theta=expected_theta,
    )


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────

def _summary(pnls):
    wins = [p for p in pnls if p > 0]
    return (f"n={len(pnls):<4} mean={mean(pnls):+8.2f}  median={median(pnls):+8.2f}  "
            f"stdev={pstdev(pnls):7.2f}  win%={len(wins)/len(pnls)*100:4.1f}  "
            f"worst={min(pnls):+8.2f}  best={max(pnls):+8.2f}")


def report(trades: list):
    if not trades:
        print("No qualifying trades. Check date range, snapshot coverage near "
              "the entry/exit times, the DTE band, and the Greek sanity guards.")
        return
    pnls = [t.pnl for t in trades]
    thetas = [t.expected_theta for t in trades]

    print("=" * 72)
    print(f"Intraday theta study — {UNDERLYING}  ({TARGET_DELTA:.2f}-delta strangle, "
          f"DTE {DTE_MIN}-{DTE_MAX} {DTE_BASIS}, target {DTE_TARGET})")
    print(f"Enter {ENTRY_TIME_ET.strftime('%H:%M')} ET, exit "
          f"{EXIT_TIME_ET.strftime('%H:%M')} ET, static, full round-trip spread")
    print(f"Window {START_DATE} .. {END_DATE}")
    print("=" * 72)
    print("ALL:    " + _summary(pnls))
    print(f"Total P&L: {sum(pnls):+.2f}   (per 1-lot strangle, $)")
    print("-" * 72)

    # Segment by realized DTE — the comparison we actually want.
    by_dte = {}
    for t in trades:
        by_dte.setdefault(t.dte, []).append(t.pnl)
    for dte in sorted(by_dte):
        print(f"DTE {dte}:  " + _summary(by_dte[dte]))
    print("-" * 72)

    print(f"Mean expected theta (baseline):  {mean(thetas):+.2f}")
    print(f"Mean realized P&L:               {mean(pnls):+.2f}")
    print(f"Mean theta NOT kept (gamma+vega bleed + spread): "
          f"{mean(thetas) - mean(pnls):+.2f}")
    print(f"Mean |entry residual delta|: {mean(abs(t.entry_net_delta) for t in trades):.1f} "
          f"(share-equiv)")
    print("=" * 72)
    print("Read: mean tells you if edge exists; stdev and worst-day tell you if "
          "it's theta or a directional gamble. If realized P&L << expected theta "
          "with large stdev, gamma is eating the decay.")

    print("\nPer-trade detail:")
    print(f"{'date':<12}{'exp':<12}{'dte':>4}{'P_K':>7}{'C_K':>7}{'netD':>8}"
          f"{'credit':>9}{'debit':>9}{'pnl':>9}{'exp_th':>9}")
    for t in sorted(trades, key=lambda x: x.entry_day):
        print(f"{t.entry_day.isoformat():<12}{t.expiry.isoformat():<12}{t.dte:>4}"
              f"{t.put_strike:>7.1f}{t.call_strike:>7.1f}{t.entry_net_delta:>8.1f}"
              f"{t.entry_credit:>9.2f}{t.exit_debit:>9.2f}{t.pnl:>+9.2f}"
              f"{t.expected_theta:>+9.2f}")


def main():
    client = get_client()
    trades = []
    skipped = 0
    for d in trading_days(START_DATE, END_DATE):
        try:
            t = simulate_day(client, d)
        except Exception as e:
            print(f"[warn] {d}: {type(e).__name__}: {e}", file=sys.stderr)
            t = None
        if t is None:
            skipped += 1
        else:
            trades.append(t)
    print(f"(simulated {len(trades)} days, skipped {skipped})\n")
    report(trades)


if __name__ == "__main__":
    main()
