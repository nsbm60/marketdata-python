"""CLI: one-shot per-contract greeks for an underlying + expiry (near-close use).

    python -m greeks.snapshot NVDA 2026-05-15
    python -m greeks.snapshot NVDA 2026-05-15 --as-of 2026-05-14 --band 0.15
    python -m greeks.snapshot NVDA 2026-05-15 --sofr-csv path/to/sofr_daily.csv

Pure orchestration of certified parts — no new pricing math:

  chain       greeks.pull.contracts.fetch_massive_contracts + filter_contracts
  spot        greeks.pull.alpaca_spot.fetch_equity_trades + spot_asof (raw SIP)
  price       greeks.pull.massive_trades.fetch_option_trades_day -> last print
  carry       greeks.forwards.carry via greeks.harness.invert_trades.invert_batch
  greeks      greeks.solver.invert (Black-76) via the same invert_batch

No SQLite staging, no broker, no ClickHouse. One shot, print a table, exit.

**Price source.** The certified option pull returns trade PRINTS, not quotes, so
there is no bid/ask mid available through these parts. The price inverted is each
contract's last print of the session; its own timestamp is shown per row so a
stale print near the close is visible rather than hidden.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import httpx

from greeks.config import GreeksConfig, load_config
from greeks.domain import RowStatus
from greeks.forwards.sofr import fetch_fred_sofr, load_sofr_csv, series_from_observations
from greeks.harness.invert_trades import invert_batch
from greeks.harness.rows import ValidationRow
from greeks.pull.alpaca_spot import (
    EquityTradePrint,
    fetch_equity_trades,
    make_stock_client,
    spot_asof,
)
from greeks.pull.contracts import (
    DEFAULT_MAX_DTE_DAYS,
    DEFAULT_MONEYNESS_BAND,
    ContractRef,
    fetch_massive_contracts,
    filter_contracts,
)
from greeks.pull.massive_trades import OptionTradePrint, fetch_option_trades_day
from greeks.pull.staging import StagedTrade

_ET = ZoneInfo("America/New_York")

# Price column label — single word, stored per run (certified pull is trades-only).
PRICE_SOURCE = "trade"


# ── pure helpers (no network; unit-tested) ─────────────────────────────────────
def select_last_print(trades: Sequence[OptionTradePrint]) -> Optional[OptionTradePrint]:
    """Most recent print of the session. ``fetch_option_trades_day`` returns
    ascending by SIP clock, so the last element is the near-close print."""
    return trades[-1] if trades else None


def staged_from_print(
    contract: ContractRef,
    last: OptionTradePrint,
    *,
    session_date: date,
    underlying: str,
    spot: float,
    spot_ts: datetime,
) -> StagedTrade:
    """Build the in-memory StagedTrade that the certified invert path consumes.

    We never touch SQLite — StagedTrade is a plain dataclass, so the snapshot
    feeds ``invert_batch`` exactly as the pull→invert pipeline would have.
    """
    return StagedTrade(
        symbol=contract.osi,
        trade_ts=last.trade_ts,
        session_date=session_date,
        underlying=underlying.upper(),
        price=last.price,
        size=last.size,
        exchange=last.exchange,
        conditions="",
        sequence_number=last.sequence_number,
        sip_timestamp_ns=last.sip_timestamp_ns,
        spot_at_trade=spot,
        spot_trade_ts=spot_ts,
    )


def _et_clock(ts: Optional[datetime]) -> str:
    return "—" if ts is None else ts.astimezone(_ET).strftime("%H:%M:%S")


def _et_full(ts: datetime) -> str:
    return ts.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S ET")


def _num(v: Optional[float], width: int, dec: int) -> str:
    return "—".rjust(width) if v is None else f"{v:>{width}.{dec}f}"


def format_rows(rows: Sequence[ValidationRow]) -> str:
    """Per-contract table: strike, side, greeks, and the inputs used.

    Includes failed inversions (with their reason as status) — no silent drop.
    Sorted by strike then side.
    """
    cols = (
        f"{'strike':>9} {'side':>4} {'price':>8} {'pxTime':>9} "
        f"{'fwd':>9} {'disc':>8} {'T(yr)':>8} "
        f"{'iv':>8} {'delta':>8} {'gamma':>8} {'theta':>9} {'vega':>8} {'rho':>8}  status"
    )
    lines = [cols, "-" * len(cols)]
    for r in sorted(rows, key=lambda x: (x.strike, x.right.value)):
        status = "ok" if r.status is RowStatus.SUCCESS else (
            r.reason_code.value if r.reason_code else "failure"
        )
        lines.append(
            f"{r.strike:>9.2f} {r.right.value:>4} "
            f"{_num(r.trade_price, 8, 2)} {_et_clock(r.trade_ts):>9} "
            f"{_num(r.forward, 9, 2)} {_num(r.discount, 8, 5)} {_num(r.time_to_expiry, 8, 5)} "
            f"{_num(r.iv, 8, 4)} {_num(r.delta, 8, 4)} {_num(r.gamma, 8, 4)} "
            f"{_num(r.theta, 9, 4)} {_num(r.vega, 8, 4)} {_num(r.rho, 8, 4)}  {status}"
        )
    return "\n".join(lines)


# ── SOFR source (live FRED by default, offline CSV override) ───────────────────
def load_sofr_series(
    cfg: GreeksConfig, as_of: date, sofr_csv: Optional[Path]
) -> dict[date, float]:
    """Return a ``date -> rate`` series for carry. CSV if given, else live FRED.

    FRED publishes the prior day's SOFR next morning, so a same-day near-close
    run has no observation for today; carry forward-fills from the most recent
    fix (``rate_on_or_before``), which is correct.
    """
    if sofr_csv is not None:
        return load_sofr_csv(sofr_csv)
    key = cfg.api_keys.fred_api_key
    if not key:
        raise SystemExit(
            "no SOFR source: set FRED_API_KEY for the live FRED pull, "
            "or pass --sofr-csv <path> for an offline series"
        )
    obs = fetch_fred_sofr(
        key, observation_start=as_of - timedelta(days=14), observation_end=as_of
    )
    series = series_from_observations(obs)
    if not series:
        raise SystemExit(
            f"FRED returned no SOFR observations in "
            f"[{(as_of - timedelta(days=14)).isoformat()}, {as_of.isoformat()}]"
        )
    return series


# ── orchestration ──────────────────────────────────────────────────────────────
def _spot_cutoff(as_of: date, now_utc: datetime) -> datetime:
    """As-of instant for spot: now for a current-day run, else end of the ET day."""
    if as_of >= now_utc.astimezone(_ET).date():
        return now_utc
    end_of_day_et = datetime.combine(as_of + timedelta(days=1), dtime(0, 0), tzinfo=_ET)
    return end_of_day_et.astimezone(timezone.utc)


def run_snapshot(
    underlying: str,
    expiry: date,
    *,
    as_of: date,
    cfg: GreeksConfig,
    band: float,
    sofr_csv: Optional[Path],
) -> int:
    und = underlying.upper()
    dte = (expiry - as_of).days
    if dte < 0:
        raise SystemExit(f"expiry {expiry.isoformat()} is before as_of {as_of.isoformat()}")

    massive_key = cfg.api_keys.massive_api_key
    if not massive_key:
        raise SystemExit("no Massive key: set MASSIVE_API_KEY (or POLYGON_API_KEY)")
    alpaca_key, alpaca_secret = cfg.api_keys.alpaca_api_key, cfg.api_keys.alpaca_api_secret
    if not alpaca_key or not alpaca_secret:
        raise SystemExit("no Alpaca creds: set ALPACA_API_KEY and ALPACA_API_SECRET")

    sofr = load_sofr_series(cfg, as_of, sofr_csv)  # fail fast before any pull

    snapshot_ts = datetime.now(timezone.utc)
    cutoff = _spot_cutoff(as_of, snapshot_ts)

    # Spot: raw SIP equity print as of the cutoff (certified pull).
    stock = make_stock_client(alpaca_key, alpaca_secret)
    equity: list[EquityTradePrint] = fetch_equity_trades(stock, und, as_of)
    spot_print = spot_asof(equity, cutoff)
    if spot_print is None:
        raise SystemExit(
            f"no SIP equity print for {und} on {as_of.isoformat()} at/before "
            f"{_et_full(cutoff)}"
        )
    spot = spot_print.price

    # Chain: certified enumeration + filter, restricted to the requested expiry.
    max_dte = max(DEFAULT_MAX_DTE_DAYS, dte)
    expired = expiry < snapshot_ts.astimezone(_ET).date()
    contracts = fetch_massive_contracts(
        massive_key, und, as_of=as_of, max_dte_days=max_dte, expired=expired
    )
    same_expiry = [c for c in contracts if c.expiry == expiry]
    if not same_expiry:
        raise SystemExit(
            f"no listed contracts for {und} {expiry.isoformat()} (as_of {as_of.isoformat()})"
        )
    filt = filter_contracts(
        same_expiry, as_of=as_of, spot=spot, max_dte_days=max_dte, moneyness_band=band
    )
    if not filt.eligible:
        raise SystemExit(
            f"no eligible contracts after filters (band +/-{band:.0%}, spot {spot:.2f}); "
            f"skipped_moneyness={filt.skipped_moneyness} nonstandard={len(filt.nonstandard)}"
        )

    # Per-contract last print (one shared HTTP client).
    staged: list[StagedTrade] = []
    no_trade: list[ContractRef] = []
    with httpx.Client(timeout=60.0) as http:
        for c in filt.eligible:
            prints = fetch_option_trades_day(massive_key, c.osi, as_of, client=http)
            last = select_last_print(prints)
            if last is None:
                no_trade.append(c)
                continue
            staged.append(
                staged_from_print(
                    c, last,
                    session_date=as_of, underlying=und,
                    spot=spot, spot_ts=spot_print.trade_ts,
                )
            )

    rows, stats = invert_batch(staged, cfg=cfg, sofr_series=sofr)

    print(f"greeks snapshot  {und}  expiry {expiry.isoformat()}  (methodology {cfg.methodology_version})")
    print(f"  session as_of : {as_of.isoformat()}   snapshot: {_et_full(snapshot_ts)}")
    print(f"  spot          : {spot:.4f}  @ {_et_clock(spot_print.trade_ts)} ET  (raw SIP)")
    print(f"  price source  : {PRICE_SOURCE} (last print of session; certified pull has no bid/ask mid)")
    print(
        f"  moneyness band: +/-{band:.0%}   eligible={len(filt.eligible)}  "
        f"priced={len(staged)}  no_trade={len(no_trade)}"
    )
    print(
        f"  filtered out  : dte={filt.skipped_dte}  moneyness={filt.skipped_moneyness}  "
        f"nonstandard={len(filt.nonstandard)}"
    )
    print()
    print(format_rows(rows))

    if no_trade:
        print()
        print(f"no trade on {as_of.isoformat()} ({len(no_trade)} contracts, no price to invert):")
        for c in sorted(no_trade, key=lambda x: (x.strike, x.right.value)):
            print(f"  {c.osi}  {c.strike:>9.2f} {c.right.value}")

    print()
    print(
        f"invert: priced={stats.n_input} ok={stats.n_success} fail={stats.n_failure} "
        f"reasons={dict(stats.by_reason)}"
    )
    return 0


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="greeks_snapshot",
        description="One-shot per-contract greeks for an underlying + expiry (near-close use).",
    )
    p.add_argument("underlying", type=str, help="e.g. NVDA")
    p.add_argument("expiry", type=_parse_date, help="option expiry, YYYY-MM-DD")
    p.add_argument(
        "--as-of", type=_parse_date, default=None,
        help="snapshot/session date (default: today)",
    )
    p.add_argument(
        "--band", type=float, default=DEFAULT_MONEYNESS_BAND,
        help=f"moneyness band K/S in [1-b, 1+b] (default {DEFAULT_MONEYNESS_BAND})",
    )
    p.add_argument(
        "--sofr-csv", type=Path, default=None,
        help="offline SOFR date,rate CSV (default: live FRED via FRED_API_KEY)",
    )
    p.add_argument("--config", type=Path, default=None, help="greeks config path override")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    as_of: date = args.as_of if args.as_of is not None else date.today()
    return run_snapshot(
        args.underlying, args.expiry,
        as_of=as_of, cfg=cfg, band=args.band, sofr_csv=args.sofr_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
