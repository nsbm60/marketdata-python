"""CLI: pull one ticker-day (or drain queue) into local SQLite staging.

Usage::

    python -m greeks.pull.run --ticker NVDA --date 2026-05-27
    python -m greeks.pull.run --ticker NVDA --date 2026-05-27 --max-contracts 5
    python -m greeks.pull.run --drain-queue --queue-db /tmp/greeks_queue.db

Skips dates in config ``excluded_dates``. Writes trades+spot to staging only
(not ``greeks_validation`` — invert is PR5).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from greeks.config import GreeksConfig, load_config
from greeks.domain import WorkStatus
from greeks.pull.alpaca_spot import (
    REQUIRED_ADJUSTMENT,
    EquityTradePrint,
    assert_raw_adjustment,
    attach_spot_to_option_trades,
    fetch_equity_session_close_print,
    fetch_equity_trades,
    make_stock_client,
)
from greeks.pull.contracts import (
    DEFAULT_MAX_DTE_DAYS,
    DEFAULT_MONEYNESS_BAND,
    fetch_massive_contracts,
    filter_contracts,
)
from greeks.pull.massive_trades import fetch_option_trades_day
from greeks.pull.staging import TradeStaging
from greeks.queue.work_queue import WorkQueue


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Greeks validation data pull (PR4)")
    p.add_argument("--ticker", type=str, help="Underlying ticker (e.g. NVDA)")
    p.add_argument("--date", type=_parse_date, help="Session date YYYY-MM-DD")
    p.add_argument(
        "--queue-db",
        type=Path,
        default=Path("data/greeks_work_queue.db"),
        help="SQLite work queue path",
    )
    p.add_argument(
        "--staging-db",
        type=Path,
        default=Path("data/greeks_staging.db"),
        help="SQLite staging path for trades+spot",
    )
    p.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Cap eligible contracts (dry-run / smoke)",
    )
    p.add_argument(
        "--min-dte",
        type=int,
        default=0,
        help="Minimum DTE inclusive (default 0; use 3+ for hard-gate band smoke)",
    )
    p.add_argument(
        "--max-dte",
        type=int,
        default=None,
        help="Maximum DTE inclusive (default: package DEFAULT_MAX_DTE_DAYS)",
    )
    p.add_argument(
        "--seed-queue-only",
        action="store_true",
        help="Enumerate contracts and enqueue; do not pull trades",
    )
    p.add_argument(
        "--drain-queue",
        action="store_true",
        help="Claim and process pending queue items until empty",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override greeks.yaml path",
    )
    return p


def ensure_session_allowed(cfg: GreeksConfig, session_date: date) -> None:
    if session_date in cfg.window.excluded_dates:
        raise SystemExit(
            f"date {session_date} is in excluded_dates — skip entirely"
        )
    if not (cfg.window.start <= session_date <= cfg.window.end):
        # Allow explicit CLI outside window with warning path: still refuse by default
        raise SystemExit(
            f"date {session_date} outside validation window "
            f"[{cfg.window.start} .. {cfg.window.end}]"
        )


def pull_contract_day(
    cfg: GreeksConfig,
    *,
    contract_osi: str,
    session_date: date,
    underlying: str,
    staging: TradeStaging,
    equity_trades: list[EquityTradePrint],
) -> int:
    """Pull Massive trades for one OSI, attach spot, stage. Returns trade count."""
    key = cfg.api_keys.massive_api_key
    trades = fetch_option_trades_day(key, contract_osi, session_date)
    pairs = attach_spot_to_option_trades(trades, equity_trades, underlying)
    return staging.upsert_trades(session_date, underlying, pairs)


def seed_and_maybe_pull(
    cfg: GreeksConfig,
    *,
    ticker: str,
    session_date: date,
    queue: WorkQueue,
    staging: TradeStaging,
    max_contracts: Optional[int],
    seed_only: bool,
    min_dte_days: int = 0,
    max_dte_days: Optional[int] = None,
) -> None:
    ensure_session_allowed(cfg, session_date)
    assert_raw_adjustment()
    print(f"adjustment={REQUIRED_ADJUSTMENT.value!r} (must be raw)")

    und = ticker.upper()
    massive_key = cfg.api_keys.massive_api_key
    if not massive_key:
        raise SystemExit("MASSIVE_API_KEY / POLYGON_API_KEY not set")

    # Spot for moneyness filter: last SIP trade of the day from Alpaca RAW path.
    alpaca_key = cfg.api_keys.alpaca_api_key
    alpaca_secret = cfg.api_keys.alpaca_api_secret
    if not alpaca_key or not alpaca_secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_API_SECRET not set")

    stock_client = make_stock_client(alpaca_key, alpaca_secret)

    # Full SIP tape is only required when attaching as-of spot to option trades.
    # Seed-only needs a single late-session print for the moneyness filter.
    equity_trades: list[EquityTradePrint] = []
    if seed_only:
        close_print = fetch_equity_session_close_print(
            stock_client, und, session_date
        )
        spot = close_print.price
        print(
            f"{und} {session_date}: seed spot~{spot} "
            f"(close print @ {close_print.trade_ts.isoformat()}; "
            "full tape deferred until pull)"
        )
    else:
        equity_trades = fetch_equity_trades(stock_client, und, session_date)
        if not equity_trades:
            raise SystemExit(
                f"no Alpaca SIP equity trades for {und} on {session_date}"
            )
        spot = equity_trades[-1].price
        print(f"{und} {session_date}: {len(equity_trades)} equity trades; spot~{spot}")

    dte_hi = DEFAULT_MAX_DTE_DAYS if max_dte_days is None else max_dte_days
    raw_contracts = fetch_massive_contracts(
        massive_key,
        und,
        as_of=session_date,
        max_dte_days=dte_hi,
        expired=True,
    )
    filtered = filter_contracts(
        raw_contracts,
        as_of=session_date,
        spot=spot,
        min_dte_days=min_dte_days,
        max_dte_days=dte_hi,
        moneyness_band=DEFAULT_MONEYNESS_BAND,
    )
    print(
        f"contracts: raw={len(raw_contracts)} eligible={len(filtered.eligible)} "
        f"nonstandard={len(filtered.nonstandard)} "
        f"skip_dte={filtered.skipped_dte} skip_mny={filtered.skipped_moneyness}"
    )

    eligible = list(filtered.eligible)
    if max_contracts is not None:
        # Prefer near-ATM when smoke-capping so hard-gate (NTM) is exercisable.
        eligible.sort(key=lambda c: abs(c.strike / spot - 1.0))
        eligible = eligible[: max(0, max_contracts)]
        print(
            f"capped to max_contracts={max_contracts} → {len(eligible)} "
            f"(nearest-ATM first; strikes={[c.strike for c in eligible]})"
        )

    n_enq = queue.enqueue_many((c.osi, session_date) for c in eligible)
    print(f"enqueued {n_enq} new work items (existing left unchanged)")

    # Nonstandard: record as skipped with reason (no silent drop of eligibility)
    for c, reason in filtered.nonstandard:
        queue.enqueue(c.osi, session_date, status=WorkStatus.SKIPPED)
        queue.skip(c.osi, session_date, reason.value)

    if seed_only:
        print("seed-queue-only: done")
        return

    total_trades = 0
    for c in eligible:
        try:
            n = pull_contract_day(
                cfg,
                contract_osi=c.osi,
                session_date=session_date,
                underlying=und,
                staging=staging,
                equity_trades=equity_trades,
            )
            queue.mark_done(c.osi, session_date)
            total_trades += n
            print(f"  {c.osi}: {n} trades")
        except Exception as e:
            queue.mark_failed(c.osi, session_date, str(e))
            print(f"  {c.osi}: FAILED {e}", file=sys.stderr)

    print(
        f"done: staged_trades={staging.count(underlying=und, session_date=session_date)} "
        f"written_this_run={total_trades} queue={queue.count_by_status()}"
    )


def drain_queue(
    cfg: GreeksConfig,
    *,
    queue: WorkQueue,
    staging: TradeStaging,
) -> None:
    assert_raw_adjustment()
    # Cache equity tape per (underlying, date)
    equity_cache: dict[tuple[str, date], list[EquityTradePrint]] = {}
    stock_client = None

    while True:
        item = queue.claim_next()
        if item is None:
            print("queue empty")
            break
        try:
            from greeks.occ import parse_occ

            occ = parse_occ(item.contract)
            und = occ.root
            ensure_session_allowed(cfg, item.work_date)
            key = (und, item.work_date)
            if key not in equity_cache:
                if stock_client is None:
                    stock_client = make_stock_client(
                        cfg.api_keys.alpaca_api_key,
                        cfg.api_keys.alpaca_api_secret,
                    )
                equity_cache[key] = fetch_equity_trades(
                    stock_client, und, item.work_date
                )
            n = pull_contract_day(
                cfg,
                contract_osi=item.contract,
                session_date=item.work_date,
                underlying=und,
                staging=staging,
                equity_trades=equity_cache[key],
            )
            queue.mark_done(item.contract, item.work_date)
            print(f"done {item.contract} {item.work_date}: {n} trades")
        except Exception as e:
            queue.mark_failed(item.contract, item.work_date, str(e))
            print(f"fail {item.contract} {item.work_date}: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()

    with WorkQueue(args.queue_db) as queue, TradeStaging(args.staging_db) as staging:
        if args.drain_queue:
            drain_queue(cfg, queue=queue, staging=staging)
            return
        if not args.ticker or not args.date:
            raise SystemExit("--ticker and --date are required (or use --drain-queue)")
        seed_and_maybe_pull(
            cfg,
            ticker=args.ticker,
            session_date=args.date,
            queue=queue,
            staging=staging,
            max_contracts=args.max_contracts,
            seed_only=args.seed_queue_only,
            min_dte_days=args.min_dte,
            max_dte_days=args.max_dte,
        )


if __name__ == "__main__":
    main()
