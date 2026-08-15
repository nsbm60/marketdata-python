"""CLI: invert staged trades and optionally join vendor snapshots.

Usage::

    # Invert only (local store; SOFR fixture if no CH SOFR)
    python -m greeks.harness.run --underlying NVDA --date 2026-05-27

    # Invert + join from vendor JSONL (tests / offline)
    python -m greeks.harness.run --underlying NVDA --date 2026-05-27 \\
        --vendor-jsonl path/to/snapshots.jsonl

    # Invert + join from ClickHouse option_snapshot; write rows to CH
    python -m greeks.harness.run --underlying NVDA --date 2026-05-27 \\
        --vendor-ch --write-ch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from greeks.config import GreeksConfig, load_config
from greeks.domain import FailureReason, JoinClass, RowStatus
from greeks.forwards.sofr import load_sofr_csv
from greeks.harness.invert_trades import invert_batch, invert_staged_trade
from greeks.harness.join_vendor import (
    filter_baseline_snapshots,
    join_batch,
    match_capture,
)
from greeks.harness.rows import VendorSnapshot
from greeks.harness.store import ResultsStore
from greeks.occ import parse_occ
from greeks.pull.staging import StagedTrade, TradeStaging


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Greeks harness invert + join (PR5)")
    p.add_argument("--underlying", type=str, required=True)
    p.add_argument("--date", type=_parse_date, required=True)
    p.add_argument(
        "--staging-db",
        type=Path,
        default=Path("data/greeks_staging.db"),
    )
    p.add_argument(
        "--results-db",
        type=Path,
        default=Path("data/greeks_results.db"),
    )
    p.add_argument(
        "--sofr-csv",
        type=Path,
        default=None,
        help="SOFR fixture CSV (default: package fixture; overridden by --sofr-ch)",
    )
    p.add_argument(
        "--sofr-ch",
        action="store_true",
        help=(
            "Load SOFR from trading.sofr_daily source=fred only "
            "(FINAL). Fails if FRED rows missing — never uses fixture rows."
        ),
    )
    p.add_argument(
        "--vendor-jsonl",
        type=Path,
        default=None,
        help="Optional vendor snapshots JSONL for offline join",
    )
    p.add_argument(
        "--vendor-ch",
        action="store_true",
        help="Pull vendor snapshots from trading.option_snapshot for join",
    )
    p.add_argument(
        "--write-ch",
        action="store_true",
        help="Insert validation (+ residuals if joined) into ClickHouse",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--join-only",
        action="store_true",
        help="Skip invert; re-join from results_db validation rows",
    )
    p.add_argument(
        "--price",
        choices=("trade", "mid"),
        default="trade",
        help=(
            "Option price to invert: trade print (default), or vendor bid/ask mid "
            "from the same capture-clock match used for residuals. "
            "--price mid requires --vendor-ch or --vendor-jsonl; rows without a "
            "two-sided quote fail with missing_vendor_mid (no silent fallback)."
        ),
    )
    return p


def _vendor_mid(snap: VendorSnapshot) -> Optional[float]:
    """(bid+ask)/2 when two-sided and well-ordered; else None (no fallback)."""
    if snap.bid is None or snap.ask is None:
        return None
    if snap.bid <= 0 or snap.ask <= 0 or snap.ask < snap.bid:
        return None
    return 0.5 * (float(snap.bid) + float(snap.ask))


def _failure_missing_mid(trade: StagedTrade, cfg: GreeksConfig) -> Any:
    """Explicit failure when --price mid cannot form a mid from vendor quote."""
    from greeks.harness.invert_trades import _failure

    try:
        occ = parse_occ(trade.symbol)
        expiry, strike, right = occ.expiry, occ.strike, occ.right
    except ValueError:
        expiry, strike, right = trade.session_date, 0.0, None
    return _failure(
        trade,
        cfg.methodology_version,
        FailureReason.MISSING_VENDOR_MID,
        expiry=expiry,
        strike=strike,
        right=right,
        spot=trade.spot_at_trade,
    )


def load_vendor_jsonl(path: Path) -> list[VendorSnapshot]:
    out: list[VendorSnapshot] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o: dict[str, Any] = json.loads(line)
            out.append(
                VendorSnapshot(
                    symbol=str(o["symbol"]),
                    underlying=str(o.get("underlying", "")),
                    timestamp=_parse_ts(o["timestamp"]),
                    quote_timestamp=(
                        _parse_ts(o["quote_timestamp"])
                        if o.get("quote_timestamp")
                        else None
                    ),
                    bid=_opt_float(o.get("bid")),
                    ask=_opt_float(o.get("ask")),
                    iv=_opt_float(o.get("iv")),
                    delta=_opt_float(o.get("delta")),
                    gamma=_opt_float(o.get("gamma")),
                    vega=_opt_float(o.get("vega")),
                    theta=_opt_float(o.get("theta")),
                    rho=_opt_float(o.get("rho")),
                )
            )
    return out


def _parse_ts(v: object) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def _opt_float(v: object) -> Optional[float]:
    if v is None:
        return None
    return float(v)  # type: ignore[arg-type]


def _group_by_symbol(
    snaps: list[VendorSnapshot],
) -> dict[str, list[VendorSnapshot]]:
    by_sym: dict[str, list[VendorSnapshot]] = {}
    for s in snaps:
        by_sym.setdefault(s.symbol, []).append(s)
    return by_sym


def _print_join_summary(residuals: list[Any]) -> None:
    classes: dict[str, int] = {}
    for r in residuals:
        classes[r.join_class.value] = classes.get(r.join_class.value, 0) + 1
    matched_bps = [
        r.residual_iv_bps
        for r in residuals
        if r.join_class is JoinClass.MATCHED and r.residual_iv_bps is not None
    ]
    print(f"join: classes={classes}")
    if matched_bps:
        med = sorted(abs(x) for x in matched_bps)[len(matched_bps) // 2]
        print(f"join: matched={len(matched_bps)} median_|iv_bps|={med:.2f}")
    print(
        f"note: vendor_missing={classes.get('vendor_missing', 0)} "
        f"(not counted as disagreement)"
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    und = args.underlying.upper()
    session = args.date

    if session in cfg.window.excluded_dates:
        raise SystemExit(f"excluded date {session}")
    if args.vendor_jsonl is not None and args.vendor_ch:
        raise SystemExit("use only one of --vendor-jsonl / --vendor-ch")

    ch_client = None
    if args.sofr_ch or args.vendor_ch or args.write_ch:
        from greeks.ch import get_ch_client

        ch_client = get_ch_client()

    if args.sofr_ch:
        from greeks.ch import load_sofr_series_from_ch

        # source=fred only — no fixture mix-in, no silent alternate source.
        try:
            sofr = load_sofr_series_from_ch(
                ch_client,  # type: ignore[arg-type]
                table=cfg.tables.sofr_daily,
                start=cfg.window.start,
                end=cfg.window.end,
                source="fred",
            )
        except ValueError as e:
            raise SystemExit(f"sofr-ch failed: {e}") from e
        rates = list(sofr.values())
        print(
            f"sofr-ch: source=fred dates={len(sofr)} "
            f"rate_min={min(rates):.6f} rate_max={max(rates):.6f}"
        )
    elif args.sofr_csv is not None:
        sofr = load_sofr_csv(args.sofr_csv)
        print(f"sofr-csv: {args.sofr_csv} dates={len(sofr)}")
    else:
        # Explicit offline path only (unit tests / no CH). Not a CH fallback.
        sofr = load_sofr_csv()
        print(f"sofr-csv: package fixture dates={len(sofr)}")

    if args.price == "mid" and not (args.vendor_ch or args.vendor_jsonl):
        raise SystemExit("--price mid requires --vendor-ch or --vendor-jsonl")
    if args.price == "mid" and args.join_only:
        raise SystemExit("--price mid cannot be combined with --join-only")

    residuals: list[Any] = []
    snaps: list[VendorSnapshot] = []
    with ResultsStore(args.results_db) as store:
        if not args.join_only:
            with TradeStaging(args.staging_db) as staging:
                trades = list(staging.iter_session(und, session))
            if not trades:
                raise SystemExit(
                    f"no staged trades for {und} {session} in {args.staging_db}"
                )

            # Vendor snaps needed before invert when pricing off mid.
            if args.vendor_jsonl is not None or args.vendor_ch:
                if args.vendor_jsonl is not None:
                    snaps = load_vendor_jsonl(args.vendor_jsonl)
                else:
                    from greeks.ch import fetch_vendor_snapshots

                    symbols = sorted({t.symbol for t in trades})
                    snaps = fetch_vendor_snapshots(
                        ch_client,  # type: ignore[arg-type]
                        underlying=und,
                        session_date=session,
                        symbols=symbols,
                        table=cfg.tables.option_snapshot,
                    )
                    print(
                        f"vendor-ch: symbols={len(symbols)} snapshots={len(snaps)}"
                    )
                snaps = filter_baseline_snapshots(
                    snaps,
                    window_start=cfg.window.start,
                    excluded_dates=cfg.window.excluded_dates,
                )

            option_prices: Optional[list[Optional[float]]] = None
            if args.price == "mid":
                by_sym = _group_by_symbol(snaps)
                option_prices = []
                n_mid = 0
                n_miss = 0
                for tr in trades:
                    m = match_capture(
                        tr.trade_ts,
                        by_sym.get(tr.symbol, ()),
                        max_staleness_s=cfg.join_staleness_s,
                    )
                    mid = _vendor_mid(m.snapshot) if m is not None else None
                    if mid is None:
                        option_prices.append(None)  # marker; handle below
                        n_miss += 1
                    else:
                        option_prices.append(mid)
                        n_mid += 1
                print(
                    f"price=mid: two_sided_mid={n_mid} missing_vendor_mid={n_miss} "
                    f"(no fallback to trade print)"
                )
                # Invert only rows with mid; explicit failure rows for the rest.
                rows = []
                from greeks.harness.invert_trades import InvertBatchStats

                n_success = n_failure = 0
                by_reason: dict[str, int] = {}
                for tr, mid in zip(trades, option_prices, strict=True):
                    if mid is None:
                        row = _failure_missing_mid(tr, cfg)
                    else:
                        row = invert_staged_trade(
                            tr, cfg=cfg, sofr_series=sofr, option_price=mid
                        )
                    rows.append(row)
                    if row.status is RowStatus.SUCCESS:
                        n_success += 1
                    else:
                        n_failure += 1
                        key = (
                            row.reason_code.value if row.reason_code else "unknown"
                        )
                        by_reason[key] = by_reason.get(key, 0) + 1
                stats = InvertBatchStats(
                    n_input=len(trades),
                    n_success=n_success,
                    n_failure=n_failure,
                    by_reason=by_reason,
                )
            else:
                rows, stats = invert_batch(
                    trades, cfg=cfg, sofr_series=sofr
                )

            if not stats.balanced():
                raise SystemExit(f"accounting failed: {stats}")
            n = store.upsert_validation(rows)
            print(
                f"invert: price={args.price} input={stats.n_input} "
                f"success={stats.n_success} failure={stats.n_failure} "
                f"wrote={n} reasons={dict(stats.by_reason)}"
            )
            val_rows = rows
        else:
            val_rows = [
                r
                for r in store.iter_validation()
                if r.underlying == und and r.trade_ts.date() == session
            ]
            print(f"join-only: loaded {len(val_rows)} validation rows")

        if args.vendor_jsonl is not None or args.vendor_ch:
            if not snaps:
                # join-only path still needs snaps
                if args.vendor_jsonl is not None:
                    snaps = load_vendor_jsonl(args.vendor_jsonl)
                else:
                    from greeks.ch import fetch_vendor_snapshots

                    symbols = sorted({r.symbol for r in val_rows})
                    snaps = fetch_vendor_snapshots(
                        ch_client,  # type: ignore[arg-type]
                        underlying=und,
                        session_date=session,
                        symbols=symbols,
                        table=cfg.tables.option_snapshot,
                    )
                    print(
                        f"vendor-ch: symbols={len(symbols)} snapshots={len(snaps)}"
                    )
                snaps = filter_baseline_snapshots(
                    snaps,
                    window_start=cfg.window.start,
                    excluded_dates=cfg.window.excluded_dates,
                )
            residuals = join_batch(
                val_rows,
                _group_by_symbol(snaps),
                max_staleness_s=cfg.join_staleness_s,
            )
            store.upsert_residuals(residuals)
            _print_join_summary(residuals)
        else:
            print("no --vendor-jsonl/--vendor-ch; skip join")

        if args.write_ch:
            from greeks.ch import insert_residual_rows, insert_validation_rows

            n_v = insert_validation_rows(
                ch_client,  # type: ignore[arg-type]
                val_rows,
                table=cfg.tables.greeks_validation,
            )
            n_r = 0
            if residuals:
                n_r = insert_residual_rows(
                    ch_client,  # type: ignore[arg-type]
                    residuals,
                    table=cfg.tables.greeks_residuals,
                )
            print(f"write-ch: validation={n_v} residuals={n_r}")

        print(
            f"store validation={store.count_validation(methodology_version=cfg.methodology_version)} "
            f"residuals={store.count_residuals_by_class(methodology_version=cfg.methodology_version)}"
        )


if __name__ == "__main__":
    main()
