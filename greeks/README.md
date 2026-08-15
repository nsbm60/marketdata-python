# greeks — validation phase

CPU-only methodology gate: Black-76 IV (py_vollib) vs Massive vendor greeks.

## Specs

| Doc | Path |
|-----|------|
| Brief | `Scala/MarketData/docs/investigations/greeks-validation-brief.md` |
| Plan | `Scala/MarketData/docs/plans/greeks-validation.md` |
| DDL (system of record) | `Scala/MarketData` `database_schema.sql` + `migrations/` — see [SCHEMA.md](SCHEMA.md) |

## Constraints

- Python 3.12+, `mypy --strict` on this package
- **No** `import asyncio` / `import threading` under `greeks/` (CI grep)
- Frozen dataclasses; failure reason codes; no silent defaults / NaN fills
- Array math via `xp` module handle (NumPy-bound this phase)
- Config only through `greeks.config.load_config` / `get_config`

## Config

```bash
# default: config/greeks.yaml at repo root
# override path:
export GREEKS_CONFIG=/path/to/greeks.yaml
```

Env (read only inside `greeks.config`): `CLICKHOUSE_*`, `MASSIVE_API_KEY` / `POLYGON_API_KEY`, `ALPACA_*` / `APCA_*`, `FRED_API_KEY`, `GREEKS_CONFIG`.

## Dev checks

```bash
pip install -e ".[dev]"
./scripts/check_greeks.sh
pytest tests/greeks
```

Tier-2 (QuantLib) optional:

```bash
pip install -e ".[dev,tier2]"
pytest tests/greeks/test_tier2_quantlib.py -q
```

Conventions / accepted systematics: [docs/tier2_conventions.md](docs/tier2_conventions.md).

## Phases

| PR | Scope |
|----|--------|
| PR0 | package, domain, config, CI hygiene |
| PR1 | solver + Tier-1 fixtures |
| PR2 | QuantLib cross-check (optional extra `tier2`) |
| PR3 | forwards (escrowed carry + SOFR) + CH DDL pointer |
| PR4 | work queue + Massive trades + Alpaca RAW spot → staging |
| PR5 | invert staged trades + vendor join → results store |
| **PR6** | residual reports / NTM gate / sub-1-DTE / earnings slice |

## Harness CLI (PR5)

```bash
# After pull staging exists:
python -m greeks.harness.run --underlying NVDA --date 2026-05-27

# Offline join with vendor snapshots JSONL:
python -m greeks.harness.run --underlying NVDA --date 2026-05-27 \
  --vendor-jsonl path/to/snapshots.jsonl
```

Scalar invert only (no vectorized path). Every staged trade → one validation row.
Join: capture clock at-or-after (5 min); capture lag and quote lag stored separately.

## Report CLI (PR6)

```bash
python -m greeks.harness.report_cli --results-db data/greeks_results.db
python -m greeks.harness.report_cli --earnings NVDA:2026-05-28
```

Hard gate: matched NTM (0.95–1.05) × 3–30 DTE → median |IV| ≤ 20 bps, |Δ| ≤ 0.005.  
Sub-1-DTE ATM (>2h) target 100 bps and earnings T-3..T+2 (quote-lag buckets) are report-only.  
CH SQL: `greeks/harness/sql/residual_report.sql`.

## Pull CLI (PR4)

```bash
# Smoke: few contracts for one day → data/greeks_staging.db
python -m greeks.pull.run --ticker NVDA --date 2026-05-27 --max-contracts 5

# Seed queue only
python -m greeks.pull.run --ticker NVDA --date 2026-05-27 --seed-queue-only

# Drain pending queue items
python -m greeks.pull.run --drain-queue
```

Requires `MASSIVE_API_KEY`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`. Skips
`excluded_dates` (e.g. 2026-06-08). Staging only — invert is PR5.

Operational notes (live):
- Alpaca SIP full-day equity tape is large (NVDA ≈ 2.7M prints). Pulls are
  **hour-chunked with retries** (`greeks.pull.alpaca_spot.fetch_equity_trades`).
- Smoke / hard-gate band: `--min-dte 3 --max-dte 30 --max-contracts N` prefers
  nearest-ATM when capping.
- Invert uses the package SOFR CSV by default (`--sofr-csv`). Prefer
  `--sofr-ch` once `trading.sofr_daily` is loaded.
- SOFR load: `python -m greeks.forwards --load-fixture` (or `--fred` with
  `FRED_API_KEY`).
- Vendor join: `--vendor-ch` pulls `trading.option_snapshot`; `--write-ch`
  inserts validation + residuals into ClickHouse.
- Example (after pull/staging exists)::

      python -m greeks.harness.run --underlying NVDA --date 2026-05-27 \
        --sofr-ch --vendor-ch --write-ch
