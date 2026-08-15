#!/usr/bin/env python3
"""Generate frozen Tier-1 round-trip fixtures for methodology_version v1.

Run once and commit outputs under greeks/fixtures/v1/. Never edit v1 in place
after commit — bump methodology_version and write a new directory instead.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from greeks.domain import OptionRight, SolverInput
from greeks.solver.black76 import forward_intrinsic, price as black_price
from greeks.solver.invert import DEFAULT_TICK_SIZE, invert, relative_iv_error

OUT_DIR = Path(__file__).resolve().parent / "v1"
METHODOLOGY_VERSION = "v1"
MAX_REL_ERR = 1e-10
# Carry-free unit forward for pure round-trip (F=S=100, D=exp(-rT) with r=0.05)
F = 100.0
R = 0.05
ONE_DAY = 1.0 / 365.0


def _moneyness_grid() -> list[float]:
    # Dense near 1.0, span 0.70–1.30
    coarse = [0.70, 0.80, 0.90, 1.10, 1.20, 1.30]
    near = [0.95, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.05]
    return sorted(set(coarse + near))


def _t_grid_years() -> list[float]:
    minutes = [15, 30, 60, 120, 240, 390]  # through 6.5h
    days = [1, 2, 3, 5, 10, 21, 30, 45, 60, 90]
    ts = [m * 60.0 / (365.0 * 86400.0) for m in minutes]
    ts += [d / 365.0 for d in days]
    return ts


def _vol_grid() -> list[float]:
    return [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]


def _invertible(px: float, F: float, K: float, D: float, t: float, right: OptionRight) -> bool:
    """Whether this synthetic price is a fair Tier-1 recovery target."""
    if not math.isfinite(px) or px <= 1e-12:
        return False
    intrins = forward_intrinsic(F, K, D, right)
    extrinsic = px - intrins
    # Need meaningful time value so Jäckel IV is well-defined at 1e-10.
    min_extrinsic = max(DEFAULT_TICK_SIZE, 1e-6 * F)
    if extrinsic < min_extrinsic:
        return False
    # Very short T + deep wing: d1/d2 explode; keep wings for T>=1d only.
    if t < ONE_DAY and (K / F < 0.90 or K / F > 1.10):
        return False
    return True


def main() -> None:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    skipped = 0
    for m in _moneyness_grid():
        K = F * m
        for t in _t_grid_years():
            D = math.exp(-R * t)
            for vol in _vol_grid():
                for right in (OptionRight.CALL, OptionRight.PUT):
                    px = black_price(F, K, t, D, vol, right)
                    if not _invertible(px, F, K, D, t, right):
                        skipped += 1
                        continue
                    result = invert(
                        SolverInput(
                            trade_price=px,
                            strike=K,
                            right=right,
                            forward=F,
                            discount=D,
                            time_to_expiry=t,
                            spot=F,
                        ),
                        methodology_version=METHODOLOGY_VERSION,
                        t_floor_minutes=0.0,  # grid includes 15m; floor applied in prod config
                    )
                    if result.status.value != "success" or result.iv is None:
                        failures.append(
                            f"fail m={m} t={t} vol={vol} right={right} "
                            f"reason={result.reason_code} px={px}"
                        )
                        continue
                    err = relative_iv_error(result.iv, vol)
                    if err > MAX_REL_ERR:
                        failures.append(
                            f"err m={m} t={t} vol={vol} right={right} "
                            f"iv={result.iv} true={vol} rel={err}"
                        )
                        continue
                    rows.append(
                        {
                            "moneyness": m,
                            "strike": K,
                            "forward": F,
                            "spot": F,
                            "discount": D,
                            "r": R,
                            "t": t,
                            "true_vol": vol,
                            "right": right.value,
                            "price": px,
                            "expected_iv": result.iv,
                            "expected_delta": result.delta,
                            "expected_gamma": result.gamma,
                            "expected_vega": result.vega,
                            "expected_theta": result.theta,
                            "expected_rho": result.rho,
                        }
                    )

    if failures:
        raise SystemExit(
            f"Tier-1 generation failed ({len(failures)} issues). First:\n"
            + "\n".join(failures[:20])
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUT_DIR / "grid.jsonl"
    with grid_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    meta = {
        "methodology_version": METHODOLOGY_VERSION,
        "max_relative_iv_error": MAX_REL_ERR,
        "n_rows": len(rows),
        "forward": F,
        "r": R,
        "notes": (
            "Prices generated with Black-76 via py_vollib; expected_iv from scalar invert. "
            "Delta in spot terms (S=F). Theta per calendar day. "
            "Do not edit this fixture set; bump methodology_version for changes."
        ),
    }
    (OUT_DIR / "META.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} rows to {grid_path} (skipped {skipped} non-invertible)")


if __name__ == "__main__":
    main()
