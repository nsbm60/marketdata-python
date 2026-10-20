"""Tier-1 frozen fixtures: 1e-10 relative IV recovery + frozen expected iv to 1e-12 relative.

The frozen-IV check was bit for bit. The v1 grid was generated on macOS arm64, and the last ulp
of Jäckel's IV differs on Linux (x86_64 and aarch64: e.g. 2.0 vs 1.9999999999999996), so the
check failed on drogon's certify step. 1e-12 relative is 10^4 ulps and still far inside the
1e-10 recovery gate; any methodology change moves IV by orders of magnitude more.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greeks.domain import OptionRight, SolverInput
from greeks.solver.invert import invert, relative_iv_error

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "greeks" / "fixtures" / "v1"
MAX_REL = 1e-10
FROZEN_REL = 1e-12


def _load_rows() -> list[dict[str, object]]:
    path = FIXTURE_DIR / "grid.jsonl"
    if not path.is_file():
        pytest.skip(f"fixtures not generated: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_meta_methodology_version() -> None:
    meta_path = FIXTURE_DIR / "META.json"
    if not meta_path.is_file():
        pytest.skip("META.json missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["methodology_version"] == "v1"
    assert meta["max_relative_iv_error"] == MAX_REL


def test_tier1_relative_and_bit_for_bit() -> None:
    rows = _load_rows()
    assert len(rows) > 100  # non-trivial grid

    worst = 0.0
    for row in rows:
        right = OptionRight(str(row["right"]))
        result = invert(
            SolverInput(
                trade_price=float(row["price"]),  # type: ignore[arg-type]
                strike=float(row["strike"]),  # type: ignore[arg-type]
                right=right,
                forward=float(row["forward"]),  # type: ignore[arg-type]
                discount=float(row["discount"]),  # type: ignore[arg-type]
                time_to_expiry=float(row["t"]),  # type: ignore[arg-type]
                spot=float(row["spot"]),  # type: ignore[arg-type]
            ),
            methodology_version="v1",
            t_floor_minutes=0.0,
        )
        assert result.status.value == "success", row
        assert result.iv is not None
        true_vol = float(row["true_vol"])  # type: ignore[arg-type]
        rel = relative_iv_error(result.iv, true_vol)
        worst = max(worst, rel)
        assert rel <= MAX_REL, f"rel={rel} row={row}"
        # Frozen expected_iv under current methodology (portable tolerance, see docstring)
        expected = float(row["expected_iv"])  # type: ignore[arg-type]
        assert abs(result.iv - expected) <= FROZEN_REL * abs(expected), row

    assert worst <= MAX_REL
