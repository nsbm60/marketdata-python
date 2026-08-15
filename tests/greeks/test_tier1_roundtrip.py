"""Tier-1 frozen fixtures: 1e-10 relative IV recovery + bit-for-bit expected iv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greeks.domain import OptionRight, SolverInput
from greeks.solver.invert import invert, relative_iv_error

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "greeks" / "fixtures" / "v1"
MAX_REL = 1e-10


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
        # Bit-for-bit vs frozen expected_iv under current methodology
        assert result.iv == float(row["expected_iv"])  # type: ignore[arg-type]

    assert worst <= MAX_REL
