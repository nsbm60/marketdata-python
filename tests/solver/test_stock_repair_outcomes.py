"""The stock repair reports a specific outcome for every out-of-band underlying, and the response
turns a failing repair into ERR_STOCK_REPAIR (ERR_POST_ROUNDING stays for a breach the repair is
not responsible for). Driven directly with hand-built bundles — no solver, no fixtures.

Bundle layout: decision variables are [option, long stock, short stock] per underlying; the option
carries all the delta, stock carries exactly 1 delta/share. Bands are in delta (share) units.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import optimizer  # noqa: E402
from wire_format import (  # noqa: E402
    BC_TYPE, BC_UNDERLYING, BC_LIMIT, BC_VALUE_AT_SOLUTION, BC_VALUE_PCT, BC_REPAIR,
    BCT_BP_LIMIT, BCT_DELTA_TARGET,
)


def _one(delta_opt: float, m_long: float, m_short: float, bp_limit: float,
         *, tradeable: bool = True, legs: bool = True, band: float = 5.0) -> dict:
    """One underlying AAA with an option (index 0) and long/short stock (1, 2)."""
    return {
        "und_names": ["AAA"],
        "u_lo": np.array([-band]), "u_hi": np.array([band]),
        "G_delta": np.array([[delta_opt, 1.0, -1.0]]),
        "margin": np.array([0.0, m_long, m_short]),
        "equity_legs": {"AAA": {"long": 1, "short": 2}} if legs else {"AAA": {}},
        "tradeable": {"AAA": tradeable},
        "bp_limit": bp_limit, "net_liq": 100_000.0,
        "pu": {"AAA": (100.0, 0.0, 5.0)},
    }


def _inside(bundle: dict, q: list) -> bool:
    nd = bundle["G_delta"] @ np.asarray(q, dtype=float)
    return bool(bundle["u_lo"][0] - 1e-6 <= nd[0] <= bundle["u_hi"][0] + 1e-6)


# ── the four outcomes ─────────────────────────────────────────────────────────────────────────

def test_outcome_repaired() -> None:
    b = _one(delta_opt=10.0, m_long=1.0, m_short=1.0, bp_limit=1000.0)  # net 10, band ±5 -> short 5
    q, outcomes = optimizer._repair_band_with_stock([1, 0, 0], b)
    assert outcomes == {"AAA": {"outcome": "repaired"}}
    assert _inside(b, q)


def test_outcome_bp_capped_carries_the_numbers() -> None:
    # net 19, band ±5 -> wants to short 14; BP (150) only affords one 100-margin share.
    b = _one(delta_opt=19.0, m_long=100.0, m_short=100.0, bp_limit=150.0)
    q, outcomes = optimizer._repair_band_with_stock([1, 0, 0], b)
    assert outcomes == {"AAA": {"outcome": "bp_capped", "wanted_shares": -14,
                                "moved_shares": -1, "bp_remaining": 150.0}}


def test_outcome_no_stock_legs() -> None:
    b = _one(delta_opt=10.0, m_long=1.0, m_short=1.0, bp_limit=1000.0, legs=False)
    q, outcomes = optimizer._repair_band_with_stock([1, 0, 0], b)
    assert outcomes == {"AAA": {"outcome": "no_stock_legs"}}
    assert q == [1, 0, 0]  # nothing moved


def test_outcome_not_tradeable() -> None:
    b = _one(delta_opt=10.0, m_long=1.0, m_short=1.0, bp_limit=1000.0, tradeable=False)
    q, outcomes = optimizer._repair_band_with_stock([1, 0, 0], b)
    assert outcomes == {"AAA": {"outcome": "not_tradeable"}}
    assert q == [1, 0, 0]


# ── BP runs out partway: earlier names repaired, later names bp_capped ──────────────────────────

def test_bp_runs_out_partway_earlier_repaired_later_capped() -> None:
    # AAA (processed first) shorts its 5 shares within BP; that leaves only 200 BP, so BBB can
    # short just 2 of the 5 it wants.
    bundle = {
        "und_names": ["AAA", "BBB"],
        "u_lo": np.array([-5.0, -5.0]), "u_hi": np.array([5.0, 5.0]),
        "G_delta": np.array([
            [10.0, 1.0, -1.0, 0.0, 0.0, 0.0],   # AAA: option + its stock legs
            [0.0, 0.0, 0.0, 10.0, 1.0, -1.0],   # BBB
        ]),
        "margin": np.array([0.0, 100.0, 100.0, 0.0, 100.0, 100.0]),
        "equity_legs": {"AAA": {"long": 1, "short": 2}, "BBB": {"long": 4, "short": 5}},
        "tradeable": {"AAA": True, "BBB": True},
        "bp_limit": 700.0, "net_liq": 100_000.0,
        "pu": {"AAA": (100.0, 0.0, 5.0), "BBB": (100.0, 0.0, 5.0)},
    }
    q, outcomes = optimizer._repair_band_with_stock([1, 0, 0, 1, 0, 0], bundle)
    assert outcomes["AAA"] == {"outcome": "repaired"}
    assert outcomes["BBB"] == {"outcome": "bp_capped", "wanted_shares": -5,
                               "moved_shares": -2, "bp_remaining": 200.0}


# ── the reply: error text, repair object, and which error fires ─────────────────────────────────

def test_error_text_and_repair_object_for_a_bp_capped_case() -> None:
    breaches = [{BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: "AAPL", BC_VALUE_PCT: 2.5}]
    outcomes = {"AAPL": {"outcome": "bp_capped", "wanted_shares": -14,
                         "moved_shares": -1, "bp_remaining": 832.80}}
    error, binding = optimizer._post_rounding_error(breaches, outcomes)
    assert error == ("Stock repair could not restore the delta band: "
                     "AAPL bp_capped (wanted -14 shares, moved -1, BP left 832.80).")
    assert error.startswith(optimizer.ERR_STOCK_REPAIR)
    assert binding[0][BC_REPAIR] == outcomes["AAPL"]


def test_error_names_each_failing_underlying_with_its_cause() -> None:
    breaches = [
        {BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: "AAPL", BC_VALUE_PCT: 2.5},
        {BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: "SPY", BC_VALUE_PCT: -1.0},
    ]
    outcomes = {
        "AAPL": {"outcome": "bp_capped", "wanted_shares": -14, "moved_shares": -1, "bp_remaining": 832.80},
        "SPY": {"outcome": "not_tradeable"},
    }
    error, binding = optimizer._post_rounding_error(breaches, outcomes)
    assert ("AAPL bp_capped (wanted -14 shares, moved -1, BP left 832.80); SPY not_tradeable."
            in error)
    assert {b[BC_UNDERLYING]: b[BC_REPAIR] for b in binding} == outcomes


def test_err_post_rounding_only_for_a_non_repair_breach() -> None:
    # A BP-limit breach with every repair succeeded: the repair is not responsible -> a bug.
    bp_breach = [{BC_TYPE: BCT_BP_LIMIT, BC_LIMIT: 1000.0, BC_VALUE_AT_SOLUTION: 1200.0}]
    error, binding = optimizer._post_rounding_error(bp_breach, {"AAA": {"outcome": "repaired"}})
    assert error.startswith(optimizer.ERR_POST_ROUNDING) and "BUG" in error
    assert binding == bp_breach  # no repair object attached

    # A band breach on an underlying the repair reported repaired is also the bug case.
    band_on_repaired = [{BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: "AAA", BC_VALUE_PCT: 3.0}]
    error2, _ = optimizer._post_rounding_error(band_on_repaired, {"AAA": {"outcome": "repaired"}})
    assert error2.startswith(optimizer.ERR_POST_ROUNDING)


def test_a_failing_repair_wins_over_a_bp_breach_and_feasible_is_none() -> None:
    # Even with a BP-limit breach present, a failing repair reports ERR_STOCK_REPAIR, not the bug.
    breaches = [
        {BC_TYPE: BCT_BP_LIMIT, BC_LIMIT: 1000.0, BC_VALUE_AT_SOLUTION: 1200.0},
        {BC_TYPE: BCT_DELTA_TARGET, BC_UNDERLYING: "AAA", BC_VALUE_PCT: 3.0},
    ]
    error, _ = optimizer._post_rounding_error(breaches, {"AAA": {"outcome": "no_stock_legs"}})
    assert error.startswith(optimizer.ERR_STOCK_REPAIR)

    # Feasible: no breach, every repair succeeded -> no error.
    assert optimizer._post_rounding_error([], {"AAA": {"outcome": "repaired"}}) is None
    assert optimizer._post_rounding_error([], {}) is None
