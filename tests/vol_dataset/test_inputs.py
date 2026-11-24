"""assemble_inputs: the look-ahead test (perturbing source rows dated after t must not change t's
inputs — the one fixed exception being the confirmed earnings report the vendor publishes ahead), a
basic column/shape sanity check, and the Parquet round trip."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd
import pytest

from ml.shared.calendar import previous_trading_day

from vol_dataset import HORIZONS_TD
from vol_dataset import inputs as inputs_mod
from vol_dataset.inputs import Frames, assemble_inputs

T = dt.date(2024, 6, 14)   # a Friday


def _days_ending(t: dt.date, n: int) -> List[dt.date]:
    out = [t]
    d = t
    for _ in range(n - 1):
        d = previous_trading_day(d)
        out.append(d)
    return sorted(out)


def _rv_rows(sym: str, days: List[dt.date]) -> List[tuple]:
    rows = []
    for i, d in enumerate(days):
        rows.append((sym, d, 1.0 + 0.01 * i, 0.1, 0.4 + 0.001 * i, 0.5 + 0.001 * i))
    return rows


def _jump_rows(sym: str, days: List[dt.date]) -> List[tuple]:
    rows = []
    for i, d in enumerate(days):
        # one NULL intraday-flag day to exercise the unsplit path
        ij: Optional[int] = None if i == 3 else (1 if i % 7 == 0 else 0)
        jv: Optional[float] = None if ij is None else (0.2 if ij == 1 else 0.0)
        oj = 1 if i % 11 == 0 else 0
        rows.append((sym, d, ij, jv, oj, 0))
    return rows


def _curve_rows(sym: str, t: dt.date) -> List[tuple]:
    # ok horizon rows for all four horizons; total var rising with H.
    return [(sym, t, h, "ok", 0.002 * h, 120.0 + h) for h in HORIZONS_TD]


def _make_frames(perturb: bool) -> Frames:
    days = _days_ending(T, 30)
    rv = _rv_rows("AAA", days) + _rv_rows("SPY", days)
    jump = _jump_rows("AAA", days) + _jump_rows("SPY", days)
    curve = _curve_rows("AAA", T) + _curve_rows("SPY", T)
    events = [("AAA", T, 5, 3.0, 2, 3.0), ("AAA", T, 20, 8.0, 2, 1.5)]
    # Confirmed report 10 days ahead (within 21) — the vendor publishes it ahead, so it is KNOWN at t
    # and must stay fixed under perturbation. Plus a past report (would be 'expected', confirmed wins).
    earnings = [("AAA", T + dt.timedelta(days=10), None, "pre-market"),
                ("AAA", T - dt.timedelta(days=350), None, "post-market")]
    outcomes = [("AAA", T - dt.timedelta(days=30), 0.03, 0.05, 0.001)]
    market_wide: set = set()

    if perturb:
        future = [T + dt.timedelta(days=k) for k in range(1, 6)]
        # wild future rows in every realized/market table — must not reach t's inputs
        rv += [("AAA", d, 99.0, 9.0, 9.0, 9.0) for d in future]
        rv += [("SPY", d, 99.0, 9.0, 9.0, 9.0) for d in future]
        jump += [("AAA", d, 1, 9.0, 1, 1) for d in future]
        jump += [("SPY", d, 1, 9.0, 1, 1) for d in future]
        curve += [("AAA", T + dt.timedelta(days=1), h, "ok", 9.9, 9.0) for h in HORIZONS_TD]
        events.append(("AAA", T + dt.timedelta(days=1), 5, 3.0, 2, 9.9))
        # an earnings report beyond the 21-day confirmation window — must not affect t
        earnings.append(("AAA", T + dt.timedelta(days=60), None, "post-market"))
        market_wide = {T + dt.timedelta(days=2)}

    return Frames(
        rv=pd.DataFrame(rv, columns=["symbol", "trading_date", "rv_total", "rv_overnight",
                                     "rs_down", "rs_up"]),
        jump=pd.DataFrame(jump, columns=["symbol", "trading_date", "intraday_jump", "jump_var",
                                         "overnight_jump", "overnight_jump_adj"]),
        curve=pd.DataFrame(curve, columns=["underlying", "session_date", "horizon_td", "status",
                                           "fit_total_var", "support"]),
        events=pd.DataFrame(events, columns=["underlying", "session_date", "end_td", "eff_days",
                                             "start_td", "excess_days"]),
        earnings=pd.DataFrame(earnings, columns=["symbol", "reported_date", "reaction_date",
                                                 "vendor_report_time"]),
        outcomes=pd.DataFrame(outcomes, columns=["underlying", "end_expiry", "realized_excess_var",
                                                 "excess_var", "normal_daily_var"]),
        market_wide=market_wide)


def _aaa_rows(frames: Frames) -> pd.DataFrame:
    df = assemble_inputs(frames, T, T)
    df = df[df["underlying"] == "AAA"].sort_values("H").reset_index(drop=True)
    return df


def test_assemble_shape_and_key_inputs() -> None:
    df = _aaa_rows(_make_frames(perturb=False))
    assert list(df["H"]) == list(HORIZONS_TD)             # one row per ok horizon
    assert (df["earnings_date_status"] == "confirmed").all()
    assert (df["earnings_date"] == T + dt.timedelta(days=10)).all()
    # within_h is exactly (earnings_td <= H), and the reaction falls within the longest horizon.
    td = df["earnings_td"].iloc[0]
    assert td is not None and td > 0
    assert list(df["earnings_within_h"]) == [td <= h for h in df["H"]]
    assert bool(df[df["H"] == 42]["earnings_within_h"].iloc[0])
    # 30 sessions of history -> the 22-day means are present; the 252-day surprise stats are NULL.
    assert df["c_mean_22"].notna().all()
    assert df["surprise_count_252"].isna().all()
    # the event rule window (end_td 5, eff 3, start 2, excess 3) counts for every H >= 5.
    assert (df["event_rule_count"] >= 1).all()
    # jump scaled is a PLAIN ratio (j_mean / iv_perday_H); C scaled is a LOG ratio.
    import numpy as np
    r = df[df["H"] == 21].iloc[0]
    assert abs(r["j_mean_22_scaled"] - r["j_mean_22"] / r["iv_perday_21"]) < 1e-12
    assert abs(r["c_mean_22_scaled"] - np.log(r["c_mean_22"] / r["iv_perday_21"])) < 1e-12


def test_curve_row_without_rv_row_gives_null_realized_inputs() -> None:
    # AAA has an ok curve row at t but NO rv/jump row anywhere; SPY carries the regime. The AAA rows
    # must build without error, with every realized input and n_unsplit_days_22 NULL (not 0).
    days = _days_ending(T, 30)
    frames = Frames(
        rv=pd.DataFrame(_rv_rows("SPY", days), columns=["symbol", "trading_date", "rv_total",
                                                        "rv_overnight", "rs_down", "rs_up"]),
        jump=pd.DataFrame(_jump_rows("SPY", days), columns=["symbol", "trading_date", "intraday_jump",
                                                           "jump_var", "overnight_jump",
                                                           "overnight_jump_adj"]),
        curve=pd.DataFrame(_curve_rows("AAA", T) + _curve_rows("SPY", T),
                           columns=["underlying", "session_date", "horizon_td", "status",
                                    "fit_total_var", "support"]),
        events=pd.DataFrame([], columns=["underlying", "session_date", "end_td", "eff_days",
                                         "start_td", "excess_days"]),
        earnings=pd.DataFrame([], columns=["symbol", "reported_date", "reaction_date",
                                           "vendor_report_time"]),
        outcomes=pd.DataFrame([], columns=["underlying", "end_expiry", "realized_excess_var",
                                           "excess_var", "normal_daily_var"]),
        market_wide=set())
    df = assemble_inputs(frames, T, T)
    aaa = df[df["underlying"] == "AAA"]
    assert len(aaa) == len(HORIZONS_TD)
    for col in ("c_mean_1", "c_mean_5", "c_mean_22", "j_mean_22", "rs_down_mean_1", "rs_up_mean_22"):
        assert aaa[col].isna().all()
    assert aaa["n_unsplit_days_22"].isna().all()               # NULL, not 0
    assert str(aaa["n_unsplit_days_22"].dtype) == "Int64"       # nullable integer
    assert (aaa["earnings_date_status"] == "unknown").all()


def test_look_ahead_inputs_do_not_change_when_future_rows_change() -> None:
    base = _aaa_rows(_make_frames(perturb=False))
    perturbed = _aaa_rows(_make_frames(perturb=True))
    pd.testing.assert_frame_equal(base, perturbed, check_dtype=False)


def test_parquet_round_trip(tmp_path) -> None:
    df = _aaa_rows(_make_frames(perturb=False))
    # add a couple of target-shaped columns to exercise float/NaN round trip
    df = df.assign(target_down=0.1, target_up=-0.2, sample_weight=130.0)
    path = tmp_path / "2024.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    back = pd.read_parquet(path, engine="pyarrow")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), back.reset_index(drop=True),
                                  check_dtype=False)


def test_build_inputs_uses_assemble(monkeypatch) -> None:
    # build_inputs must be assemble_inputs wrapped around the reads (one path): stub the reads and
    # assert the returned frame is exactly assemble_inputs(frames).
    frames = _make_frames(perturb=False)
    monkeypatch.setattr(inputs_mod, "_read_frames", lambda ch, a, b: frames)
    out = inputs_mod.build_inputs(object(), T, T)
    expected = assemble_inputs(frames, T, T)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))


# ── surprise_mean_jump_ratio (the rolling-sum bug) ──────────────────────────────────────────────────

def _daily_rows(spec):
    """Build rv/jump rows for one symbol from (rv_total, rv_overnight, ij, jv, oj, oja) per day."""
    rv, jump = [], []
    for i, (rt, ov, ij, jv, oj, oja) in enumerate(spec):
        d = dt.date(2024, 1, 1) + dt.timedelta(days=i)
        rv.append(("AAA", d, rt, ov, 0.0, 0.0))
        jump.append(("AAA", d, ij, jv, oj, oja))
    rv_df = pd.DataFrame(rv, columns=["symbol", "trading_date", "rv_total", "rv_overnight",
                                      "rs_down", "rs_up"])
    jump_df = pd.DataFrame(jump, columns=["symbol", "trading_date", "intraday_jump", "jump_var",
                                          "overnight_jump", "overnight_jump_adj"])
    return rv_df, jump_df


def test_surprise_mean_jump_ratio_computed(monkeypatch) -> None:
    # Small windows so the trailing stats resolve on a tiny fixture.
    monkeypatch.setattr(inputs_mod, "SURPRISE_WINDOW", 4)
    monkeypatch.setattr(inputs_mod, "UNSPLIT_WINDOW", 2)
    # 6 days, rv_total 1.0, no overnight jumps; intraday jumps (surprises) on d2 (0.5) and d4 (0.3).
    spec = [
        (1.0, 0.0, 0, 0.0, 0, 0),   # d0  C=1.0
        (1.0, 0.0, 0, 0.0, 0, 0),   # d1  C=1.0
        (1.0, 0.0, 1, 0.5, 0, 0),   # d2  J=0.5 C=0.5  (surprise)
        (1.0, 0.0, 0, 0.0, 0, 0),   # d3  C=1.0
        (1.0, 0.0, 1, 0.3, 0, 0),   # d4  J=0.3 C=0.7  (surprise)
        (1.0, 0.0, 0, 0.0, 0, 0),   # d5  C=1.0
    ]
    rv_df, jump_df = _daily_rows(spec)
    d = inputs_mod._daily(rv_df, jump_df, market_wide=set(), own_reaction=set())
    last = d.iloc[-1]   # d5
    # trailing-2 C means: c22(d2)=mean(1.0,0.5)=0.75, c22(d4)=mean(1.0,0.7)=0.85
    # window = last 4 sessions d2..d5; surprise days d2, d4; mean of J_d / c22(d):
    expected = ((0.5 / 0.75) + (0.3 / 0.85)) / 2
    assert last["surprise_count_252"] == 2
    assert abs(last["surprise_mean_jump_ratio"] - expected) < 1e-12


def test_surprise_mean_jump_ratio_null_when_count_zero(monkeypatch) -> None:
    monkeypatch.setattr(inputs_mod, "SURPRISE_WINDOW", 4)
    monkeypatch.setattr(inputs_mod, "UNSPLIT_WINDOW", 2)
    # 6 days, no jumps -> no surprise days -> count 0 and the ratio is NULL (not 0).
    spec = [(1.0, 0.0, 0, 0.0, 0, 0) for _ in range(6)]
    rv_df, jump_df = _daily_rows(spec)
    d = inputs_mod._daily(rv_df, jump_df, market_wide=set(), own_reaction=set())
    last = d.iloc[-1]
    assert last["surprise_count_252"] == 0
    assert pd.isna(last["surprise_mean_jump_ratio"])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
