"""The pure parts of the detector scoring report (out_v1 vs fwd_v1): the hit rule, the band
classifiers, the cutoff-choice rule and the two list dedupe rules. No ClickHouse."""

from __future__ import annotations

from datetime import date

from iv_event import score
from iv_event.score import AlarmRow, SurpriseRow


# ── the hit rule ──────────────────────────────────────────────────────────────────────────────────

def test_is_hit_needs_half_the_priced_excess_and_null_crush_is_not_a_hit() -> None:
    assert score.is_hit(0.6, 1.0) is True          # 0.6 >= 0.5 * 1.0
    assert score.is_hit(0.5, 1.0) is True           # exactly half counts
    assert score.is_hit(0.49, 1.0) is False         # below half
    assert score.is_hit(None, 1.0) is False         # no release day → never a hit
    assert score.is_hit(0.0, 0.0) is True           # 0 >= 0.5*0 (degenerate, still defined)


# ── band classifiers ────────────────────────────────────────────────────────────────────────────────

def test_support_band_boundaries() -> None:
    assert score.support_band(100) == "100-250"
    assert score.support_band(249.9) == "100-250"
    assert score.support_band(250) == "250-1000"
    assert score.support_band(999.9) == "250-1000"
    assert score.support_band(1000) == "1000+"
    assert score.support_band(50_000) == "1000+"


def test_start_td_band_boundaries() -> None:
    assert score.start_td_band(0) == "0-5"
    assert score.start_td_band(5) == "0-5"
    assert score.start_td_band(6) == "6-20"
    assert score.start_td_band(20) == "6-20"
    assert score.start_td_band(21) == "21+"
    assert score.start_td_band(99) == "21+"


# ── the cutoff-choice rule ──────────────────────────────────────────────────────────────────────────

def test_choose_cutoff_picks_the_smallest_cutoff_reaching_the_target() -> None:
    # hit share rises with cutoff; first to reach 0.80 is 1.5.
    shares = [(0.5, 0.40), (1.0, 0.70), (1.5, 0.82), (2.0, 0.90), (3.0, 0.95)]
    cutoff, reached = score.choose_cutoff(shares)
    assert cutoff == 1.5 and reached is True


def test_choose_cutoff_ignores_a_lower_cutoff_that_dips_back_below() -> None:
    # Must scan in grid order and take the FIRST at/above target, not the largest.
    shares = [(0.5, 0.81), (1.0, 0.60), (1.5, 0.95)]
    cutoff, reached = score.choose_cutoff(shares)
    assert cutoff == 0.5 and reached is True


def test_choose_cutoff_falls_back_to_two_when_none_reaches_target() -> None:
    shares = [(0.5, 0.40), (1.0, 0.55), (1.5, 0.70), (2.0, 0.79)]
    cutoff, reached = score.choose_cutoff(shares)
    assert cutoff == score.FALLBACK_CUTOFF == 2.0 and reached is False


def test_choose_cutoff_treats_a_missing_or_none_share_as_not_reaching() -> None:
    # A cutoff with no windows (hit share None) is not eligible; the next qualifying grid cutoff wins.
    shares = [(0.5, None), (1.0, 0.85)]
    cutoff, reached = score.choose_cutoff(shares)
    assert cutoff == 1.0 and reached is True
    # Nothing qualifies at all → fallback.
    assert score.choose_cutoff([(0.5, None)]) == (2.0, False)


# ── dedupe rules ─────────────────────────────────────────────────────────────────────────────────────

def _surprise(underlying: str, fj: date, start_td: int, end_td: int, exc: float = 3.0) -> SurpriseRow:
    return SurpriseRow(underlying=underlying, first_jump_date=fj, max_abs_ret_adj=0.05,
                       excess_days=exc, session_date=date(2025, 1, 2), end_td=end_td, start_td=start_td)


def test_dedupe_smallest_start_td_keeps_one_per_underlying_and_first_jump() -> None:
    rows = [
        _surprise("LYFT", date(2025, 2, 14), start_td=10, end_td=12),
        _surprise("LYFT", date(2025, 2, 14), start_td=3, end_td=8),    # smallest start_td → kept
        _surprise("LYFT", date(2025, 2, 14), start_td=3, end_td=20),   # tie start_td, larger end_td → not
        _surprise("LYFT", date(2025, 5, 1), start_td=7, end_td=9),     # different first_jump → separate
        _surprise("UBER", date(2025, 2, 14), start_td=99, end_td=100),  # different name → separate
    ]
    kept = {(r.underlying, r.first_jump_date): r for r in score.dedupe_smallest_start_td(rows)}
    assert len(kept) == 3
    lyft_feb = kept[("LYFT", date(2025, 2, 14))]
    assert lyft_feb.start_td == 3 and lyft_feb.end_td == 8   # smallest start_td, tie broken by end_td


def _alarm(underlying: str, end_expiry: date, exc: float, session: date) -> AlarmRow:
    return AlarmRow(underlying=underlying, session_date=session, end_expiry=end_expiry,
                    excess_days=exc, crush_days=0.1, support=150.0)


def test_dedupe_largest_excess_days_keeps_one_per_underlying_and_end_expiry() -> None:
    rows = [
        _alarm("SPY", date(2025, 3, 21), exc=2.0, session=date(2025, 2, 1)),
        _alarm("SPY", date(2025, 3, 21), exc=5.0, session=date(2025, 2, 10)),  # largest excess → kept
        _alarm("SPY", date(2025, 3, 21), exc=5.0, session=date(2025, 1, 5)),   # tie excess, earlier session → kept
        _alarm("SPY", date(2025, 6, 20), exc=1.0, session=date(2025, 4, 1)),   # different expiry → separate
        _alarm("QQQ", date(2025, 3, 21), exc=9.0, session=date(2025, 2, 1)),   # different name → separate
    ]
    kept = {(r.underlying, r.end_expiry): r for r in score.dedupe_largest_excess_days(rows)}
    assert len(kept) == 3
    spy_mar = kept[("SPY", date(2025, 3, 21))]
    assert spy_mar.excess_days == 5.0 and spy_mar.session_date == date(2025, 1, 5)  # tie → earliest session
