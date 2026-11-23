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


def test_is_short_window_boundary() -> None:
    assert score.is_short_window(0.0) is True
    assert score.is_short_window(4.99) is True
    assert score.is_short_window(5.0) is True        # exactly 5 is short
    assert score.is_short_window(5.01) is False
    assert score.is_short_window(10.0) is False
    assert score.is_short_window(42.0) is False


# ── the three-way group rule (own, start_expiry, session_date, end_expiry, reaction_days) ─────────────

# A non-anchor window: start_expiry 2026-01-20 (the window start), end_expiry 2026-02-20, on a session
# well before the window.
_S = date(2026, 1, 5)          # session_date (not the window start here)
_START = date(2026, 1, 20)     # start_expiry = the window start
_END = date(2026, 2, 20)       # end_expiry


def test_group_earnings_when_own_earnings_date_is_set() -> None:
    # own_earnings_date present → earnings, regardless of nearby reaction days.
    assert score.group_of(date(2026, 2, 6), _START, _S, _END, []) == "earnings"
    assert score.group_of(date(2026, 2, 6), _START, _S, _END, [date(2026, 1, 25)]) == "earnings"


def test_group_other_when_no_own_and_no_nearby_reaction() -> None:
    # own NULL, the one reaction day is far outside the 14-day bands → other.
    assert score.group_of(None, _START, _S, _END, [date(2025, 11, 1)]) == "other"
    assert score.group_of(None, _START, _S, _END, []) == "other"


def test_group_near_earnings_14_days_before_window_start_boundary() -> None:
    # window start = start_expiry (2026-01-20). Exactly 14 days before → near; 15 days before → other.
    assert score.group_of(None, _START, _S, _END, [date(2026, 1, 6)]) == "near_earnings"  # start - 14
    assert score.group_of(None, _START, _S, _END, [date(2026, 1, 5)]) == "other"          # start - 15


def test_group_near_earnings_14_days_after_end_expiry_boundary() -> None:
    # Exactly 14 days after end_expiry (2026-02-20) → near; 15 days after → other.
    assert score.group_of(None, _START, _S, _END, [date(2026, 3, 6)]) == "near_earnings"  # end + 14
    assert score.group_of(None, _START, _S, _END, [date(2026, 3, 7)]) == "other"          # end + 15


def test_group_anchor_window_start_is_the_session_date() -> None:
    # start_expiry NULL → the window start is the session_date; the near test uses the session_date.
    session, end = date(2026, 1, 20), date(2026, 2, 20)
    # a reaction exactly 14 days before the SESSION date is near for the anchor window …
    assert score.group_of(None, None, session, end, [date(2026, 1, 6)]) == "near_earnings"
    # … and 15 days before is not (confirming the anchor uses session_date, not some other start).
    assert score.group_of(None, None, session, end, [date(2026, 1, 5)]) == "other"


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


def test_choose_cutoff_fed_the_short_table_picks_on_short_hit_shares() -> None:
    # The threshold is chosen on the OTHER group's SHORT table. Here the short windows reach 0.80 at
    # cutoff 2; a long table (which fails the one-day test and never reaches 0.80) must not be the
    # input. Feeding the short (cutoff, hit_share) rows yields the short-driven cutoff.
    short_table = [(0.5, 0.52), (1.0, 0.68), (1.5, 0.74), (2.0, 0.83), (3.0, 0.9), (5.0, 0.95)]
    long_table = [(0.5, 0.10), (1.0, 0.15), (1.5, 0.2), (2.0, 0.28), (3.0, 0.35), (5.0, 0.4)]
    assert score.choose_cutoff(short_table) == (2.0, True)
    # The long table would fall back; confirming the two differ, i.e. feeding the wrong one matters.
    assert score.choose_cutoff(long_table) == (score.FALLBACK_CUTOFF, False)


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
    return AlarmRow(underlying=underlying, session_date=session, start_expiry=None,
                    end_expiry=end_expiry, excess_days=exc, crush_days=0.1, support=150.0,
                    eff_days=7.5)


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
