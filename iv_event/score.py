"""
The detector scoring report (roadmap-2026-11-16 Stage 3b part 3 step 2): `python -m iv_event
outcome-score`. Read-only — it scores the fwd_v1 event windows against the out_v1 outcome record,
picks the detection threshold, and prints a six-part report. It writes NOTHING to ClickHouse.

Population (every part unless stated): iv_event_outcome FINAL (method out_v1) INNER JOINed to
iv_event_window FINAL (method fwd_v1, status ok) on (underlying, session_date, end_td) — the shared
_joined_cte in outcome_event, extended to carry the window's normal_daily_var and start_td. Kept:
corporate_action = 0, realized_var IS NOT NULL, support >= OUTCOME_SUMMARY_MIN_SUPPORT (100). The unit
is the window row — one window seen from one session; the report header says so.

Definitions (in plain words):
  - crush_days            = crush_excess_var / normal_daily_var — the largest one-day excess release,
                            in normal days;
  - realized_excess_days  = realized_excess_var / normal_daily_var;
  - hit                   = crush_excess_var >= 0.5 * excess_var — at least half the priced excess came
                            off on one day. A NULL crush_excess_var is NOT a hit;
  - earnings group        = own_earnings_date IS NOT NULL; other group = own_earnings_date IS NULL;
  - cutoff grid (excess_days): 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10.

The threshold (part 2) is the smallest grid cutoff at which the OTHER group's hit share >= 0.80; if no
cutoff reaches it, the report says so and uses 2 for parts 3-6. The band classifiers, the hit rule, the
cutoff choice and the two list dedupe rules are pure functions (below), unit-tested without ClickHouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import (
    JUMP_DAILY_TABLE, JUMP_METHOD, MARKET_WIDE_MIN_NAMES, OUTCOME_SUMMARY_MIN_SUPPORT,
)
from . import outcome_event

# ── scoring constants (a first setting; all named here) ────────────────────────────────────────────
CUTOFF_GRID: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)
HIT_FRACTION = 0.5            # a window is a "hit" when crush_excess_var >= this * excess_var
HIT_SHARE_TARGET = 0.80       # the chosen cutoff is the smallest where the other group's hit share hits this
FALLBACK_CUTOFF = 2.0         # used for parts 3-6 when no cutoff reaches the target
RECENT_SINCE = date(2025, 1, 1)   # the "recent" lists (parts 3, 4) start here
LIST_LIMIT = 30               # rows in the parts 3 / 4 lists
TOP_DATES_PER_YEAR = 5        # the learned-calendar top dates per year (part 6)


# ── pure rules (unit-tested, no ClickHouse) ────────────────────────────────────────────────────────

def is_hit(crush_excess_var: Optional[float], excess_var: float) -> bool:
    """A window is a hit when at least HIT_FRACTION of the priced excess came off on one day. A NULL
    crush_excess_var (no release day) is never a hit."""
    if crush_excess_var is None:
        return False
    return crush_excess_var >= HIT_FRACTION * excess_var


def support_band(support: float) -> str:
    """The support band label (population support is >= 100): 100-250, 250-1000 or 1000+."""
    if support < 250:
        return "100-250"
    if support < 1000:
        return "250-1000"
    return "1000+"


def start_td_band(start_td: int) -> str:
    """The start_td band label: 0-5, 6-20 or 21+ (trading days from the session to the window start)."""
    if start_td <= 5:
        return "0-5"
    if start_td <= 20:
        return "6-20"
    return "21+"


def choose_cutoff(other_hit_shares: Sequence[Tuple[float, Optional[float]]]) -> Tuple[float, bool]:
    """The chosen detection cutoff: the smallest grid cutoff at which the other group's hit share is
    >= HIT_SHARE_TARGET. Returns (cutoff, reached_target). When no cutoff reaches it (or a cutoff has
    no windows, hit share None), returns (FALLBACK_CUTOFF, False)."""
    shares = {float(c): s for c, s in other_hit_shares}
    for c in CUTOFF_GRID:
        s = shares.get(float(c))
        if s is not None and s >= HIT_SHARE_TARGET:
            return float(c), True
    return FALLBACK_CUTOFF, False


@dataclass(frozen=True)
class SurpriseRow:
    """A missed-event (surprise) candidate for part 3's recent list."""

    underlying: str
    first_jump_date: date
    max_abs_ret_adj: Optional[float]
    excess_days: float
    session_date: date
    end_td: int
    start_td: int


@dataclass(frozen=True)
class AlarmRow:
    """A false-alarm candidate for part 4's recent list."""

    underlying: str
    session_date: date
    end_expiry: date
    excess_days: float
    crush_days: Optional[float]
    support: float


def dedupe_smallest_start_td(rows: Sequence[SurpriseRow]) -> List[SurpriseRow]:
    """One row per (underlying, first_jump_date): the window with the smallest start_td that contains
    the jump (ties → smallest end_td, for determinism). Input order does not matter."""
    best: Dict[Tuple[str, date], SurpriseRow] = {}
    for r in rows:
        key = (r.underlying, r.first_jump_date)
        cur = best.get(key)
        if cur is None or (r.start_td, r.end_td) < (cur.start_td, cur.end_td):
            best[key] = r
    return list(best.values())


def dedupe_largest_excess_days(rows: Sequence[AlarmRow]) -> List[AlarmRow]:
    """One row per (underlying, end_expiry): the one with the largest excess_days (ties → earliest
    session_date, for determinism). Input order does not matter."""
    best: Dict[Tuple[str, date], AlarmRow] = {}
    for r in rows:
        key = (r.underlying, r.end_expiry)
        cur = best.get(key)
        if cur is None or (r.excess_days, -r.session_date.toordinal()) > (
                cur.excess_days, -cur.session_date.toordinal()):
            best[key] = r
    return list(best.values())


# ── the scored population CTE (reuses outcome_event._joined_cte) ───────────────────────────────────

def _scored_cte() -> str:
    """The full `WITH … s AS (…)` prefix: the shared join, filtered to the scoring population, with the
    derived columns (is_earnings, is_hit, crush_days, realized_excess_days). Queries select FROM s."""
    return (
        f"WITH {outcome_event._joined_cte()}, "
        f"s AS (SELECT *, "
        f"(w_own_earnings_date IS NOT NULL) AS is_earnings, "
        f"toUInt8(crush_excess_var IS NOT NULL AND crush_excess_var >= {HIT_FRACTION!r} * w_excess_var) "
        f"  AS is_hit, "
        f"crush_excess_var / w_normal_daily_var AS crush_days, "
        f"realized_excess_var / w_normal_daily_var AS realized_excess_days "
        f"FROM j "
        f"WHERE corporate_action = 0 AND realized_var IS NOT NULL "
        f"AND w_support >= {OUTCOME_SUMMARY_MIN_SUPPORT!r})")


def _run(ch: Any, name: str, sql: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
    """Run one read-only query; on failure raise naming the query and the ClickHouse error. Quiet."""
    try:
        return ch.query(sql, parameters=params).result_rows
    except Exception as exc:   # noqa: BLE001 — re-raised with context, never swallowed
        raise RuntimeError(f"outcome-score query '{name}' failed: {exc}") from exc


# ── part queries ────────────────────────────────────────────────────────────────────────────────────

def _part1(ch: Any, is_earnings: int) -> List[tuple]:
    """(cutoff, n, hit_share, jump_share, fj_eq_crush, med_crush_over_excess) over windows with
    excess_days >= cutoff, one row per cutoff that has windows."""
    grid = ", ".join(repr(float(c)) for c in CUTOFF_GRID)
    sql = (
        f"{_scored_cte()} "
        f"SELECT cutoff, count() AS n, "
        f"countIf(is_hit = 1) / count() AS hit_share, "
        f"countIf(n_jump_days > 0) / count() AS jump_share, "
        f"countIf(first_jump_date = crush_date) / count() AS fj_eq_crush, "
        f"medianIf(crush_days / w_excess_days, crush_days IS NOT NULL AND w_excess_days != 0) AS med_ratio "
        f"FROM s ARRAY JOIN [{grid}] AS cutoff "
        f"WHERE is_earnings = {is_earnings} AND w_excess_days >= cutoff "
        f"GROUP BY cutoff ORDER BY cutoff")
    return _run(ch, f"part1 hit-rate-by-cutoff (is_earnings={is_earnings})", sql)


def _part2_band(ch: Any, band_expr: str, order_expr: str, cutoff: float, name: str) -> List[tuple]:
    """(band, n, hit_share) for the other group at the chosen cutoff, grouped by `band_expr`."""
    sql = (
        f"{_scored_cte()} "
        f"SELECT {band_expr} AS band, count() AS n, countIf(is_hit = 1) / count() AS hit_share "
        f"FROM s WHERE is_earnings = 0 AND w_excess_days >= {cutoff!r} "
        f"GROUP BY band ORDER BY {order_expr}")
    return _run(ch, name, sql)


def _part3_by_year(ch: Any, cutoff: float) -> List[tuple]:
    """(year, n, share with excess_days < cutoff) over the other-group surprises."""
    sql = (
        f"{_scored_cte()} "
        f"SELECT toYear(session_date) AS y, count() AS n, "
        f"countIf(w_excess_days < {cutoff!r}) / count() AS miss_share "
        f"FROM s WHERE is_earnings = 0 AND n_jump_days > 0 AND explained = 'none' "
        f"GROUP BY y ORDER BY y")
    return _run(ch, "part3 surprise miss-share by year", sql)


def _part3_list(ch: Any) -> List[SurpriseRow]:
    sql = (
        f"{_scored_cte()} "
        f"SELECT underlying, first_jump_date, max_abs_ret_adj, w_excess_days, session_date, end_td, "
        f"w_start_td "
        f"FROM s WHERE is_earnings = 0 AND n_jump_days > 0 AND explained = 'none' "
        f"AND first_jump_date >= %(since)s")
    rows = _run(ch, "part3 recent surprises", sql, {"since": RECENT_SINCE})
    return [SurpriseRow(str(r[0]).upper(), r[1], None if r[2] is None else float(r[2]),
                        float(r[3]), r[4], int(r[5]), int(r[6])) for r in rows]


def _part4_by_year(ch: Any, cutoff: float) -> List[tuple]:
    """(year, n windows at cutoff, false-alarm share) — the false-alarm rate among other-group windows
    at the chosen cutoff (not a hit and n_jump_days = 0), by year."""
    sql = (
        f"{_scored_cte()} "
        f"SELECT toYear(session_date) AS y, count() AS n, "
        f"countIf(is_hit = 0 AND n_jump_days = 0) / count() AS fa_share "
        f"FROM s WHERE is_earnings = 0 AND w_excess_days >= {cutoff!r} "
        f"GROUP BY y ORDER BY y")
    return _run(ch, "part4 false-alarm share by year", sql)


def _part4_list(ch: Any, cutoff: float) -> List[AlarmRow]:
    sql = (
        f"{_scored_cte()} "
        f"SELECT underlying, session_date, end_expiry, w_excess_days, crush_days, w_support "
        f"FROM s WHERE is_earnings = 0 AND w_excess_days >= {cutoff!r} AND is_hit = 0 "
        f"AND n_jump_days = 0 AND session_date >= %(since)s")
    rows = _run(ch, "part4 recent false alarms", sql, {"since": RECENT_SINCE})
    return [AlarmRow(str(r[0]).upper(), r[1], r[2], float(r[3]),
                     None if r[4] is None else float(r[4]), float(r[5])) for r in rows]


def _part5(ch: Any, cutoff: float) -> List[tuple]:
    """(is_earnings, year, n, median, p25, p75 of realized_excess_days/excess_days, share realized
    excess > priced excess) for windows at the chosen cutoff, per group per year."""
    sql = (
        f"{_scored_cte()} "
        f"SELECT is_earnings, toYear(session_date) AS y, count() AS n, "
        f"quantileExact(0.5)(realized_excess_days / w_excess_days) AS med, "
        f"quantileExact(0.25)(realized_excess_days / w_excess_days) AS p25, "
        f"quantileExact(0.75)(realized_excess_days / w_excess_days) AS p75, "
        f"countIf(realized_excess_var > w_excess_var) / count() AS over_share "
        f"FROM s WHERE w_excess_days >= {cutoff!r} "
        f"GROUP BY is_earnings, y ORDER BY is_earnings DESC, y")
    return _run(ch, "part5 priced vs realized", sql)


def _part6_counts(ch: Any, cutoff: float) -> List[tuple]:
    """(crush_date, year, distinct-underlying count) over other-group hits at the chosen cutoff."""
    sql = (
        f"{_scored_cte()} "
        f"SELECT crush_date, toYear(crush_date) AS y, uniqExact(underlying) AS names "
        f"FROM s WHERE is_earnings = 0 AND w_excess_days >= {cutoff!r} AND is_hit = 1 "
        f"GROUP BY crush_date")
    return _run(ch, "part6 crush-date name counts", sql)


def _market_wide_dates(ch: Any, dates: Sequence[date]) -> set:
    """The subset of `dates` that are market-wide — the same jump_daily rule as outcome_event's
    _read_market_wide: >= MARKET_WIDE_MIN_NAMES names with overnight_jump_adj = 1 that date."""
    if not dates:
        return set()
    sql = (
        f"SELECT trading_date FROM {JUMP_DAILY_TABLE} FINAL "
        f"WHERE method = %(m)s AND trading_date IN %(dates)s "
        f"GROUP BY trading_date HAVING countIf(overnight_jump_adj = 1) >= %(n)s")
    rows = _run(ch, "part6 market-wide dates", sql,
                {"m": JUMP_METHOD, "dates": list(dates), "n": MARKET_WIDE_MIN_NAMES})
    return {r[0] for r in rows}


# ── formatting helpers ──────────────────────────────────────────────────────────────────────────────

def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{float(x) * 100:.1f}%"


def _f(x: Optional[float], fmt: str = "{:.3f}") -> str:
    return "—" if x is None else fmt.format(float(x))


# ── the report ────────────────────────────────────────────────────────────────────────────────────

def print_score_report(ch: Any) -> None:
    print("iv_event outcome-score — detector scoring (out_v1 outcomes vs fwd_v1 windows)")
    print("Population: iv_event_outcome (out_v1) ⋈ iv_event_window (fwd_v1, ok); corporate_action=0, "
          "realized_var present, support >= {:g}.".format(OUTCOME_SUMMARY_MIN_SUPPORT))
    print("Unit: one window row — one window seen from one session.")

    # Part 1 — hit rate by cutoff, one table per group.
    earnings = _part1(ch, 1)
    other = _part1(ch, 0)
    for label, rows in (("earnings (own_earnings_date present)", earnings),
                        ("other (no own earnings)", other)):
        print(f"\n── Part 1. Hit rate by cutoff — {label} ──")
        print(f"    {'cutoff':>7s} {'n':>9s} {'hit':>8s} {'jump>0':>8s} {'fj=crush':>9s} "
              f"{'med crush/xs':>13s}")
        for cutoff, n, hit, jump, fj, med in rows:
            print(f"    {float(cutoff):>7.1f} {int(n):>9,d} {_pct(hit):>8s} {_pct(jump):>8s} "
                  f"{_pct(fj):>9s} {_f(med):>13s}")

    # Part 2 — the threshold.
    cutoff, reached = choose_cutoff([(c, hit) for c, _n, hit, _j, _fj, _m in other])
    print("\n── Part 2. The threshold ──")
    if reached:
        print(f"    Chosen cutoff = {cutoff:g} (smallest grid cutoff where the other group's hit share "
              f">= {HIT_SHARE_TARGET:.0%}).")
    else:
        print(f"    No cutoff reaches the other group's hit share >= {HIT_SHARE_TARGET:.0%}; "
              f"using {cutoff:g} for parts 3-6.")
    sup = _part2_band(ch, "multiIf(w_support < 250, '100-250', w_support < 1000, '250-1000', '1000+')",
                      "min(w_support)", cutoff, "part2 hit share by support band")
    std = _part2_band(ch, "multiIf(w_start_td <= 5, '0-5', w_start_td <= 20, '6-20', '21+')",
                      "min(w_start_td)", cutoff, "part2 hit share by start_td band")
    print(f"    Other group at cutoff {cutoff:g} — hit share by support band:")
    for band, n, hit in sup:
        print(f"        {str(band):>10s} {int(n):>9,d} {_pct(hit):>8s}")
    print(f"    Other group at cutoff {cutoff:g} — hit share by start_td band:")
    for band, n, hit in std:
        print(f"        {str(band):>6s} {int(n):>9,d} {_pct(hit):>8s}")

    # Part 3 — missed events (surprises).
    print("\n── Part 3. Missed events (surprises) — other group, n_jump_days>0 and explained='none' ──")
    print(f"    share with excess_days < {cutoff:g} (would be missed), by year:")
    print(f"        {'year':>6s} {'n':>8s} {'miss':>8s}")
    for y, n, miss in _part3_by_year(ch, cutoff):
        print(f"        {int(y):>6d} {int(n):>8,d} {_pct(miss):>8s}")
    recent = dedupe_smallest_start_td(_part3_list(ch))
    recent.sort(key=lambda r: r.first_jump_date, reverse=True)
    print(f"    {LIST_LIMIT} most recent since {RECENT_SINCE} "
          f"(one per underlying+first_jump_date, smallest start_td):")
    print(f"        {'underlying':>10s} {'first_jump':>11s} {'max|ret_adj|':>12s} {'excess_d':>9s} "
          f"{'session':>11s} {'end_td':>6s}")
    for sr in recent[:LIST_LIMIT]:
        print(f"        {sr.underlying:>10s} {str(sr.first_jump_date):>11s} "
              f"{_f(sr.max_abs_ret_adj):>12s} {_f(sr.excess_days, '{:.2f}'):>9s} "
              f"{str(sr.session_date):>11s} {sr.end_td:>6d}")

    # Part 4 — false alarms.
    print("\n── Part 4. False alarms — other group, excess_days>=cutoff, not a hit, n_jump_days=0 ──")
    print("    false-alarm share among other-group windows at the cutoff, by year:")
    print(f"        {'year':>6s} {'n':>8s} {'false':>8s}")
    for y, n, fa in _part4_by_year(ch, cutoff):
        print(f"        {int(y):>6d} {int(n):>8,d} {_pct(fa):>8s}")
    alarms = dedupe_largest_excess_days(_part4_list(ch, cutoff))
    alarms.sort(key=lambda r: r.excess_days, reverse=True)
    print(f"    {LIST_LIMIT} largest by excess_days since {RECENT_SINCE} "
          f"(one per underlying+end_expiry, largest excess_days):")
    print(f"        {'underlying':>10s} {'session':>11s} {'end_expiry':>11s} {'excess_d':>9s} "
          f"{'crush_d':>9s} {'support':>9s}")
    for ar in alarms[:LIST_LIMIT]:
        print(f"        {ar.underlying:>10s} {str(ar.session_date):>11s} {str(ar.end_expiry):>11s} "
              f"{_f(ar.excess_days, '{:.2f}'):>9s} {_f(ar.crush_days, '{:.2f}'):>9s} "
              f"{_f(ar.support, '{:.0f}'):>9s}")

    # Part 5 — priced vs realized.
    print("\n── Part 5. Priced vs realized — windows with excess_days>=cutoff, per group, by year ──")
    print(f"    {'group':>9s} {'year':>6s} {'n':>8s} {'median':>8s} {'p25':>8s} {'p75':>8s} "
          f"{'over':>8s}  (ratio = realized_excess_days / excess_days; over = realized > priced)")
    for is_earn, y, n, med, p25, p75, over in _part5(ch, cutoff):
        grp = "earnings" if int(is_earn) == 1 else "other"
        print(f"    {grp:>9s} {int(y):>6d} {int(n):>8,d} {_f(med):>8s} {_f(p25):>8s} {_f(p75):>8s} "
              f"{_pct(over):>8s}")

    # Part 6 — learned macro calendar.
    print("\n── Part 6. Learned macro calendar — other group, excess_days>=cutoff, hit ──")
    counts = _part6_counts(ch, cutoff)
    mw = _market_wide_dates(ch, [r[0] for r in counts])
    by_year: Dict[int, List[tuple]] = {}
    for crush_date, y, names in counts:
        by_year.setdefault(int(y), []).append((crush_date, int(names)))
    print(f"    top {TOP_DATES_PER_YEAR} crush dates per year by distinct-underlying count:")
    print(f"        {'date':>11s} {'wday':>4s} {'names':>6s} {'market-wide':>12s}")
    for y in sorted(by_year):
        top = sorted(by_year[y], key=lambda t: (-t[1], t[0]))[:TOP_DATES_PER_YEAR]
        for crush_date, names in top:
            wide = "yes" if crush_date in mw else "no"
            print(f"        {str(crush_date):>11s} {crush_date.strftime('%a'):>4s} {names:>6d} "
                  f"{wide:>12s}")
