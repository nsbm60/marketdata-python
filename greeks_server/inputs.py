"""Per-date inputs: SOFR series and upcoming dividends (greeks-server-spec.md §3, §5.3, §9).

A set of greeks priced as of T needs the rate and dividends for T's date. There is no session: the
holder loads once per date and caches the immutable result, so the compute thread and the router
thread (cold path, get_greeks) can both ask for any date and share the answer. A load that can't
produce usable inputs raises at the point of use.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Mapping

from greeks.domain import Dividend
from greeks.forwards.sofr import (
    combined_overnight_series,
    continuous_rate_from_sofr,
    tbill_to_overnight_act360,
)

log = logging.getLogger(__name__)

SOFR_LOOKBACK_DAYS = 30
SOFR_MAX_AGE_DAYS = 10
# Written by the Scala BatchService sofr job from the NY Fed API (the publisher of SOFR).
# The validation-phase rows are source='fred' (and 'fixture'); the server reads only these.
SOFR_SOURCE = "nyfed"
# Pre-SOFR dates (before the first SOFR fix, 2018-04-02) take the overnight rate from the 1-month
# Treasury bill in trading.treasury_yields (Scala TreasuryYieldsJob). Live greeks never reach these
# dates, so the live server only ever sees SOFR.
TBILL_SOURCE = "massive"


@dataclass(frozen=True)
class SessionInputs:
    # session_date is the date these inputs are for — the ET date of the moment T being priced.
    session_date: date
    sofr: Mapping[date, float]
    dividends: Mapping[str, tuple[Dividend, ...]]
    # Continuous rate depends only on (session_date, expiry): memoized per expiry. Owned by this
    # value, so a reload starts a fresh memo. Dict writes are atomic in CPython; two threads that
    # race compute the same number.
    _rates: dict[date, float] = field(default_factory=dict, compare=False, repr=False)

    def rate_for(self, expiry: date) -> float:
        r = self._rates.get(expiry)
        if r is None:
            r = continuous_rate_from_sofr(self.sofr, self.session_date, expiry)
            self._rates[expiry] = r
        return r

    def dividends_for(self, underlying: str) -> tuple[Dividend, ...]:
        return self.dividends.get(underlying.upper(), ())


Loader = Callable[[date], SessionInputs]


def validate_sofr(series: Mapping[date, float], session_date: date) -> None:
    """Raise unless there is a fix on or before session_date no older than SOFR_MAX_AGE_DAYS."""
    prior = [d for d in series if d <= session_date]
    if not prior:
        raise ValueError(f"no SOFR fix on or before {session_date}")
    newest = max(prior)
    if (session_date - newest).days > SOFR_MAX_AGE_DAYS:
        raise ValueError(
            f"newest SOFR fix {newest} is more than {SOFR_MAX_AGE_DAYS} days before {session_date}"
            " — is the BatchService sofr job running?"
        )


def clickhouse_loader(get_client: Callable[[], object]) -> Loader:
    """Loader backed by ClickHouse (greeks/ch.py helpers on the canonical client).

    One client, used under a lock: the compute thread (at open) and the router thread
    (compute_greeks off-session) may both load, and a clickhouse-connect client is not safe for
    concurrent queries.

    The overnight series is SOFR on and after the first SOFR fix (read from the table, not hard-coded)
    and the ACT/360-converted 1-month T-bill before it, so historical greeks price back to 2016. A
    post-SOFR gap is still an error — the T-bill never fills one. Live greeks only ever request recent
    dates, whose window never reaches before the first SOFR fix, so the live server only sees SOFR.
    """
    from greeks.ch import (
        first_sofr_date_from_ch,
        load_sofr_series_from_ch,
        load_tbill_1m_series_from_ch,
        load_upcoming_dividends,
    )

    lock = threading.Lock()
    client_box: list[object] = []
    first_sofr_box: list[date] = []  # the first SOFR fix date, stable — read once per loader

    def load(session_date: date) -> SessionInputs:
        window_start = session_date - timedelta(days=SOFR_LOOKBACK_DAYS)
        with lock:
            if not client_box:
                client_box.append(get_client())
            client = client_box[0]
            if not first_sofr_box:
                first_sofr_box.append(first_sofr_date_from_ch(client, source=SOFR_SOURCE))  # type: ignore[arg-type]
            first_sofr = first_sofr_box[0]
            sofr = load_sofr_series_from_ch(
                client,  # type: ignore[arg-type]
                start=window_start,
                end=session_date,
                source=SOFR_SOURCE,
                required=False,  # a pre-SOFR window has no SOFR rows; T-bills fill it below
            )
            # Only the window dates before the first SOFR fix are filled from the T-bill.
            tbill_overnight: dict[date, float] = {}
            if window_start < first_sofr:
                pre_end = min(session_date, first_sofr - timedelta(days=1))
                bills = load_tbill_1m_series_from_ch(
                    client,  # type: ignore[arg-type]
                    start=window_start, end=pre_end, source=TBILL_SOURCE,
                )
                tbill_overnight = {d: tbill_to_overnight_act360(y) for d, y in bills.items()}
            divs = load_upcoming_dividends(client, after=session_date)  # type: ignore[arg-type]

        combined = combined_overnight_series(sofr, tbill_overnight, first_sofr)
        # A date on or after the first SOFR fix must have a recent SOFR fix (a gap is an error, as
        # before); a pre-SOFR date must have a T-bill yield on or before it (named if absent). Nothing
        # is defaulted.
        if session_date >= first_sofr:
            validate_sofr(sofr, session_date)
        elif not any(d <= session_date for d in tbill_overnight):
            raise ValueError(
                f"no 1-month T-bill yield on or before {session_date}"
                " (pre-SOFR date needs a bill rate — is the treasury-yields job loaded?)"
            )

        has_sofr, has_tbill = bool(sofr), bool(tbill_overnight)
        rate_src = "TBILL1M+SOFR" if (has_sofr and has_tbill) else ("SOFR" if has_sofr else "TBILL1M")
        log.info(
            "session inputs for %s: rate=%s, %d overnight fixes (newest %s), dividends for %d symbols",
            session_date, rate_src, len(combined), max(combined), len(divs),
        )
        return SessionInputs(session_date=session_date, sofr=dict(combined), dividends=divs)

    return load


class InputsHolder:
    """SessionInputs cached by date, loaded on first use for that date.

    Both the compute thread and the router thread ask for arbitrary dates and share the result.
    The SOFR fix for a date is stable once published, so caching it is safe; a restart (roughly
    daily) clears the cache. The loader has its own lock, so it is called outside this one.
    """

    def __init__(self, loader: Loader) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._by_date: dict[date, SessionInputs] = {}

    def for_date(self, d: date) -> SessionInputs:
        """The inputs for date ``d``: the cached load if present, else a fresh one (cached). Raises
        if the load can't produce usable inputs (e.g. no recent SOFR fix)."""
        with self._lock:
            cur = self._by_date.get(d)
        if cur is not None:
            return cur
        inputs = self._loader(d)
        with self._lock:
            return self._by_date.setdefault(d, inputs)
