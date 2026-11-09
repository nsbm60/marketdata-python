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
from greeks.forwards.sofr import continuous_rate_from_sofr

log = logging.getLogger(__name__)

SOFR_LOOKBACK_DAYS = 30
SOFR_MAX_AGE_DAYS = 10
# Written by the Scala BatchService sofr job from the NY Fed API (the publisher of SOFR).
# The validation-phase rows are source='fred' (and 'fixture'); the server reads only these.
SOFR_SOURCE = "nyfed"


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
    """
    from greeks.ch import load_sofr_series_from_ch, load_upcoming_dividends

    lock = threading.Lock()
    client_box: list[object] = []

    def load(session_date: date) -> SessionInputs:
        with lock:
            if not client_box:
                client_box.append(get_client())
            client = client_box[0]
            sofr = load_sofr_series_from_ch(
                client,  # type: ignore[arg-type]
                start=session_date - timedelta(days=SOFR_LOOKBACK_DAYS),
                end=session_date,
                source=SOFR_SOURCE,
            )
            validate_sofr(sofr, session_date)
            divs = load_upcoming_dividends(client, after=session_date)  # type: ignore[arg-type]
        log.info(
            "session inputs for %s: %d SOFR fixes (newest %s), dividends for %d symbols",
            session_date, len(sofr), max(sofr), len(divs),
        )
        return SessionInputs(session_date=session_date, sofr=dict(sofr), dividends=divs)

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
