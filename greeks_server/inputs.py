"""Session inputs: SOFR series and upcoming dividends (greeks-server-spec.md §3, §5.3, §9).

Loaded at each session open (and on demand for compute_greeks outside a session). Immutable once
built; the holder swaps the whole value atomically, so the compute thread and the router thread
never see a half-updated set. A load that can't produce usable inputs raises: the server exits and
systemd restarts it through the normal startup path.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Mapping, Optional

from greeks.domain import Dividend
from greeks.forwards.sofr import continuous_rate_from_sofr

log = logging.getLogger(__name__)

SOFR_LOOKBACK_DAYS = 30
SOFR_MAX_AGE_DAYS = 10


@dataclass(frozen=True)
class SessionInputs:
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
                source="fred",
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
    """The session's SessionInputs, swapped whole.

    Only a session load fills it: startup inside a session, or the compute loop at the open.
    An off-session compute_greeks loads fresh and does not cache, so an early-morning request
    can't leave the open running on inputs loaded before the 08:45 SOFR job.
    """

    def __init__(self, loader: Loader) -> None:
        self._loader = loader
        self._current: Optional[SessionInputs] = None

    def current(self) -> Optional[SessionInputs]:
        return self._current

    def load_session(self, session_date: date) -> SessionInputs:
        """Load for a session that is open now, and keep it."""
        inputs = self._loader(session_date)
        self._current = inputs
        return inputs

    def session_inputs(self, session_date: date) -> SessionInputs:
        """The open's inputs: reuse a load already made inside this session (startup), else load."""
        cur = self._current
        if cur is not None and cur.session_date == session_date:
            return cur
        return self.load_session(session_date)

    def for_request(self, session_date: date, in_session: bool) -> SessionInputs:
        """compute_greeks: the session's inputs while it is open, a fresh uncached load otherwise."""
        cur = self._current
        if in_session and cur is not None and cur.session_date == session_date:
            return cur
        return self._loader(session_date)
