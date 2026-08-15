"""Time-to-expiry: ACT/365 calendar, second resolution, 16:00 ET expiry."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_SECONDS_PER_ACT365_YEAR = 365.0 * 86400.0


def expiry_instant_utc(expiry_date: date, expiry_time_et: str = "16:00") -> datetime:
    """Equity option expiry as UTC datetime (date at HH:MM America/New_York)."""
    hour_s, minute_s = expiry_time_et.split(":", 1)
    local = datetime.combine(
        expiry_date,
        time(hour=int(hour_s), minute=int(minute_s)),
        tzinfo=_ET,
    )
    return local.astimezone(timezone.utc)


def years_to_expiry(
    as_of_utc: datetime,
    expiry_date: date,
    *,
    expiry_time_et: str = "16:00",
) -> float:
    """ACT/365 year fraction from ``as_of_utc`` to expiry, second resolution.

    ``as_of_utc`` must be timezone-aware (UTC or convertible). Returns 0.0 if
    at or past expiry.
    """
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    as_of = as_of_utc.astimezone(timezone.utc)
    expiry = expiry_instant_utc(expiry_date, expiry_time_et)
    seconds = (expiry - as_of).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / _SECONDS_PER_ACT365_YEAR


def years_from_seconds(seconds: float) -> float:
    """Convert a positive duration in seconds to ACT/365 years."""
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    return seconds / _SECONDS_PER_ACT365_YEAR
