"""
ml/shared/utils.py

Common utilities shared across ML modules.
"""

from datetime import datetime, timezone


def utc_dt(dt: datetime) -> datetime:
    """Ensure a datetime is UTC-aware. Assumes naive datetimes are UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
