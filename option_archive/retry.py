"""Retry-with-backoff for vendor REST calls — the same discipline ``_download_day``
uses for the S3 transfer, applied to the fetches that had none.

A dropped Massive connection (``httpx.RemoteProtocolError``) killed a run inside the
pooled enumerate fetch, which had no retry. This wraps the vendor calls: 8 attempts,
exponential backoff, retry only on TRANSIENT failures — transport errors, HTTP 429,
and 5xx — then raise. A non-transient error (a 4xx that isn't 429) raises immediately.

Small helper, not a framework. The retryable predicate is duck-typed so it covers
both vendor clients without importing their exception modules: Massive/greeks raises
``httpx`` errors; Alpaca-py raises ``requests`` errors and ``alpaca…APIError``
(a ``requests`` transport error is an ``OSError`` subclass; ``APIError`` carries a
``.status_code``).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, TypeVar

import httpx

log = logging.getLogger("option_archive.retry")

# 8 attempts, 5s base doubling to a 120s (~2 min) cap → ~6.6 min total ride-out
# (sleeps 5+10+20+40+80+120+120) so a sustained 5xx/429 episode stalls the day
# rather than killing it on one contract's patience.
_ATTEMPTS = 8
_BACKOFF_BASE_SEC = 5.0
_BACKOFF_MAX_SEC = 120.0

T = TypeVar("T")

# Per-day retry tally. Retries are routine self-healing (logged at DEBUG, invisible
# at the service's INFO), so the operator-facing signal is a COUNT, not warnings:
# the archive resets it at the start of each _process_day and reads count() into the
# day log line + ledger. list.append is atomic under the GIL; reset()/count() run on
# the main thread between phases (pools joined), so no lock is needed — and
# `import threading` is banned here anyway (process-fleet rule).
_retry_tally: list[int] = []


def reset_retry_count() -> None:
    _retry_tally.clear()


def retry_count() -> int:
    return len(_retry_tally)


def _status_of(e: Exception) -> Optional[int]:
    """HTTP status carried by the error, if any — from ``.status_code``
    (alpaca ``APIError``) or ``.response.status_code`` (``httpx.HTTPStatusError`` /
    ``requests`` ``HTTPError``)."""
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc
    sc = getattr(getattr(e, "response", None), "status_code", None)
    return sc if isinstance(sc, int) else None


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, httpx.TransportError):  # Massive: conn drop/reset, timeout, RemoteProtocolError
        return True
    status = _status_of(e)
    if status is not None:  # any HTTP error (httpx / requests / alpaca): only 429 + 5xx are transient
        return status == 429 or status >= 500
    if isinstance(e, OSError):  # requests transport errors are RequestException < IOError; socket errors too
        return True
    return False


def with_retry(fn: Callable[[], T], *, what: str) -> T:
    """Call ``fn()``; on a transient vendor error retry up to ``_ATTEMPTS`` with
    exponential backoff. Non-transient errors, and the final attempt, raise."""
    backoff = _BACKOFF_BASE_SEC
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — classified by _is_retryable, re-raised otherwise
            # Log type + HTTP status only, never the exception str — an
            # httpx.HTTPStatusError embeds the full request URL, which carries apiKey.
            if _is_retryable(e) and attempt < _ATTEMPTS:
                _retry_tally.append(1)
                # Routine self-healing — DEBUG, invisible at the service's INFO level.
                log.debug("%s: attempt %d/%d failed (%s status=%s); retry in %.0fs",
                          what, attempt, _ATTEMPTS, type(e).__name__, _status_of(e), backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_SEC)
                continue
            if _is_retryable(e):  # retryable but out of attempts — re-raise; the CALLER decides
                # (enumerate/tape abort the day; the quote phase records + skips the contract).
                log.error("%s: exhausted %d attempts (%s status=%s); propagating to caller",
                          what, _ATTEMPTS, type(e).__name__, _status_of(e))
            raise
    raise AssertionError("unreachable: loop returns or raises")
