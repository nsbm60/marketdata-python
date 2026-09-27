"""option_archive — production option trade/quote history backfill and rescue.

Build-out of the validated ``greeks/pull``, ``greeks/queue`` and ``greeks/ch``
primitives into a resumable, oldest-first archive of option trades (plus the
quote at each trade) for a screened universe. Outlives the one-time backfill: it
also runs incremental capture forever.

Spec (system of record):
``Scala/MarketData/docs/plans/backfill-rescue.md``.

Conventions inherited from ``greeks`` (enforced by ``scripts/check_option_archive.sh``):
mypy --strict; no ``asyncio`` / ``threading`` (process fleet only); frozen
dataclasses; enums for every status/source; no silent defaults.
"""

from __future__ import annotations
