"""ColdPath.price() — the get_greeks cold path, driven directly with a scripted MDS transport.

The stock comes from get_stock_quotes ("at": "latest"): spot = the stock quote mid, timed at its
quote_ts, never a trade. Every current contract is priced as of the stock reply's as_of. An MDS call
that raises fails only the contracts that depended on it; the rest of the request is still answered.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from greeks.domain import FailureReason, RowStatus
from greeks_server.coldpath import ColdPath
from greeks_server.compute import Methodology
from greeks_server.inputs import InputsHolder, SessionInputs
from greeks_server.price import PriceKind

ET = ZoneInfo("America/New_York")
DAY = date(2026, 11, 13)  # Friday, before the 2026-11-20 expiry; EST
M = Methodology("v2", 15.0, "16:00")

NVDA_C = "NVDA261120C00180000"
NVDA_STALE = "NVDA261120C00300000"
MU_C = "MU261120C00120000"


def et_ms(h: int, m: int, s: int = 0, day: date = DAY) -> int:
    return int(datetime.combine(day, time(h, m, s), tzinfo=ET).astimezone(timezone.utc).timestamp() * 1000)


SESSION_OPEN = et_ms(9, 30)


def _inputs() -> InputsHolder:
    def loader(d: date) -> SessionInputs:
        return SessionInputs(session_date=d, sofr={d - timedelta(days=i): 0.043 for i in range(1, 20)}, dividends={})
    return InputsHolder(loader)


def quote(bid: float, ask: float, at: int) -> dict[str, Any]:
    return {"quote": {"bid": bid, "ask": ask, "at": at}}


def transport(*, snaps: dict[str, dict], session_open: int, stock: dict[str, tuple[float, float, int]],
              as_of: int, raise_ops: tuple[str, ...] = (), log: list[dict] | None = None):
    """A ColdPath transport serving get_option_snapshots + get_stock_quotes from seeded dicts.
    `stock` is sym -> (bid, ask, quote_ts). An op in `raise_ops` raises (a timeout)."""
    def send(req: dict[str, Any]) -> dict[str, Any]:
        if log is not None:
            log.append(req)
        op = req["op"]
        if op in raise_ops:
            raise TimeoutError(f"{op} timed out")
        if op == "get_option_snapshots":
            want = set(req["symbols"])
            return {"ok": True, "data": {"snapshots": {o: s for o, s in snaps.items() if o in want},
                                         "session_open_ts": session_open}}
        if op == "get_stock_quotes":
            want = set(req["symbols"])
            quotes = {s: {"bid": b, "ask": a, "quote_ts": t} for s, (b, a, t) in stock.items() if s in want}
            return {"ok": True, "data": {"quotes": quotes, "missing": [s for s in want if s not in stock], "as_of": as_of}}
        raise AssertionError(f"unexpected op {op}")  # get_equity_prices_at must never be called
    return ColdPath(send, _inputs(), M)


def test_spot_is_the_stock_quote_mid_timed_at_its_quote_ts() -> None:
    qts, stock_ts, as_of = et_ms(11, 0), et_ms(11, 0, 2), et_ms(11, 0, 5)
    cold = transport(
        snaps={NVDA_C: quote(9.0, 9.2, qts)}, session_open=SESSION_OPEN,
        stock={"NVDA": (180.9, 181.1, stock_ts)}, as_of=as_of)
    (r,) = cold.price([NVDA_C])
    assert r.message.status is RowStatus.SUCCESS
    assert r.message.spot == 181.0 and r.message.spot_kind is PriceKind.MID and r.message.spot_ts == stock_ts
    assert r.message.price == 9.1 and r.message.price_kind is PriceKind.MID
    assert r.message.priced_as_of == as_of   # one time for pricing: the stock reply's as_of


def test_underlying_in_missing_gives_missing_spot() -> None:
    qts = et_ms(11, 0)
    cold = transport(
        snaps={NVDA_C: quote(9.0, 9.2, qts)}, session_open=SESSION_OPEN,
        stock={}, as_of=et_ms(11, 0, 5))   # NVDA not quoted -> reply lists it in `missing`
    (r,) = cold.price([NVDA_C])
    assert r.message.status is RowStatus.FAILURE and r.message.reason is FailureReason.MISSING_SPOT


def test_option_quote_before_session_open_plus_one_minute_is_no_current_quote() -> None:
    log: list[dict] = []
    at = et_ms(9, 30, 40)  # 09:30:40 — after the open but before open + 1 min
    cold = transport(
        snaps={NVDA_C: quote(9.0, 9.2, at)}, session_open=SESSION_OPEN,
        stock={"NVDA": (180.9, 181.1, at)}, as_of=et_ms(9, 31), log=log)
    (r,) = cold.price([NVDA_C])
    assert r.message.status is RowStatus.FAILURE and r.message.reason is FailureReason.NO_CURRENT_QUOTE
    assert not any(req["op"] == "get_stock_quotes" for req in log)  # no stock lookup for a stale quote


def test_mds_timeout_fails_only_the_dependent_contracts_and_answers_the_rest() -> None:
    # get_stock_quotes times out: the current contract (which needed stock) -> mds_unavailable; the
    # stale contract (no stock needed) is still answered as no_current_quote.
    fresh_at, stale_at = et_ms(11, 0), SESSION_OPEN - 60_000
    cold = transport(
        snaps={NVDA_C: quote(9.0, 9.2, fresh_at), NVDA_STALE: quote(0.1, 0.2, stale_at)},
        session_open=SESSION_OPEN, stock={"NVDA": (180.9, 181.1, fresh_at)}, as_of=et_ms(11, 0, 5),
        raise_ops=("get_stock_quotes",))
    out = {r.osi: r.message for r in cold.price([NVDA_C, NVDA_STALE])}
    assert out[NVDA_C].reason is FailureReason.MDS_UNAVAILABLE
    assert out[NVDA_STALE].reason is FailureReason.NO_CURRENT_QUOTE
    assert len(out) == 2  # the whole request is still answered


def test_option_snapshots_timeout_fails_every_contract_as_mds_unavailable() -> None:
    cold = transport(
        snaps={}, session_open=SESSION_OPEN, stock={"NVDA": (180.9, 181.1, et_ms(11, 0))},
        as_of=et_ms(11, 0), raise_ops=("get_option_snapshots",))
    out = {r.osi: r.message for r in cold.price([NVDA_C, MU_C])}
    assert all(m.reason is FailureReason.MDS_UNAVAILABLE for m in out.values()) and len(out) == 2


def test_no_call_to_get_equity_prices_at_remains() -> None:
    src = (Path(__file__).resolve().parents[2] / "greeks_server" / "coldpath.py").read_text()
    assert "get_equity_prices_at" not in src
