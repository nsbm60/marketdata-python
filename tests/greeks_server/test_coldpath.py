"""ColdPath.price() — the get_greeks cold path, driven directly with a scripted MDS transport.

The cold path makes ONE call: get_quotes, carrying the requested OSIs plus their distinct
underlyings, as of the caller's `at`. From the one reply: an option is current if it is in `quotes`
and its quote_ts is at or after the reply's session_open_ts + 1 min (else no_current_quote); the spot
is the underlying's quote mid (an underlying in `missing` -> missing_spot); everything is priced as
of the reply's as_of. If the call fails, every contract fails mds_unavailable.
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
MU_C = "MU261120C00120000"


def et_ms(h: int, m: int, s: int = 0, day: date = DAY) -> int:
    return int(datetime.combine(day, time(h, m, s), tzinfo=ET).astimezone(timezone.utc).timestamp() * 1000)


SESSION_OPEN = et_ms(9, 30)


def _inputs() -> InputsHolder:
    def loader(d: date) -> SessionInputs:
        return SessionInputs(session_date=d, sofr={d - timedelta(days=i): 0.043 for i in range(1, 20)}, dividends={})
    return InputsHolder(loader)


def transport(*, quotes: dict[str, tuple[float, float, int]], session_open: int | None, as_of: int | None,
              raise_op: bool = False, log: list[dict] | None = None):
    """A ColdPath transport serving get_quotes from a seeded dict. `quotes` is sym -> (bid, ask,
    quote_ts); a symbol not in it is returned in `missing`. `raise_op` makes the call raise (a
    timeout). `session_open`/`as_of` None omit those fields from the reply."""
    def send(req: dict[str, Any]) -> dict[str, Any]:
        if log is not None:
            log.append(req)
        assert req["op"] == "get_quotes", f"the cold path must make only get_quotes, got {req['op']}"
        if raise_op:
            raise TimeoutError("get_quotes timed out")
        want = set(req["symbols"])
        q = {s: {"bid": b, "ask": a, "bid_size": 1, "ask_size": 1, "quote_ts": t}
             for s, (b, a, t) in quotes.items() if s in want}
        data: dict[str, Any] = {"quotes": q, "missing": {s: "no_quote" for s in want if s not in quotes}}
        if session_open is not None:
            data["session_open_ts"] = session_open
        if as_of is not None:
            data["as_of"] = as_of
        return {"ok": True, "data": data}
    return ColdPath(send, _inputs(), M)


def test_spot_is_the_underlying_quote_mid_and_priced_as_of_is_the_reply_as_of() -> None:
    qts, as_of = et_ms(11, 0), et_ms(11, 0, 5)
    cold = transport(quotes={NVDA_C: (9.0, 9.2, qts), "NVDA": (180.9, 181.1, qts)},
                     session_open=SESSION_OPEN, as_of=as_of)
    results, reply_as_of = cold.price([NVDA_C], "latest")
    (r,) = results
    assert r.message.status is RowStatus.SUCCESS
    assert r.message.spot == 181.0 and r.message.spot_kind is PriceKind.MID and r.message.spot_ts == qts
    assert r.message.price == 9.1 and r.message.price_kind is PriceKind.MID
    assert r.message.priced_as_of == as_of   # one time for pricing: the reply's as_of
    assert reply_as_of == as_of              # and the cold path surfaces it for the reply


def test_at_is_passed_through_unchanged_both_forms_in_one_call_with_the_underlyings() -> None:
    log: list[dict] = []
    cold = transport(quotes={NVDA_C: (9.0, 9.2, et_ms(11, 0)), "NVDA": (180.9, 181.1, et_ms(11, 0))},
                     session_open=SESSION_OPEN, as_of=et_ms(11, 0, 5), log=log)
    cold.price([NVDA_C], "latest")
    cold.price([NVDA_C], 1762000000000)
    assert [r["at"] for r in log] == ["latest", 1762000000000]        # passed through unchanged
    assert set(log[0]["symbols"]) == {NVDA_C, "NVDA"}                  # OSIs + their distinct underlyings
    assert len(log) == 2                                              # one get_quotes call per price()


def test_current_quote_rule_uses_the_replys_session_open_ts() -> None:
    # A non-standard session open in the reply (10:00, not the real 09:30): the rule follows the
    # reply's value, not any calendar of the cold path's own.
    odd_open = et_ms(10, 0)
    below = et_ms(10, 0, 30)   # 30s after the reply's open -> below open + 1 min
    above = et_ms(10, 1, 30)   # 90s after -> at/after open + 1 min
    stale = transport(quotes={NVDA_C: (9.0, 9.2, below), "NVDA": (180.9, 181.1, below)},
                      session_open=odd_open, as_of=et_ms(10, 2))
    (r,), _ = stale.price([NVDA_C], "latest")
    assert r.message.status is RowStatus.FAILURE and r.message.reason is FailureReason.NO_CURRENT_QUOTE

    fresh = transport(quotes={NVDA_C: (9.0, 9.2, above), "NVDA": (180.9, 181.1, above)},
                      session_open=odd_open, as_of=et_ms(10, 2))
    (r2,), _ = fresh.price([NVDA_C], "latest")
    assert r2.message.status is RowStatus.SUCCESS


def test_underlying_in_missing_gives_missing_spot() -> None:
    qts = et_ms(11, 0)
    cold = transport(quotes={NVDA_C: (9.0, 9.2, qts)},   # NVDA absent -> returned in `missing`
                     session_open=SESSION_OPEN, as_of=et_ms(11, 0, 5))
    (r,), _ = cold.price([NVDA_C], "latest")
    assert r.message.status is RowStatus.FAILURE and r.message.reason is FailureReason.MISSING_SPOT


def test_option_not_in_quotes_is_no_current_quote() -> None:
    qts = et_ms(11, 0)
    cold = transport(quotes={"NVDA": (180.9, 181.1, qts)},   # option absent -> not two-sided per MDS
                     session_open=SESSION_OPEN, as_of=et_ms(11, 0, 5))
    (r,), _ = cold.price([NVDA_C], "latest")
    assert r.message.status is RowStatus.FAILURE and r.message.reason is FailureReason.NO_CURRENT_QUOTE


def test_a_failed_get_quotes_fails_every_contract_as_mds_unavailable() -> None:
    cold = transport(quotes={}, session_open=SESSION_OPEN, as_of=et_ms(11, 0), raise_op=True)
    results, reply_as_of = cold.price([NVDA_C, MU_C], "latest")
    out = {r.osi: r.message for r in results}
    assert all(m.reason is FailureReason.MDS_UNAVAILABLE for m in out.values()) and len(out) == 2
    assert reply_as_of is None   # no as_of to surface on a failed call


def test_reply_missing_as_of_is_malformed_fails_mds_unavailable() -> None:
    # A reply with no as_of is malformed: never price as of 1970 -> every contract mds_unavailable.
    cold = transport(quotes={NVDA_C: (9.0, 9.2, et_ms(11, 0)), "NVDA": (180.9, 181.1, et_ms(11, 0))},
                     session_open=SESSION_OPEN, as_of=None)
    (r,), reply_as_of = cold.price([NVDA_C], "latest")
    assert r.message.reason is FailureReason.MDS_UNAVAILABLE and reply_as_of is None


def test_the_old_ops_and_the_calendar_are_gone() -> None:
    src = (Path(__file__).resolve().parents[2] / "greeks_server" / "coldpath.py").read_text()
    assert "get_option_snapshots" not in src
    assert "get_stock_quotes" not in src
    assert "get_equity_prices_at" not in src
    assert "SessionCache" not in src and "greeks_server.calendar" not in src
