"""probe: near-ATM contract pick, and per-check status/latency/redaction + answered."""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import httpx
import pytest

from option_archive import probe


# -- _pick_contract -----------------------------------------------------------


def test_pick_contract_near_atm_nearest_expiry() -> None:
    results = [
        {"ticker": "O:SPY220617C00300000", "expiration_date": "2022-06-17", "strike_price": 300, "contract_type": "call"},
        {"ticker": "O:SPY220617C00380000", "expiration_date": "2022-06-17", "strike_price": 380, "contract_type": "call"},
        {"ticker": "O:SPY220617C00460000", "expiration_date": "2022-06-17", "strike_price": 460, "contract_type": "call"},
        {"ticker": "O:SPY221216C00380000", "expiration_date": "2022-12-16", "strike_price": 380, "contract_type": "call"},  # later expiry
        {"ticker": "O:SPY220617P00380000", "expiration_date": "2022-06-17", "strike_price": 380, "contract_type": "put"},   # put
    ]
    assert probe._pick_contract(results, date(2022, 6, 13)) == "SPY220617C00380000"


def test_pick_contract_empty_is_none() -> None:
    assert probe._pick_contract([], date(2022, 6, 13)) is None


# -- quote check (fake httpx client) ------------------------------------------


class _FakeResp:
    def __init__(self, status: int, payload: Optional[dict[str, Any]] = None) -> None:
        self.status_code = status
        self._p = payload or {}

    def json(self) -> dict[str, Any]:
        return self._p


class _FakeClient:
    def __init__(self, resp: Optional[_FakeResp] = None, exc: Optional[Exception] = None) -> None:
        self._resp, self._exc = resp, exc

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *a: Any) -> bool:
        return False

    def get(self, url: str, params: Any = None) -> _FakeResp:
        if self._exc is not None:
            raise self._exc
        assert self._resp is not None
        return self._resp


def _patch(monkeypatch: pytest.MonkeyPatch, resp: Optional[_FakeResp] = None, exc: Optional[Exception] = None) -> None:
    monkeypatch.setattr(probe.httpx, "Client", lambda **k: _FakeClient(resp=resp, exc=exc))


def test_quotes_check_200_answers_and_redacts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, resp=_FakeResp(200, {"results": [1, 2, 3]}))
    ok, line = probe._quotes_check("SECRETKEY", "SPY220617C00380000", date(2022, 6, 13), 10)
    assert ok is True
    assert "status=200" in line and "rows=3" in line and "SPY220617C00380000@2022-06-13" in line
    assert "SECRETKEY" not in line and "apiKey" not in line


def test_quotes_check_502_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, resp=_FakeResp(502))
    ok, line = probe._quotes_check("SECRETKEY", "SPY220617C00380000", date(2022, 6, 13), 50000)
    assert ok is True and "status=502" in line and "rows=-" in line
    assert "SECRETKEY" not in line


def test_quotes_check_no_response_logs_type_not_str(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, exc=httpx.ConnectError("https://api.massive.com/...apiKey=SECRETKEY refused"))
    ok, line = probe._quotes_check("SECRETKEY", "SPY220617C00380000", date(2022, 6, 13), 10)
    assert ok is False and "NO-RESPONSE" in line and "ConnectError" in line
    # the exception str (which could carry URL + key) is NOT logged — type name only
    assert "SECRETKEY" not in line and "apiKey" not in line and "refused" not in line


def test_quotes_check_no_contract_is_not_answered() -> None:
    ok, line = probe._quotes_check("K", None, date(2022, 6, 13), 10)
    assert ok is False and "NOT-ANSWERED" in line


def test_s3_head_not_configured_is_not_answered() -> None:
    ok, line = probe._s3_head_check(None, date(2022, 6, 13))
    assert ok is False and "NOT-CONFIGURED" in line
