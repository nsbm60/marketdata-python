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


# -- --width-test aggregation (pure, no network) ------------------------------


def test_parse_widths_rejects_nonpositive() -> None:
    assert probe._parse_widths("16, 24,32 ,48") == [16, 24, 32, 48]
    with pytest.raises(ValueError):
        probe._parse_widths("16,0,32")


def test_percentile_nearest_rank() -> None:
    vals = [0.1, 0.2, 0.3, 0.4]  # already sorted
    assert probe._percentile(vals, 0.50) == 0.3   # int(0.5*4)=2
    assert probe._percentile(vals, 0.95) == 0.4   # int(0.95*4)=3
    assert probe._percentile([], 0.5) == 0.0


def test_summarize_aggregates_across_contracts() -> None:
    from option_archive.quotes import BenchPull, PageStat

    # two contracts: one clean 2-page pull, one that 200s then 5xxs (partial, no retry)
    a = BenchPull(
        pages=2, quotes_fetched=100, network_s=4.0, local_s=1.0,
        page_stats=[PageStat(0.10, 200, 60), PageStat(0.30, 200, 40)],
    )
    b = BenchPull(
        pages=1, quotes_fetched=50, network_s=2.0, local_s=0.0,
        page_stats=[PageStat(0.20, 200, 50), PageStat(0.40, 503, 0)],
    )
    m = probe._summarize(wall=5.0, pulls=[a, b])
    assert m["contracts"] == 2.0 and m["pages"] == 3.0
    assert m["c_per_s"] == 2 / 5.0 and m["pg_per_s"] == 3 / 5.0
    # latencies sorted: [0.10,0.20,0.30,0.40] -> p50 idx2=0.30, p95 idx3=0.40
    assert m["p50_ms"] == 300.0 and m["p95_ms"] == 400.0
    assert m["n429"] == 0.0 and m["n5xx"] == 1.0
    assert m["net_s"] == 6.0 and m["cpu_s"] == 1.0
    assert m["net_pct"] == pytest.approx(600 / 7) and m["cpu_pct"] == pytest.approx(100 / 7)


def test_summarize_counts_429_and_formats_line() -> None:
    from option_archive.quotes import BenchPull, PageStat

    b = BenchPull(0, 0, 1.5, 0.0, [PageStat(0.05, 429, 0)])
    m = probe._summarize(wall=1.0, pulls=[b])
    assert m["n429"] == 1.0 and m["n5xx"] == 0.0 and m["pages"] == 0.0
    line = probe._format_width_line(32, m)
    assert line.startswith("width=32 ") and "429=1" in line and "net " in line and "cpu " in line
