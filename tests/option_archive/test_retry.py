"""with_retry: transient vendor errors retried, non-transient raised immediately."""

from __future__ import annotations

import httpx
import pytest

from option_archive.retry import _ATTEMPTS, with_retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("option_archive.retry.time.sleep", lambda *_a: None)


class _Flaky:
    """Raise ``exc`` the first ``fail_times`` calls, then return ``value``."""

    def __init__(self, exc: Exception | None, fail_times: int, value: str = "ok") -> None:
        self.exc = exc
        self.fail_times = fail_times
        self.value = value
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.exc is not None and self.calls <= self.fail_times:
            raise self.exc
        return self.value


class _ApiErr(Exception):
    """Mimics alpaca APIError: an exception carrying a `.status_code`."""

    def __init__(self, code: int) -> None:
        self.status_code = code
        super().__init__(f"api {code}")


def test_success_no_retry() -> None:
    f = _Flaky(None, 0)
    assert with_retry(f, what="x") == "ok"
    assert f.calls == 1


def test_retry_then_succeed_on_transport() -> None:
    f = _Flaky(httpx.RemoteProtocolError("server dropped connection"), fail_times=2)
    assert with_retry(f, what="x") == "ok"
    assert f.calls == 3


@pytest.mark.parametrize("code", [500, 502, 503, 429])
def test_retry_on_5xx_and_429(code: int) -> None:
    f = _Flaky(_ApiErr(code), fail_times=1)
    assert with_retry(f, what="x") == "ok"
    assert f.calls == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_no_retry_on_other_4xx(code: int) -> None:
    f = _Flaky(_ApiErr(code), fail_times=5)
    with pytest.raises(_ApiErr):
        with_retry(f, what="x")
    assert f.calls == 1  # raised on first attempt, never retried


def test_httpx_status_error_5xx_retried() -> None:
    req = httpx.Request("GET", "https://vendor/x")
    resp = httpx.Response(503, request=req)
    f = _Flaky(httpx.HTTPStatusError("boom", request=req, response=resp), fail_times=1)
    assert with_retry(f, what="x") == "ok"
    assert f.calls == 2


def test_oserror_transport_retried() -> None:
    # requests transport errors (alpaca) are RequestException < IOError(OSError)
    f = _Flaky(OSError("connection reset by peer"), fail_times=1)
    assert with_retry(f, what="x") == "ok"


def test_non_transient_raises_immediately() -> None:
    f = _Flaky(ValueError("malformed payload"), fail_times=5)
    with pytest.raises(ValueError):
        with_retry(f, what="x")
    assert f.calls == 1


def test_exhausts_attempts_then_raises() -> None:
    f = _Flaky(httpx.ConnectError("refused"), fail_times=99)
    with pytest.raises(httpx.ConnectError):
        with_retry(f, what="x")
    assert f.calls == _ATTEMPTS  # tried exactly _ATTEMPTS times, then re-raised
