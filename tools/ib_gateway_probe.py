"""
IB Gateway sanity probe.

Exercises the gateway directly via the official ibapi package to isolate
whether the gateway itself is healthy, independent of IBServer.

Tests (in order):
  1. Connect — socket handshake + API init
  2. Account summary — NetLiquidation, BuyingPower, etc. for the account
  3. Positions — all positions for the account
  4. Market data — live quote for SPY

Each step reports pass/fail with timing. A failure at step N does not prevent
later steps from running (we want to know where the gateway struggles).

Usage:
    pip install ibapi
    python tools/ib_gateway_probe.py
    python tools/ib_gateway_probe.py --continuous
    python tools/ib_gateway_probe.py --host 127.0.0.1 --port 4001 --client-id 4

Interpretation:
  - All steps pass here while IBServer struggles → IBServer is the problem.
  - Steps fail here too → gateway (or gateway's upstream) is the problem.
  - Continuous mode: leave running during an IBServer incident; if the probe
    reports sustained success while IBServer misbehaves, gateway is cleared.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
except ImportError:
    print("ibapi not installed. Run: pip install ibapi", file=sys.stderr)
    sys.exit(2)


@dataclass
class StepResult:
    name: str
    passed: bool
    elapsed_ms: int
    detail: str = ""


@dataclass
class ProbeState:
    """Shared state between main thread and ibapi's callback thread."""
    connected: threading.Event = field(default_factory=threading.Event)
    next_valid_id_received: threading.Event = field(default_factory=threading.Event)

    # Account summary
    account_summary_done: threading.Event = field(default_factory=threading.Event)
    account_summary_rows: list = field(default_factory=list)

    # Positions
    positions_done: threading.Event = field(default_factory=threading.Event)
    positions: list = field(default_factory=list)

    # Market data
    market_data_received: threading.Event = field(default_factory=threading.Event)
    bid: Optional[float] = None
    ask: Optional[float] = None

    # Errors
    last_error: Optional[str] = None
    connection_failed: threading.Event = field(default_factory=threading.Event)


class ProbeClient(EWrapper, EClient):
    """Minimal IB API client for gateway diagnostics."""

    def __init__(self, state: ProbeState):
        EClient.__init__(self, self)
        self.state = state

    # ---- Connection callbacks ----

    def nextValidId(self, orderId: int) -> None:
        # Signal that the API handshake is complete and the gateway is ready.
        self.state.next_valid_id_received.set()
        self.state.connected.set()

    def connectionClosed(self) -> None:
        self.state.connection_failed.set()

    def error(self, reqId: int, errorCode: int, errorString: str,
              advancedOrderRejectJson: str = "") -> None:
        # 2104/2106/2158 = market data farm connection OK
        # 2107 = historical data farm connection inactive but OK
        # 2119/2137 = misc informational
        # 2100-2199 range is mostly informational connectivity notices
        if errorCode in (2104, 2106, 2107, 2158, 2119, 2137):
            return
        self.state.last_error = f"[{errorCode}] {errorString}"
        # 502/504/1100 are connection-level failures
        if errorCode in (502, 504, 1100):
            self.state.connection_failed.set()

    # ---- Account summary callbacks ----

    def accountSummary(self, reqId: int, account: str, tag: str,
                       value: str, currency: str) -> None:
        self.state.account_summary_rows.append(
            {"account": account, "tag": tag, "value": value, "currency": currency}
        )

    def accountSummaryEnd(self, reqId: int) -> None:
        self.state.account_summary_done.set()

    # ---- Position callbacks ----

    def position(self, account: str, contract: Contract, position: float,
                 avgCost: float) -> None:
        self.state.positions.append(
            {"account": account, "contract": contract, "position": position,
             "avgCost": avgCost}
        )

    def positionEnd(self) -> None:
        self.state.positions_done.set()

    # ---- Market data callbacks ----

    def tickPrice(self, reqId: int, tickType: int, price: float,
                  attrib) -> None:
        # Tick types: 1=bid, 2=ask, 4=last
        if tickType == 1 and price > 0:
            self.state.bid = price
        elif tickType == 2 and price > 0:
            self.state.ask = price
        if self.state.bid and self.state.ask:
            self.state.market_data_received.set()


# ---- Step runners ----

def step_connect(client: ProbeClient, state: ProbeState,
                 host: str, port: int, client_id: int) -> StepResult:
    start = time.monotonic()
    try:
        client.connect(host, port, client_id)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="connect", passed=False, elapsed_ms=elapsed,
                          detail=f"connect exception: {e}")

    # Run the socket reader in a background thread
    thread = threading.Thread(target=client.run, daemon=True, name="ibapi-reader")
    thread.start()

    # Wait for handshake (nextValidId) or failure
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if state.next_valid_id_received.is_set():
            elapsed = int((time.monotonic() - start) * 1000)
            version = client.serverVersion() if client.isConnected() else "?"
            return StepResult(name="connect", passed=True, elapsed_ms=elapsed,
                              detail=f"server version {version}")
        if state.connection_failed.is_set():
            elapsed = int((time.monotonic() - start) * 1000)
            return StepResult(name="connect", passed=False, elapsed_ms=elapsed,
                              detail=state.last_error or "connection failed")
        time.sleep(0.05)

    elapsed = int((time.monotonic() - start) * 1000)
    return StepResult(name="connect", passed=False, elapsed_ms=elapsed,
                      detail="timeout waiting for handshake (no nextValidId in 10s)")


def step_account_summary(client: ProbeClient, state: ProbeState,
                         account: str) -> StepResult:
    start = time.monotonic()
    req_id = 9001
    tags = "NetLiquidation,BuyingPower,TotalCashValue,GrossPositionValue"

    try:
        client.reqAccountSummary(req_id, "All", tags)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="account_summary", passed=False,
                          elapsed_ms=elapsed, detail=f"request exception: {e}")

    if not state.account_summary_done.wait(timeout=10):
        try:
            client.cancelAccountSummary(req_id)
        except Exception:
            pass
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="account_summary", passed=False,
                          elapsed_ms=elapsed, detail="timeout waiting for summary (10s)")

    try:
        client.cancelAccountSummary(req_id)
    except Exception:
        pass

    elapsed = int((time.monotonic() - start) * 1000)
    rows_for_account = [r for r in state.account_summary_rows if r["account"] == account]

    if not rows_for_account:
        total = len(state.account_summary_rows)
        other_accounts = sorted({r["account"] for r in state.account_summary_rows})
        return StepResult(
            name="account_summary", passed=False, elapsed_ms=elapsed,
            detail=f"no rows for {account} (got {total} rows for accounts: {other_accounts})",
        )

    by_tag = {r["tag"]: (r["value"], r["currency"]) for r in rows_for_account}
    net_liq, currency = by_tag.get("NetLiquidation", ("?", ""))
    bp, _ = by_tag.get("BuyingPower", ("?", ""))

    return StepResult(
        name="account_summary", passed=True, elapsed_ms=elapsed,
        detail=f"{len(rows_for_account)} rows, NetLiq={net_liq} {currency}, BP={bp}",
    )


def step_positions(client: ProbeClient, state: ProbeState,
                   account: str) -> StepResult:
    start = time.monotonic()
    try:
        client.reqPositions()
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="positions", passed=False, elapsed_ms=elapsed,
                          detail=f"request exception: {e}")

    if not state.positions_done.wait(timeout=15):
        try:
            client.cancelPositions()
        except Exception:
            pass
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="positions", passed=False, elapsed_ms=elapsed,
                          detail="timeout waiting for positions (15s)")

    try:
        client.cancelPositions()
    except Exception:
        pass

    elapsed = int((time.monotonic() - start) * 1000)
    account_positions = [p for p in state.positions if p["account"] == account]
    stocks = sum(1 for p in account_positions if p["contract"].secType == "STK")
    options = sum(1 for p in account_positions if p["contract"].secType == "OPT")
    other = len(account_positions) - stocks - options

    return StepResult(
        name="positions", passed=True, elapsed_ms=elapsed,
        detail=f"{len(account_positions)} positions ({stocks} stk, {options} opt, {other} other)",
    )


def step_market_data(client: ProbeClient, state: ProbeState,
                     symbol: str = "SPY") -> StepResult:
    start = time.monotonic()
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    req_id = 9002
    try:
        client.reqMktData(req_id, contract, "", False, False, [])
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(name="market_data", passed=False, elapsed_ms=elapsed,
                          detail=f"request exception: {e}")

    received = state.market_data_received.wait(timeout=5)

    try:
        client.cancelMktData(req_id)
    except Exception:
        pass

    elapsed = int((time.monotonic() - start) * 1000)

    if received and state.bid and state.ask:
        spread = state.ask - state.bid
        return StepResult(
            name="market_data", passed=True, elapsed_ms=elapsed,
            detail=f"{symbol} bid={state.bid} ask={state.ask} spread={spread:.4f}",
        )

    # Partial data is useful diagnostic info
    detail = f"{symbol} no bid/ask within 5s"
    if state.bid:
        detail += f" (bid={state.bid} received, ask missing)"
    elif state.ask:
        detail += f" (ask={state.ask} received, bid missing)"
    else:
        detail += " (market closed, subscription issue, or gateway not relaying)"
    return StepResult(name="market_data", passed=False, elapsed_ms=elapsed, detail=detail)


def run_probe(host: str, port: int, client_id: int, account: str) -> list[StepResult]:
    """Run all probe steps. Returns list of step results in order."""
    state = ProbeState()
    client = ProbeClient(state)
    results: list[StepResult] = []

    connect_result = step_connect(client, state, host, port, client_id)
    results.append(connect_result)

    if not connect_result.passed:
        try:
            client.disconnect()
        except Exception:
            pass
        return results

    try:
        results.append(step_account_summary(client, state, account))
        results.append(step_positions(client, state, account))
        results.append(step_market_data(client, state))
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    return results


def format_result(r: StepResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    marker = "✓" if r.passed else "✗"
    return f"  {marker} {r.name:20s} {status}  {r.elapsed_ms:>5d}ms  {r.detail}"


def print_run(results: list[StepResult], timestamp: Optional[datetime] = None) -> bool:
    ts = timestamp or datetime.now()
    print(f"\n[{ts.strftime('%Y-%m-%d %H:%M:%S')}] IB Gateway probe:")
    for r in results:
        print(format_result(r))
    all_pass = all(r.passed for r in results)
    print(f"  overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(description="IB Gateway sanity probe")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=4001, help="Gateway port (default: 4001)")
    parser.add_argument(
        "--client-id",
        type=int,
        default=4,
        help="Client ID (default: 4; must not conflict with IBServer's client IDs)",
    )
    parser.add_argument("--account", help="IB account ID (prompts if not provided)")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Poll every 30s continuously until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between polls in continuous mode (default: 30)",
    )
    args = parser.parse_args()

    account = args.account or input("IB account ID (e.g. U1234567): ").strip()
    if not account:
        print("account ID required", file=sys.stderr)
        return 2

    print(f"Gateway: {args.host}:{args.port}  client={args.client_id}  account={account}")

    if not args.continuous:
        results = run_probe(args.host, args.port, args.client_id, account)
        all_pass = print_run(results)
        return 0 if all_pass else 1

    # Continuous mode
    print(f"Continuous mode — polling every {args.interval}s. Ctrl+C to stop.\n")
    last_status: Optional[bool] = None
    try:
        while True:
            results = run_probe(args.host, args.port, args.client_id, account)
            all_pass = print_run(results)
            if last_status is not None and all_pass != last_status:
                transition = "recovered" if all_pass else "degraded"
                print(f"  *** state change: {transition} ***")
            last_status = all_pass
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
