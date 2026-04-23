"""
tools/test_alpaca_opra_direct.py

Measure Alpaca OPRA stream latency directly — no MDS dependency.
MDS must be stopped before running (one connection per API key).

Usage:
    python tools/test_alpaca_opra_direct.py \
        --contracts NVDA260422P00200000 \
        --duration 120
"""

import argparse
import asyncio
import os
import statistics
import time
from datetime import datetime, timezone

import msgpack
import websockets

OPRA_URL = "wss://stream.data.alpaca.markets/v1beta1/opra"


def ts_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def parse_event_ts(raw) -> tuple[str, int]:
    """Parse RFC3339 UTC timestamp to (local time string, epoch_ms)."""
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        return local_dt.strftime("%H:%M:%S.%f")[:-3], int(dt.timestamp() * 1000)
    except Exception:
        return raw, 0


async def run(contracts: list[str], duration: int):
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET")
        return

    lags: list[int] = []
    trade_count = 0
    end_time = time.time() + duration

    print(f"Connecting to {OPRA_URL}")
    print(f"Contracts: {contracts}")
    print(f"Duration: {duration}s")

    async with websockets.connect(OPRA_URL, max_size=10_000_000) as ws:
        # Wait for welcome
        welcome_raw = await ws.recv()
        welcome = msgpack.unpackb(welcome_raw, raw=False)
        print(f"Connected: {welcome}")

        # Auth
        await ws.send(msgpack.packb({
            "action": "auth",
            "key": api_key,
            "secret": api_secret,
        }))
        auth_raw = await ws.recv()
        auth_resp = msgpack.unpackb(auth_raw, raw=False)
        print(f"Auth: {auth_resp}")

        # Subscribe
        await ws.send(msgpack.packb({
            "action": "subscribe",
            "trades": contracts,
        }))
        sub_raw = await ws.recv()
        sub_resp = msgpack.unpackb(sub_raw, raw=False)
        print(f"Subscribe: {sub_resp}")

        print(f"\n{'='*80}")
        print(f"Listening at {ts_str()}")
        print(f"{'='*80}\n")

        while time.time() < end_time:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            recv_ms = int(time.time() * 1000)
            msgs = msgpack.unpackb(raw, raw=False)
            if not isinstance(msgs, list):
                msgs = [msgs]

            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                if msg.get("T") != "t":
                    continue

                symbol = msg.get("S", "?")
                price = msg.get("p", "?")
                size = msg.get("s", "?")
                event_raw = msg.get("t", "")

                event_str, event_ms = parse_event_ts(event_raw)
                lag = recv_ms - event_ms if event_ms > 0 else 0

                print(f"{ts_str()} | event_ts={event_str} | lag={lag}ms | "
                      f"symbol={symbol} price={price} size={size}",
                      flush=True)

                trade_count += 1
                if lag > 0:
                    lags.append(lag)

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total trades received: {trade_count}")
    if lags:
        print(f"  Min lag:    {min(lags)}ms")
        print(f"  Max lag:    {max(lags)}ms")
        print(f"  Median lag: {statistics.median(lags):.0f}ms")
        print(f"  Mean lag:   {statistics.mean(lags):.0f}ms")
    else:
        print(f"  No lag data collected")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct Alpaca OPRA stream latency test",
    )
    parser.add_argument("--contracts", required=True,
                        help="Comma-separated OSI symbols")
    parser.add_argument("--duration", type=int, default=120,
                        help="Duration in seconds (default: 120)")
    args = parser.parse_args()

    contracts = [c.strip().upper() for c in args.contracts.split(",")]
    asyncio.run(run(contracts, args.duration))


if __name__ == "__main__":
    main()
