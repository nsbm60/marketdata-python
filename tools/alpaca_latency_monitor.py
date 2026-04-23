"""
tools/alpaca_latency_monitor.py

Direct Alpaca WebSocket latency monitor. Connects to OPRA (msgpack)
or SIP/IEX (JSON) and measures baseline latency per symbol/type.

MDS must be stopped before running OPRA (one connection per API key).

Usage:
    # OPRA options (msgpack)
    python tools/alpaca_latency_monitor.py \
        --feed opra \
        --symbols NVDA260425P00100000,NVDA260425C00110000 \
        --duration 300 \
        --csv data/latency/alpaca_opra.csv

    # SIP equities (JSON)
    python tools/alpaca_latency_monitor.py \
        --feed sip \
        --symbols NVDA,AMD \
        --duration 300 \
        --csv data/latency/alpaca_sip.csv
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FEED_URLS = {
    "opra": "wss://stream.data.alpaca.markets/v1beta1/opra",
    "sip":  "wss://stream.data.alpaca.markets/v2/sip",
    "iex":  "wss://stream.data.alpaca.markets/v2/iex",
}

MSG_TYPES = {"t": "trade", "q": "quote", "b": "bar"}


def now_ms():
    return int(time.time() * 1000)


def parse_event_ts(raw) -> int:
    """Parse RFC3339/ISO timestamp to epoch_ms. Returns 0 on failure."""
    if not raw:
        return 0
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


class LatencyTracker:
    def __init__(self, csv_path=None):
        self.buckets = defaultdict(list)  # (symbol, msg_type) -> [lag_ms]
        self.window_buckets = defaultdict(list)
        self.csv_writer = None
        self.csv_file = None
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "receive_ts_utc", "event_ts_utc", "lag_ms",
                "feed", "msg_type", "symbol",
            ])

    def record(self, symbol, msg_type, event_ms, recv_ms, feed):
        lag = recv_ms - event_ms if event_ms > 0 else 0
        key = (symbol, msg_type)
        self.buckets[key].append(lag)
        self.window_buckets[key].append(lag)
        if self.csv_writer and event_ms > 0:
            self.csv_writer.writerow([
                datetime.fromtimestamp(recv_ms / 1000, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(event_ms / 1000, tz=timezone.utc).isoformat(),
                lag, feed, msg_type, symbol,
            ])

    def print_window(self, label="10s"):
        if not self.window_buckets:
            return
        print(f"\n--- {label} rolling stats ---")
        print(f"{'Symbol':<25} {'Type':<6} {'Count':>6} {'Min':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'Max':>6}")
        for (sym, mtype), lags in sorted(self.window_buckets.items()):
            if not lags:
                continue
            s = sorted(lags)
            n = len(s)
            p50 = s[n // 2]
            p95 = s[int(n * 0.95)]
            p99 = s[int(n * 0.99)]
            print(f"{sym:<25} {mtype:<6} {n:>6} {min(s):>6} {p50:>6} {p95:>6} {p99:>6} {max(s):>6}")
        self.window_buckets.clear()
        print("", flush=True)

    def print_summary(self):
        if not self.buckets:
            print("\nNo data collected")
            return
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"{'Symbol':<25} {'Type':<6} {'Count':>6} {'Min':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'Max':>6} {'Mean':>6}")
        total_lags = []
        for (sym, mtype), lags in sorted(self.buckets.items()):
            if not lags:
                continue
            s = sorted(lags)
            n = len(s)
            p50 = s[n // 2]
            p95 = s[int(n * 0.95)]
            p99 = s[int(n * 0.99)]
            mean = int(statistics.mean(s))
            print(f"{sym:<25} {mtype:<6} {n:>6} {min(s):>6} {p50:>6} {p95:>6} {p99:>6} {max(s):>6} {mean:>6}")
            total_lags.extend(s)
        if total_lags:
            s = sorted(total_lags)
            n = len(s)
            print(f"\n{'AGGREGATE':<25} {'all':<6} {n:>6} {min(s):>6} {s[n//2]:>6} {s[int(n*0.95)]:>6} {s[int(n*0.99)]:>6} {max(s):>6} {int(statistics.mean(s)):>6}")
        print(f"{'='*70}")

    def close(self):
        if self.csv_file:
            self.csv_file.close()


async def run_opra(symbols, duration, tracker, feed):
    import msgpack
    import websockets

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET")
        return

    url = FEED_URLS[feed]
    print(f"Connecting to {url} (msgpack)")

    async with websockets.connect(url, max_size=10_000_000) as ws:
        welcome = msgpack.unpackb(await ws.recv(), raw=False)
        print(f"Connected: {welcome}")

        await ws.send(msgpack.packb({"action": "auth", "key": api_key, "secret": api_secret}))
        auth = msgpack.unpackb(await ws.recv(), raw=False)
        print(f"Auth: {auth}")

        await ws.send(msgpack.packb({"action": "subscribe", "trades": symbols, "quotes": symbols}))
        sub = msgpack.unpackb(await ws.recv(), raw=False)
        print(f"Subscribe: {sub}")

        print(f"\nListening for {duration}s...\n")
        end_time = time.time() + duration
        last_window = time.time()

        while time.time() < end_time:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.time() - last_window >= 10:
                    tracker.print_window()
                    last_window = time.time()
                continue

            recv = now_ms()
            msgs = msgpack.unpackb(raw, raw=False)
            if not isinstance(msgs, list):
                msgs = [msgs]

            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                tag = msg.get("T", "")
                mtype = MSG_TYPES.get(tag)
                if not mtype:
                    continue
                symbol = msg.get("S", "?")
                event_ms = parse_event_ts(msg.get("t", ""))
                tracker.record(symbol, mtype, event_ms, recv, feed)

            if time.time() - last_window >= 10:
                tracker.print_window()
                last_window = time.time()


async def run_json(symbols, duration, tracker, feed):
    import websockets

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET")
        return

    url = FEED_URLS[feed]
    print(f"Connecting to {url} (JSON)")

    async with websockets.connect(url, max_size=10_000_000) as ws:
        welcome = json.loads(await ws.recv())
        print(f"Connected: {welcome}")

        await ws.send(json.dumps({"action": "auth", "key": api_key, "secret": api_secret}))
        auth = json.loads(await ws.recv())
        print(f"Auth: {auth}")

        await ws.send(json.dumps({"action": "subscribe", "trades": symbols, "quotes": symbols}))
        sub = json.loads(await ws.recv())
        print(f"Subscribe: {sub}")

        print(f"\nListening for {duration}s...\n")
        end_time = time.time() + duration
        last_window = time.time()

        while time.time() < end_time:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                if time.time() - last_window >= 10:
                    tracker.print_window()
                    last_window = time.time()
                continue

            recv = now_ms()
            msgs = json.loads(raw)
            if not isinstance(msgs, list):
                msgs = [msgs]

            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                tag = msg.get("T", "")
                mtype = MSG_TYPES.get(tag)
                if not mtype:
                    continue
                symbol = msg.get("S", "?")
                event_ms = parse_event_ts(msg.get("t", ""))
                tracker.record(symbol, mtype, event_ms, recv, feed)

            if time.time() - last_window >= 10:
                tracker.print_window()
                last_window = time.time()


def main():
    parser = argparse.ArgumentParser(description="Direct Alpaca WebSocket latency monitor")
    parser.add_argument("--feed", required=True, choices=["opra", "sip", "iex"])
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--csv", type=str, default=None, help="CSV output path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    tracker = LatencyTracker(csv_path=args.csv)

    print(f"Feed: {args.feed}, Symbols: {symbols}, Duration: {args.duration}s")

    try:
        if args.feed == "opra":
            asyncio.run(run_opra(symbols, args.duration, tracker, args.feed))
        else:
            asyncio.run(run_json(symbols, args.duration, tracker, args.feed))
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.print_summary()
        tracker.close()


if __name__ == "__main__":
    main()
