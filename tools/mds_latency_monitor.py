"""
tools/mds_latency_monitor.py

MDS ZMQ latency monitor. Subscribes to MDS pub/sub and measures
end-to-end latency from Alpaca event timestamp to ZMQ delivery.

MDS must be running.

Usage:
    # Option trades
    PYTHONPATH=. python tools/mds_latency_monitor.py \
        --topics md.option.trade. \
        --duration 300 \
        --csv data/latency/mds_option_trade.csv

    # Equity trades + quotes
    PYTHONPATH=. python tools/mds_latency_monitor.py \
        --topics md.equity.trade.,md.equity.quote. \
        --duration 300

    # Everything
    PYTHONPATH=. python tools/mds_latency_monitor.py \
        --topics md. \
        --duration 300

    # Filter by symbol
    PYTHONPATH=. python tools/mds_latency_monitor.py \
        --topics md.option.trade. \
        --symbols NVDA,AMD \
        --duration 120
"""

import argparse
import csv
import json
import signal
import statistics
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import zmq


def now_ms():
    return int(time.time() * 1000)


def parse_event_ts(data) -> int:
    """Extract event timestamp from MDS JSON payload. Returns epoch_ms or 0."""
    if not isinstance(data, dict):
        return 0
    # Try nested data.timestamp, then data.data.timestamp, then top-level timestamp
    for path in [
        lambda d: d.get("data", {}).get("timestamp") if isinstance(d.get("data"), dict) else None,
        lambda d: d.get("timestamp"),
        lambda d: d.get("t"),
    ]:
        raw = path(data)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            # Already epoch ms or seconds
            val = int(raw)
            if val < 1e12:  # seconds
                return val * 1000
            return val
        if isinstance(raw, str) and "T" in raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                continue
    return 0


def extract_symbol(topic: str) -> str:
    """Extract symbol from topic. E.g. md.equity.trade.NVDA -> NVDA,
    md.option.trade.NVDA.2026-04-25.P.100_00 -> NVDA."""
    parts = topic.split(".")
    if len(parts) >= 4:
        return parts[3]  # underlying for both equity and option topics
    return topic


def extract_msg_type(topic: str) -> str:
    """Extract message type from topic. E.g. md.equity.trade.NVDA -> trade."""
    parts = topic.split(".")
    if len(parts) >= 3:
        return parts[2]  # trade, quote, greeks, bar, etc.
    return "unknown"


class LatencyTracker:
    def __init__(self, csv_path=None):
        self.lock = threading.Lock()
        self.buckets = defaultdict(list)
        self.window_buckets = defaultdict(list)
        self.csv_writer = None
        self.csv_file = None
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "receive_ts_utc", "event_ts_utc", "lag_ms",
                "topic", "msg_type", "symbol",
            ])

    def record(self, topic, symbol, msg_type, event_ms, recv_ms):
        lag = recv_ms - event_ms if event_ms > 0 else 0
        if lag < 0 or lag > 300_000:  # ignore nonsense (clock skew > 5 min)
            return
        key = (symbol, msg_type)
        with self.lock:
            self.buckets[key].append(lag)
            self.window_buckets[key].append(lag)
        if self.csv_writer and event_ms > 0:
            self.csv_writer.writerow([
                datetime.fromtimestamp(recv_ms / 1000, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(event_ms / 1000, tz=timezone.utc).isoformat(),
                lag, topic, msg_type, symbol,
            ])

    def print_window(self, label="10s"):
        with self.lock:
            window = dict(self.window_buckets)
            self.window_buckets.clear()
        if not window:
            return
        print(f"\n--- {label} rolling stats ---")
        print(f"{'Symbol':<12} {'Type':<8} {'Count':>6} {'Min':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'Max':>6}")
        total_count = 0
        for (sym, mtype), lags in sorted(window.items()):
            if not lags:
                continue
            s = sorted(lags)
            n = len(s)
            total_count += n
            p50 = s[n // 2]
            p95 = s[int(n * 0.95)]
            p99 = s[int(n * 0.99)]
            print(f"{sym:<12} {mtype:<8} {n:>6} {min(s):>6} {p50:>6} {p95:>6} {p99:>6} {max(s):>6}")
        print(f"  ({total_count} messages in window)", flush=True)

    def print_summary(self):
        with self.lock:
            all_buckets = dict(self.buckets)
        if not all_buckets:
            print("\nNo data collected")
            return
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Symbol':<12} {'Type':<8} {'Count':>6} {'Min':>6} {'p50':>6} {'p95':>6} {'p99':>6} {'Max':>6} {'Mean':>6}")
        total_lags = []
        for (sym, mtype), lags in sorted(all_buckets.items()):
            if not lags:
                continue
            s = sorted(lags)
            n = len(s)
            p50 = s[n // 2]
            p95 = s[int(n * 0.95)]
            p99 = s[int(n * 0.99)]
            mean = int(statistics.mean(s))
            print(f"{sym:<12} {mtype:<8} {n:>6} {min(s):>6} {p50:>6} {p95:>6} {p99:>6} {max(s):>6} {mean:>6}")
            total_lags.extend(s)
        if total_lags:
            s = sorted(total_lags)
            n = len(s)
            print(f"\n{'AGGREGATE':<12} {'all':<8} {n:>6} {min(s):>6} {s[n//2]:>6} {s[int(n*0.95)]:>6} {s[int(n*0.99)]:>6} {max(s):>6} {int(statistics.mean(s)):>6}")
        print(f"{'='*80}")

    def close(self):
        if self.csv_file:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="MDS ZMQ latency monitor")
    parser.add_argument("--endpoint", type=str, default="tcp://192.168.37.191:6006",
                        help="ZMQ PUB endpoint (default: tcp://192.168.37.191:6006)")
    parser.add_argument("--topics", required=True,
                        help="Comma-separated topic prefixes to subscribe (e.g. md.option.trade.,md.equity.trade.)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbol filter (applied after receive)")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    topic_prefixes = [t.strip() for t in args.topics.split(",")]
    symbol_filter = set(s.strip().upper() for s in args.symbols.split(",")) if args.symbols else None

    pub_endpoint = args.endpoint

    print(f"MDS endpoint: {pub_endpoint}")
    print(f"Topics: {topic_prefixes}")
    print(f"Symbol filter: {symbol_filter or 'all'}")
    print(f"Duration: {args.duration}s")

    tracker = LatencyTracker(csv_path=args.csv)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(pub_endpoint)
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    for prefix in topic_prefixes:
        sock.setsockopt_string(zmq.SUBSCRIBE, prefix)
        print(f"Subscribed to {prefix}*")

    print(f"\nListening...\n")

    running = True

    def on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    end_time = time.time() + args.duration
    last_window = time.time()

    while running and time.time() < end_time:
        try:
            frames = sock.recv_multipart()
        except zmq.Again:
            if time.time() - last_window >= 10:
                tracker.print_window()
                last_window = time.time()
            continue

        recv = now_ms()

        if len(frames) < 2:
            continue

        topic = frames[0].decode("utf-8")
        try:
            payload = json.loads(frames[1].decode("utf-8"))
        except Exception:
            continue

        symbol = extract_symbol(topic)
        msg_type = extract_msg_type(topic)

        if symbol_filter and symbol.upper() not in symbol_filter:
            continue

        event_ms = parse_event_ts(payload)
        tracker.record(topic, symbol, msg_type, event_ms, recv)

        if time.time() - last_window >= 10:
            tracker.print_window()
            last_window = time.time()

    sock.close()
    ctx.term()
    tracker.print_summary()
    tracker.close()


if __name__ == "__main__":
    main()
