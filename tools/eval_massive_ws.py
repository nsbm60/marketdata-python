#!/usr/bin/env python3
"""
WebSocket test client for the Massive options real-time feed.
Connects, authenticates, subscribes, and prints raw JSON messages to stdout.
Diagnostic info is written to stderr so stdout can be cleanly piped to a file.

Usage:
    export MASSIVE_API_KEY=your_key_here

    # Subscribe to all options activity for one underlying (T+Q channels)
    python tools/eval_massive_ws.py NVDA

    # Specific contracts only
    python tools/eval_massive_ws.py --contracts O:NVDA260515C00215000,O:NVDA260515P00215000

    # Choose channels — T=trades, Q=quotes, A=second aggs, AM=minute aggs
    python tools/eval_massive_ws.py NVDA --channels T,Q,A

    # Time-bounded run, save to file
    python tools/eval_massive_ws.py NVDA --duration 30 > messages.jsonl

    # Override websocket URL if Massive's endpoint differs from Polygon's
    export MASSIVE_WS_URL=wss://socket.example.com/options
    python tools/eval_massive_ws.py NVDA
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

try:
    import websockets
except ImportError:
    print("websockets package not installed. Run: pip install websockets",
          file=sys.stderr)
    sys.exit(1)


# Default endpoint — Polygon's URL, which Massive likely inherited.
# Override via MASSIVE_WS_URL env var or --url flag if needed.
DEFAULT_WS_URL = "wss://socket.polygon.io/options"


def build_subscriptions(underlying, contracts, channels):
    """Build subscription strings of the form CHANNEL.SYMBOL."""
    subs = []
    if contracts:
        for c in contracts:
            for ch in channels:
                subs.append(f"{ch}.{c}")
    elif underlying:
        # Wildcard pattern — assumed supported by Massive (inherited from Polygon).
        # If the server rejects this, fall back to explicit --contracts.
        for ch in channels:
            subs.append(f"{ch}.O:{underlying}*")
    return subs


def log(msg):
    """Write a diagnostic line to stderr."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"# {ts}  {msg}", file=sys.stderr, flush=True)


async def run(url, api_key, subscriptions, duration, max_messages):
    msg_counts = Counter()
    total = 0
    start = datetime.now(timezone.utc)

    log(f"connecting to {url}")
    async with websockets.connect(url) as ws:
        # Auth
        await ws.send(json.dumps({"action": "auth", "params": api_key}))

        # Subscribe
        sub_param = ",".join(subscriptions)
        log(f"subscribing to: {sub_param}")
        await ws.send(json.dumps({"action": "subscribe", "params": sub_param}))

        # Receive loop. Polygon-style servers batch messages into JSON arrays.
        try:
            while True:
                if duration is not None:
                    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                    remaining = duration - elapsed
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                else:
                    raw = await ws.recv()

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    log(f"non-JSON frame: {raw!r}")
                    continue

                messages = payload if isinstance(payload, list) else [payload]
                for m in messages:
                    ev = m.get("ev") or m.get("status") or "?"
                    msg_counts[ev] += 1
                    total += 1
                    print(json.dumps(m, indent=2), flush=True)
                    if max_messages is not None and total >= max_messages:
                        return
        except websockets.ConnectionClosed as e:
            log(f"connection closed: {e}")
        finally:
            log("--- summary ---")
            log(f"total messages: {total}")
            for ev, n in sorted(msg_counts.items(), key=lambda x: -x[1]):
                log(f"  {ev}: {n}")


def main():
    parser = argparse.ArgumentParser(
        description="Massive options websocket test client",
    )
    parser.add_argument(
        "underlying",
        nargs="?",
        help="Underlying ticker (e.g. NVDA). Subscribes via wildcard.",
    )
    parser.add_argument(
        "--contracts",
        help="Comma-separated OSI symbols (e.g. O:NVDA260515C00215000)",
    )
    parser.add_argument(
        "--channels",
        default="T,Q",
        help="Comma-separated channel codes. T=trades, Q=quotes, "
             "A=second aggs, AM=minute aggs. Default: T,Q",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Stop after N seconds (otherwise runs until Ctrl-C)",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        help="Stop after N messages",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("MASSIVE_WS_URL", DEFAULT_WS_URL),
        help="Websocket endpoint URL (default: Polygon's options endpoint)",
    )
    args = parser.parse_args()

    if not args.underlying and not args.contracts:
        parser.error("Provide either an underlying ticker or --contracts")

    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        print("MASSIVE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    contracts = (
        [c.strip() for c in args.contracts.split(",") if c.strip()]
        if args.contracts else []
    )
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    subscriptions = build_subscriptions(
        args.underlying.upper() if args.underlying else None,
        contracts,
        channels,
    )
    if not subscriptions:
        print("No subscriptions to make", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run(args.url, api_key, subscriptions,
                        args.duration, args.max_messages))
    except KeyboardInterrupt:
        log("interrupted")


if __name__ == "__main__":
    main()
