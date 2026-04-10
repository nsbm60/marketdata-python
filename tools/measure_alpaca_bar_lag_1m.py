#!/usr/bin/env python3
"""
measure_alpaca_bar_lag.py

Measures the lag between a 1m bar's close time (bar_open + 60s) and when
Alpaca publishes it on their WebSocket feed.

Usage:
    python measure_alpaca_bar_lag.py [--symbol NVDA] [--feed sip]

Requires:
    pip install websocket-client numpy
    Environment variables: ALPACA_API_KEY, ALPACA_API_SECRET

Output:
    Running table of each bar received, plus summary statistics at Ctrl-C.
"""

import argparse
import json
import os
import signal
import sys
import threading
from datetime import datetime, timezone, timedelta

import numpy as np
import websocket

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Measure Alpaca 1m bar publication lag")
parser.add_argument("--symbol", default="NVDA")
parser.add_argument("--feed",   default="sip", choices=["sip", "iex"])
args = parser.parse_args()

API_KEY    = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    print("ERROR: set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
    sys.exit(1)

WS_URL = f"wss://stream.data.alpaca.markets/v2/{args.feed}"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
lags_ms: list[float] = []
out_of_order: list[dict] = []
last_bar_open: datetime | None = None
authenticated = threading.Event()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bar_close_time(bar_open_utc: datetime) -> datetime:
    return bar_open_utc + timedelta(seconds=60)

def print_summary():
    if not lags_ms:
        print("\nNo bars received.")
        return

    arr = np.array(lags_ms)
    print(f"\n{'='*60}")
    print(f"Summary for {args.symbol} — {len(arr)} bars")
    print(f"{'='*60}")
    print(f"  min:    {arr.min():>8.1f} ms")
    print(f"  p50:    {np.percentile(arr, 50):>8.1f} ms")
    print(f"  p90:    {np.percentile(arr, 90):>8.1f} ms")
    print(f"  p95:    {np.percentile(arr, 95):>8.1f} ms")
    print(f"  p99:    {np.percentile(arr, 99):>8.1f} ms")
    print(f"  max:    {arr.max():>8.1f} ms")
    print(f"  mean:   {arr.mean():>8.1f} ms")
    print(f"  stddev: {arr.std():>8.1f} ms")
    print(f"\n  out-of-order bars: {len(out_of_order)}")
    if out_of_order:
        for oo in out_of_order:
            print(f"    {oo}")
    print(f"\n  negative lags (arrived before bar close): {(arr < 0).sum()} bars")
    print(f"  lags > 1s:  {(arr > 1000).sum()} bars")
    print(f"  lags > 5s:  {(arr > 5000).sum()} bars")
    print(f"  lags > 10s: {(arr > 10000).sum()} bars")

def handle_sigint(sig, frame):
    print_summary()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# ---------------------------------------------------------------------------
# WebSocket handlers
# ---------------------------------------------------------------------------
def on_open(ws):
    # Authenticate
    auth = json.dumps({"action": "auth", "key": API_KEY, "secret": API_SECRET})
    ws.send(auth)

def on_message(ws, message):
    global last_bar_open

    received_at = datetime.now(timezone.utc)
    msgs = json.loads(message)

    for msg in msgs:
        t = msg.get("T")

        if t == "success" and msg.get("msg") == "connected":
            print(f"Connected to Alpaca {args.feed.upper()} feed")
            return

        if t == "success" and msg.get("msg") == "authenticated":
            print(f"Authenticated. Subscribing to bars.{args.symbol}...\n")
            sub = json.dumps({"action": "subscribe", "bars": [args.symbol]})
            ws.send(sub)
            print(f"{'bar_open':<22} {'bar_close':<22} {'received_at':<22} {'lag_ms':>8}  notes")
            print("-" * 90)
            authenticated.set()
            return

        if t == "subscription":
            return

        if t == "error":
            print(f"ERROR from Alpaca: {msg}")
            return

        if t == "b":  # bar message
            symbol = msg.get("S", "")
            if symbol != args.symbol:
                return

            # Alpaca bar timestamp is the bar OPEN in RFC3339
            ts_str = msg.get("t", "")
            try:
                bar_open = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                bar_open = bar_open.astimezone(timezone.utc)
            except Exception as e:
                print(f"  [could not parse timestamp '{ts_str}': {e}]")
                return

            bar_close = bar_close_time(bar_open)
            lag_ms = (received_at - bar_close).total_seconds() * 1000.0

            notes = []
            if last_bar_open is not None:
                expected_next = last_bar_open + timedelta(seconds=60)
                if bar_open < last_bar_open:
                    notes.append("OUT-OF-ORDER")
                    out_of_order.append({
                        "bar_open":      bar_open.isoformat(),
                        "prev_bar_open": last_bar_open.isoformat(),
                        "lag_ms":        lag_ms,
                    })
                elif bar_open > expected_next + timedelta(seconds=5):
                    gap_mins = (bar_open - expected_next).total_seconds() / 60
                    notes.append(f"GAP {gap_mins:.1f}m")

            if lag_ms < 0:
                notes.append(f"EARLY {abs(lag_ms):.0f}ms")

            lags_ms.append(lag_ms)
            last_bar_open = bar_open

            fmt = "%H:%M:%S.%f"
            bo  = bar_open.strftime(fmt)[:-3]
            bc  = bar_close.strftime(fmt)[:-3]
            ra  = received_at.strftime(fmt)[:-3]
            note_str = "  *** " + ", ".join(notes) if notes else ""

            print(f"{bo:<22} {bc:<22} {ra:<22} {lag_ms:>8.1f}ms{note_str}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed: {close_status_code} {close_msg}")
    print_summary()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"Connecting to {WS_URL}")
print(f"Symbol: {args.symbol}  Feed: {args.feed.upper()}")
print(f"Measuring lag = received_wall_clock - (bar_open + 60s)\n")

ws = websocket.WebSocketApp(
    WS_URL,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever()
