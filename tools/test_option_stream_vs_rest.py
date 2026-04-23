"""
tools/test_option_stream_vs_rest.py

Event-driven comparison: on each streaming option trade, immediately
fire a REST snapshot call and compare results.

Usage:
    PYTHONPATH=. python tools/test_option_stream_vs_rest.py \
        --contracts MU260501P00545000,MU260501C00342500 \
        --duration 120
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import requests
import zmq

from discovery.service_locator import ServiceLocator

ALPACA_DATA_URL = "https://data.alpaca.markets/v1beta1"
MIN_REST_INTERVAL_MS = 100  # throttle: max 1 REST call per 100ms per contract


def ts_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def parse_event_ts(raw):
    """Parse ISO timestamp to (formatted string, epoch_ms)."""
    if not isinstance(raw, str) or "T" not in raw:
        return str(raw), 0
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S.%f")[:-3], int(dt.timestamp() * 1000)
    except Exception:
        return str(raw), 0


def parse_osi_underlying(osi: str) -> str:
    """Extract underlying from OSI symbol. E.g. MU260501P00545000 -> MU"""
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", osi)
    if not m:
        return ""
    return m.group(1)


def topic_to_osi(topic_suffix: str) -> str:
    """Convert topic suffix to OSI symbol.
    E.g. MU.2026-05-01.P.545_00 -> MU260501P00545000
    """
    parts = topic_suffix.split(".")
    if len(parts) != 4:
        return ""
    underlying, expiry_str, side, strike_str = parts
    # expiry: 2026-05-01 -> 260501
    expiry = expiry_str.replace("-", "")[2:]  # strip century
    # strike: 545_00 -> 00545000 (dollars_cents -> 8 digit, strike * 1000)
    dollar_cents = strike_str.split("_")
    if len(dollar_cents) == 2:
        dollars = int(dollar_cents[0])
        cents = int(dollar_cents[1])
        strike_int = dollars * 1000 + cents * 10
        strike_osi = f"{strike_int:08d}"
    else:
        return ""
    return f"{underlying}{expiry}{side}{strike_osi}"


def rest_snapshot(osi, headers):
    """Fetch single-contract option snapshot from Alpaca REST."""
    resp = requests.get(
        f"{ALPACA_DATA_URL}/options/snapshots",
        headers=headers,
        params={"symbols": osi, "feed": "opra"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    snapshots = resp.json().get("snapshots", {})
    snap = snapshots.get(osi, {})
    trade = snap.get("latestTrade", {})
    if not trade:
        return None, None, None
    return trade.get("p"), trade.get("s"), trade.get("t", "")


def main():
    parser = argparse.ArgumentParser(
        description="Event-driven option stream vs REST comparison",
    )
    parser.add_argument("--contracts", required=True,
                        help="Comma-separated OSI symbols")
    parser.add_argument("--duration", type=int, default=300,
                        help="Duration in seconds (default: 300)")
    args = parser.parse_args()

    contracts = [c.strip().upper() for c in args.contracts.split(",")]

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_API_SECRET")
        return

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    # Find MDS pub/sub endpoint
    md = ServiceLocator.wait_for_service(ServiceLocator.MARKET_DATA, timeout_sec=10)
    print(f"MDS pub/sub: {md.pub_sub}")
    print(f"Contracts: {contracts}")
    print(f"Duration: {args.duration}s")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(md.pub_sub)
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    for osi in contracts:
        underlying = parse_osi_underlying(osi)
        prefix = f"md.option.trade.{underlying}."
        sock.setsockopt_string(zmq.SUBSCRIBE, prefix)
        print(f"Subscribed to {prefix}*")

    contracts_set = set(contracts)

    print(f"\n{'='*70}")
    print(f"Listening at {ts_str()}")
    print(f"{'='*70}\n")

    stats = {"stream": 0, "rest_ok": 0, "rest_err": 0, "match": 0, "mismatch": 0}
    last_rest_call = {}  # per-contract throttle
    seen_event_ts = {}  # per-contract: set of event_ts_ms seen from streaming

    start_epoch = time.time()
    end_time = start_epoch + args.duration
    while time.time() < end_time:
        try:
            frames = sock.recv_multipart()
        except zmq.Again:
            continue

        if len(frames) < 2:
            continue

        recv_time = time.time()
        topic = frames[0].decode("utf-8")
        payload = json.loads(frames[1].decode("utf-8"))
        data = payload.get("data", payload)

        topic_suffix = topic.replace("md.option.trade.", "")
        osi = topic_to_osi(topic_suffix)
        if osi not in contracts_set:
            continue
        stream_price = data.get("price", data.get("p", "?"))
        stream_size = data.get("size", data.get("s", "?"))
        stream_event_raw = data.get("timestamp", data.get("t", "?"))
        stream_event_str, stream_event_ms = parse_event_ts(stream_event_raw)

        print(f"[STREAM] {ts_str()} {osi} price={stream_price} size={stream_size} event_ts={stream_event_str}",
              flush=True)
        stats["stream"] += 1
        if stream_event_ms > 0:
            seen_event_ts.setdefault(osi, set()).add(stream_event_ms)

        # Throttle: skip REST if called too recently for this contract
        now_ms = int(recv_time * 1000)
        last_ms = last_rest_call.get(osi, 0)
        if now_ms - last_ms < MIN_REST_INTERVAL_MS:
            continue
        last_rest_call[osi] = now_ms

        # Fire REST call
        try:
            rest_price, rest_size, rest_event_raw = rest_snapshot(osi, headers)
            rest_recv_time = time.time()
            latency_ms = int((rest_recv_time - recv_time) * 1000)

            if rest_price is None:
                print(f"[REST]   {ts_str()} {osi} no trade in snapshot (+{latency_ms}ms)", flush=True)
                stats["rest_ok"] += 1
                continue

            rest_event_str, rest_event_ms = parse_event_ts(rest_event_raw)
            print(f"[REST]   {ts_str()} {osi} price={rest_price} size={rest_size} event_ts={rest_event_str}  (+{latency_ms}ms)",
                  flush=True)
            stats["rest_ok"] += 1

            # Compare
            if stream_event_ms > 0 and rest_event_ms > 0:
                if stream_event_ms == rest_event_ms:
                    print(f"[MATCH]", flush=True)
                    stats["match"] += 1
                else:
                    gap_ms = rest_event_ms - stream_event_ms
                    direction = "newer" if gap_ms > 0 else "older"
                    print(f"[MISMATCH] REST reports {direction} trade ({gap_ms:+d}ms event gap)", flush=True)
                    stats["mismatch"] += 1

        except Exception as e:
            print(f"[REST ERROR] {osi}: {e}", flush=True)
            stats["rest_err"] += 1

        print("", flush=True)  # blank line between pairs

    sock.close()
    ctx.term()

    end_epoch = time.time()
    start_ms = int(start_epoch * 1000)
    end_ms = int(end_epoch * 1000)

    # Final REST snapshot for each contract
    print(f"\n{'='*70}")
    print(f"Final REST snapshot:")
    print(f"{'='*70}")
    for osi in contracts:
        try:
            rest_price, rest_size, rest_event_raw = rest_snapshot(osi, headers)
            if rest_price is None:
                print(f"  {osi}: no trade in snapshot")
                continue

            rest_event_str, rest_event_ms = parse_event_ts(rest_event_raw)
            in_window = start_ms <= rest_event_ms <= end_ms if rest_event_ms > 0 else False
            was_seen = rest_event_ms in seen_event_ts.get(osi, set()) if rest_event_ms > 0 else False

            window_label = "IN WINDOW" if in_window else "OUTSIDE WINDOW"
            print(f"  {osi}: price={rest_price} size={rest_size} event_ts={rest_event_str} [{window_label}]")

            if in_window and not was_seen:
                print(f"  >>> Streaming missed this trade!")
            elif in_window and was_seen:
                print(f"  >>> Streaming saw this trade")

        except Exception as e:
            print(f"  {osi}: REST ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Stream trades received:  {stats['stream']}")
    print(f"  REST calls successful:   {stats['rest_ok']}")
    print(f"  REST call errors:        {stats['rest_err']}")
    print(f"  Event timestamp matches: {stats['match']}")
    print(f"  Event timestamp mismatches: {stats['mismatch']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
