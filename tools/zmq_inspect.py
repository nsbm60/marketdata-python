#!/usr/bin/env python3
"""
ZMQ topic inspector — connect to a PUB socket and observe messages.

Modes:
  (default)   Print new topics as they arrive (deduplicated).
  --detail    Print topic + pretty-printed JSON for every message.
  --snapshot  Print the first matching message and exit.

Examples:
  # List all topics on CalcServer (port 6020)
  python zmq_inspect.py

  # List topics on the market data service
  python zmq_inspect.py --port 6006

  # Show all report.positions messages with full content
  python zmq_inspect.py --topic report.positions --detail

  # Grab one positions report snapshot and exit
  python zmq_inspect.py --topic report.positions --snapshot
"""

import argparse
import json
import signal
import sys

import zmq


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ZMQ PUB/SUB topics and messages."
    )
    parser.add_argument(
        "--port", "-p", type=int, default=6020,
        help="ZMQ PUB port to connect to (default: 6020 = CalcServer)",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Host to connect to (default: localhost)",
    )
    parser.add_argument(
        "--topic", "-t", action="append", default=None,
        help="Topic prefix filter (ZMQ-level). Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--detail", "-d", action="store_true",
        help="Print full message content (pretty-printed JSON).",
    )
    parser.add_argument(
        "--snapshot", "-s", action="store_true",
        help="Print the first matching message and exit.",
    )
    args = parser.parse_args()

    addr = f"tcp://{args.host}:{args.port}"
    prefixes = args.topic or [""]

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(addr)
    for p in prefixes:
        sub.setsockopt_string(zmq.SUBSCRIBE, p)

    seen_topics: set[str] = set()
    msg_count = 0

    def on_exit(sig=None, frame=None):
        print(f"\n--- {len(seen_topics)} unique topics, {msg_count} messages ---",
              file=sys.stderr)
        sub.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    label = ", ".join(f'"{p}"' for p in prefixes) if prefixes != [""] else "(all)"
    print(f"Listening on {addr}  topic={label}", file=sys.stderr)

    while True:
        topic = sub.recv_string()
        payload = sub.recv_string()
        msg_count += 1

        if args.snapshot:
            print(topic)
            try:
                obj = json.loads(payload)
                print(json.dumps(obj, indent=2))
            except (json.JSONDecodeError, ValueError):
                print(payload)
            break

        if args.detail:
            print(f"--- {topic} ---")
            try:
                obj = json.loads(payload)
                print(json.dumps(obj, indent=2))
            except (json.JSONDecodeError, ValueError):
                print(payload)
            print()
        else:
            if topic not in seen_topics:
                seen_topics.add(topic)
                print(topic)

    on_exit()


if __name__ == "__main__":
    main()
