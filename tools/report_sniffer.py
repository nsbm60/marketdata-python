#!/usr/bin/env python3
"""
Report Sniffer - Monitor CalcServer report topics

Subscribes to CalcServer's ZMQ PUB socket and displays report messages.
Use to verify report generation is working correctly.

Usage:
    python report_sniffer.py                    # Subscribe to all report.* topics
    python report_sniffer.py watchlist          # Subscribe to report.watchlist.*
    python report_sniffer.py --mds              # Also listen on MDS port (6006)
    python report_sniffer.py --port 6020        # Custom port
"""
import zmq
import json
import sys
import argparse
from datetime import datetime

def format_report(topic: str, payload: str) -> str:
    """Format report payload for display."""
    try:
        data = json.loads(payload)

        # Format based on report type
        if "report.watchlist" in topic:
            name = data.get("name", "?")
            row_count = data.get("rowCount", 0)
            as_of = data.get("asOf", 0)
            ref_date = data.get("referenceDate", "N/A")

            timestamp = datetime.fromtimestamp(as_of / 1000).strftime("%H:%M:%S.%f")[:-3] if as_of else "N/A"

            lines = [f"Watchlist '{name}' @ {timestamp} (ref: {ref_date})"]
            lines.append(f"{'Symbol':<8} {'Last':>10} {'Change':>10} {'%Chg':>8}")
            lines.append("-" * 40)

            for row in data.get("rows", []):
                sym = row.get("symbol", "?")
                last = row.get("last", 0)
                change = row.get("change", 0)
                pct = row.get("pctChange", 0)

                change_str = f"{change:+.2f}" if change != 0 else "-"
                pct_str = f"{pct:+.2f}%" if pct != 0 else "-"

                lines.append(f"{sym:<8} {last:>10.2f} {change_str:>10} {pct_str:>8}")

            return "\n".join(lines)
        else:
            # Generic JSON pretty print
            return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        return payload

def main():
    parser = argparse.ArgumentParser(description="Monitor CalcServer report topics")
    parser.add_argument("filter", nargs="?", default="", help="Topic filter (e.g., 'watchlist', 'positions')")
    parser.add_argument("--port", type=int, default=6020, help="CalcServer pub port (default: 6020)")
    parser.add_argument("--mds", action="store_true", help="Also listen on MDS port (6006)")
    parser.add_argument("--raw", action="store_true", help="Show raw JSON instead of formatted")
    args = parser.parse_args()

    ctx = zmq.Context.instance()

    # Build subscription prefix
    if args.filter:
        prefix = f"report.{args.filter}"
    else:
        prefix = "report."

    # Connect to CalcServer
    calc_endpoint = f"tcp://127.0.0.1:{args.port}"
    sub = ctx.socket(zmq.SUB)
    sub.connect(calc_endpoint)
    sub.setsockopt_string(zmq.SUBSCRIBE, prefix)
    print(f"Subscribed to '{prefix}*' on CalcServer ({calc_endpoint})")

    # Optionally also listen on MDS
    if args.mds:
        mds_endpoint = "tcp://127.0.0.1:6006"
        sub.connect(mds_endpoint)
        print(f"Also listening on MDS ({mds_endpoint})")

    print("-" * 60)
    print("Waiting for reports... (Ctrl+C to exit)")
    print("-" * 60)

    try:
        while True:
            topic = sub.recv_string()
            payload = sub.recv_string()

            print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {topic}")

            if args.raw:
                try:
                    print(json.dumps(json.loads(payload), indent=2))
                except:
                    print(payload)
            else:
                print(format_report(topic, payload))

            print("-" * 60)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sub.close()
        ctx.term()

if __name__ == "__main__":
    main()
