#!/usr/bin/env python3
"""
Options Report Sniffer - Capture options reports and check for delta/theta.

Supports two report types:
- report.options.* - Full chain reports (OptionsReportBuilder)
- report.portfolio.options.* - Portfolio position reports (PortfolioOptionsReport)

Usage:
    python tools/options_report_sniffer.py                # All options reports
    python tools/options_report_sniffer.py NVDA           # Filter by underlying (chain reports)
    python tools/options_report_sniffer.py --portfolio    # Portfolio options reports only
    python tools/options_report_sniffer.py --raw          # Raw JSON output
"""
import zmq
import json
import sys
import argparse
from datetime import datetime

def analyze_report(data: dict) -> dict:
    """Analyze report for delta/theta presence."""
    analysis = {
        "row_count": 0,
        "calls_with_delta": 0,
        "calls_with_theta": 0,
        "puts_with_delta": 0,
        "puts_with_theta": 0,
        "sample_call": None,
        "sample_put": None,
    }

    rows = data.get("rows", [])
    analysis["row_count"] = len(rows)

    for row in rows:
        call = row.get("call", {})
        put = row.get("put", {})

        if call:
            if call.get("delta") is not None:
                analysis["calls_with_delta"] += 1
            if call.get("theta") is not None:
                analysis["calls_with_theta"] += 1
            if analysis["sample_call"] is None and call:
                analysis["sample_call"] = {
                    "strike": row.get("strike"),
                    **{k: v for k, v in call.items() if k in ["delta", "theta", "gamma", "vega", "iv", "last", "bid", "ask", "mid", "theo"]}
                }

        if put:
            if put.get("delta") is not None:
                analysis["puts_with_delta"] += 1
            if put.get("theta") is not None:
                analysis["puts_with_theta"] += 1
            if analysis["sample_put"] is None and put:
                analysis["sample_put"] = {
                    "strike": row.get("strike"),
                    **{k: v for k, v in put.items() if k in ["delta", "theta", "gamma", "vega", "iv", "last", "bid", "ask", "mid", "theo"]}
                }

    return analysis

def format_report(topic: str, data: dict) -> str:
    """Format report for display."""
    lines = []

    # Parse topic
    parts = topic.split(".")
    underlying = parts[2] if len(parts) > 2 else "?"
    expiry = "-".join(parts[3:]) if len(parts) > 3 else "?"

    lines.append(f"Underlying: {underlying}, Expiry: {expiry}")
    lines.append(f"Spot: {data.get('spot', 'N/A')}")

    analysis = analyze_report(data)
    lines.append(f"Rows: {analysis['row_count']}")
    lines.append(f"Calls with delta: {analysis['calls_with_delta']}/{analysis['row_count']}")
    lines.append(f"Calls with theta: {analysis['calls_with_theta']}/{analysis['row_count']}")
    lines.append(f"Puts with delta: {analysis['puts_with_delta']}/{analysis['row_count']}")
    lines.append(f"Puts with theta: {analysis['puts_with_theta']}/{analysis['row_count']}")

    if analysis["sample_call"]:
        lines.append(f"\nSample Call (strike {analysis['sample_call'].get('strike')}):")
        for k, v in analysis["sample_call"].items():
            if k != "strike":
                lines.append(f"  {k}: {v}")

    if analysis["sample_put"]:
        lines.append(f"\nSample Put (strike {analysis['sample_put'].get('strike')}):")
        for k, v in analysis["sample_put"].items():
            if k != "strike":
                lines.append(f"  {k}: {v}")

    return "\n".join(lines)

def format_portfolio_report(topic: str, data: dict) -> str:
    """Format portfolio options report for display."""
    lines = []

    # Parse topic: report.portfolio.options.{clientId}
    parts = topic.split(".")
    client_id = parts[3] if len(parts) > 3 else "?"

    lines.append(f"Client ID: {client_id}")
    lines.append(f"Contract Count: {data.get('contractCount', 'N/A')}")

    options = data.get("options", [])
    with_delta = sum(1 for o in options if o.get("delta") is not None)
    with_theta = sum(1 for o in options if o.get("theta") is not None)

    lines.append(f"Options: {len(options)}")
    lines.append(f"With delta: {with_delta}/{len(options)}")
    lines.append(f"With theta: {with_theta}/{len(options)}")

    # Show sample options
    for i, opt in enumerate(options[:3]):
        lines.append(f"\nOption {i+1}: {opt.get('symbol', '?')}")
        for k in ["last", "bid", "ask", "mid", "delta", "gamma", "theta", "vega", "iv"]:
            if k in opt and opt[k] is not None:
                lines.append(f"  {k}: {opt[k]}")

    if len(options) > 3:
        lines.append(f"\n  ... and {len(options) - 3} more")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Monitor CalcServer options reports")
    parser.add_argument("underlying", nargs="?", default="", help="Filter by underlying (e.g., NVDA)")
    parser.add_argument("--port", type=int, default=6020, help="CalcServer pub port (default: 6020)")
    parser.add_argument("--raw", action="store_true", help="Show raw JSON")
    parser.add_argument("--portfolio", action="store_true", help="Show portfolio options reports only")
    args = parser.parse_args()

    ctx = zmq.Context.instance()

    endpoint = f"tcp://127.0.0.1:{args.port}"
    sub = ctx.socket(zmq.SUB)
    sub.connect(endpoint)

    # Subscribe to appropriate topics
    if args.portfolio:
        prefix = "report.portfolio.options."
        sub.setsockopt_string(zmq.SUBSCRIBE, prefix)
        print(f"Subscribed to '{prefix}*' on {endpoint}")
    elif args.underlying:
        prefix = f"report.options.{args.underlying.lower()}."
        sub.setsockopt_string(zmq.SUBSCRIBE, prefix)
        print(f"Subscribed to '{prefix}*' on {endpoint}")
    else:
        # Subscribe to both chain and portfolio reports
        sub.setsockopt_string(zmq.SUBSCRIBE, "report.options.")
        sub.setsockopt_string(zmq.SUBSCRIBE, "report.portfolio.options.")
        print(f"Subscribed to 'report.options.*' and 'report.portfolio.options.*' on {endpoint}")

    print("-" * 60)
    print("Waiting for options reports... (Ctrl+C to exit)")
    print("-" * 60)

    try:
        while True:
            topic = sub.recv_string()
            payload = sub.recv_string()

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"\n[{timestamp}] {topic}")

            try:
                data = json.loads(payload)
                # Handle wrapped payload (tick.data.data vs tick.data)
                if "data" in data and isinstance(data["data"], dict):
                    data = data["data"]

                if args.raw:
                    print(json.dumps(data, indent=2))
                elif topic.startswith("report.portfolio.options."):
                    print(format_portfolio_report(topic, data))
                else:
                    print(format_report(topic, data))
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(payload[:500])

            print("-" * 60)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        sub.close()
        ctx.term()

if __name__ == "__main__":
    main()
