"""
tools/test_breakout_candidates.py

Test CalcServer's get_breakout_candidates handler via ZMQ dealer/router.

Usage:
    PYTHONPATH=. python tools/test_breakout_candidates.py \
        --symbol NVDA \
        --timeframe 5m \
        --start-date 2026-04-09 \
        --end-date 2026-04-14
"""

import argparse
import json
from datetime import datetime

import zmq

from discovery.service_locator import ServiceLocator


def main():
    parser = argparse.ArgumentParser(
        description="Test get_breakout_candidates via CalcServer ZMQ",
    )
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default=None)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    end_date = args.end_date or args.start_date

    calc = ServiceLocator.wait_for_service(ServiceLocator.CALC, timeout_sec=10)
    print(f"CalcServer found: {calc.router}")

    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 10000)
    dealer.connect(calc.router)

    request = {
        "op": "get_breakout_candidates",
        "symbol": args.symbol.upper(),
        "start_date": args.start_date,
        "end_date": end_date,
    }
    if args.timeframe:
        request["timeframe"] = args.timeframe

    print(f"\nRequest: {json.dumps(request, indent=2)}")

    try:
        dealer.send_string(json.dumps(request))

        if dealer.poll(timeout=10000):
            raw = dealer.recv_string()
            response = json.loads(raw)

            print(f"\nRaw response:\n{json.dumps(response, indent=2)}")

            if response.get("ok"):
                data = response.get("data", {})
                candidates = data.get("candidates", [])
                print(f"\n--- Summary: {len(candidates)} candidates ---")
                for c in candidates:
                    ts_ms = c.get("ts", 0)
                    ts_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M")
                    direction = c.get("direction", "?")
                    tf = c.get("timeframe", "?")
                    score = c.get("score", 0)
                    eod = c.get("return_atr_eod")
                    eod_str = f"{eod:+.2f}" if eod is not None else "null"
                    print(f"  {ts_str}  {direction:5s}  {tf:4s}  score={score:.2f}  eod={eod_str}")
            else:
                print(f"\nError: {response.get('error', 'unknown')}")
        else:
            print("\nTimeout waiting for response")
    finally:
        dealer.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
