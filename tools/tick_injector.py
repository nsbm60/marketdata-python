#!/usr/bin/env python3
"""
Tick Injector - Inject synthetic ticks via MDS control

Sends inject_tick commands to MarketDataControlService for testing
without live market data.

Usage:
    python tick_injector.py AAPL 185.50                    # Single tick
    python tick_injector.py AAPL 185.50 --bid 185.48 --ask 185.52
    python tick_injector.py --simulate AAPL NVDA TSLA     # Continuous simulation
    python tick_injector.py --file ticks.json             # Batch from file
"""
import zmq
import json
import sys
import argparse
import time
import random
from datetime import datetime

MDS_CONTROL_PORT = 6007

def send_control(sock, request: dict) -> dict:
    """Send control request and wait for response."""
    sock.send_string(json.dumps(request))
    response = sock.recv_string()
    return json.loads(response)

def inject_equity_tick(sock, symbol: str, price: float, bid: float = None, ask: float = None) -> bool:
    """Inject a single equity tick."""
    request = {
        "op": "inject_tick",
        "kind": "equity",
        "symbol": symbol.upper(),
        "price": price
    }
    if bid is not None:
        request["bid"] = bid
    if ask is not None:
        request["ask"] = ask

    response = send_control(sock, request)
    return response.get("ok", False)

def inject_option_tick(sock, symbol: str, price: float, bid: float = None, ask: float = None,
                       delta: float = None, theta: float = None) -> bool:
    """Inject a single option tick."""
    request = {
        "op": "inject_tick",
        "kind": "option",
        "symbol": symbol.upper(),
        "price": price
    }
    if bid is not None:
        request["bid"] = bid
    if ask is not None:
        request["ask"] = ask
    if delta is not None:
        request["delta"] = delta
    if theta is not None:
        request["theta"] = theta

    response = send_control(sock, request)
    return response.get("ok", False)

def simulate_prices(sock, symbols: list, duration: int = 60, interval: float = 0.25):
    """Simulate continuous price updates with random walk."""
    print(f"Simulating {len(symbols)} symbols for {duration}s (interval={interval}s)")
    print("-" * 50)

    # Initialize base prices
    prices = {sym: 100 + random.random() * 200 for sym in symbols}

    end_time = time.time() + duration
    tick_count = 0

    try:
        while time.time() < end_time:
            for sym in symbols:
                # Random walk: +/- 0.5%
                change = prices[sym] * (random.random() - 0.5) * 0.01
                prices[sym] = max(1.0, prices[sym] + change)

                # Add bid/ask spread
                spread = prices[sym] * 0.001  # 0.1% spread
                bid = prices[sym] - spread / 2
                ask = prices[sym] + spread / 2

                if inject_equity_tick(sock, sym, prices[sym], bid, ask):
                    tick_count += 1

            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            status = " | ".join([f"{s}: ${prices[s]:.2f}" for s in symbols])
            print(f"[{timestamp}] {status}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\nInjected {tick_count} ticks")

def main():
    parser = argparse.ArgumentParser(description="Inject synthetic ticks via MDS control")
    parser.add_argument("symbols", nargs="*", help="Symbols to inject (e.g., AAPL 185.50)")
    parser.add_argument("--port", type=int, default=MDS_CONTROL_PORT, help="MDS control port")
    parser.add_argument("--bid", type=float, help="Bid price")
    parser.add_argument("--ask", type=float, help="Ask price")
    parser.add_argument("--simulate", action="store_true", help="Run continuous simulation")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--interval", type=float, default=0.25, help="Interval between ticks")
    parser.add_argument("--file", type=str, help="JSON file with ticks to inject")
    args = parser.parse_args()

    # Connect to MDS control
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    print(f"Connected to MDS control on port {args.port}")

    try:
        if args.file:
            # Batch inject from file
            with open(args.file) as f:
                ticks = json.load(f)
            request = {"op": "inject_ticks", "ticks": ticks}
            response = send_control(sock, request)
            if response.get("ok"):
                print(f"Injected {response.get('data', {}).get('injected', 0)} ticks")
            else:
                print(f"Error: {response.get('error')}")

        elif args.simulate:
            # Continuous simulation
            if not args.symbols:
                print("Error: Provide symbols for simulation (e.g., --simulate AAPL NVDA TSLA)")
                sys.exit(1)
            simulate_prices(sock, args.symbols, args.duration, args.interval)

        elif len(args.symbols) >= 2:
            # Single tick: symbol price
            symbol = args.symbols[0]
            try:
                price = float(args.symbols[1])
            except ValueError:
                print(f"Error: Invalid price '{args.symbols[1]}'")
                sys.exit(1)

            if inject_equity_tick(sock, symbol, price, args.bid, args.ask):
                print(f"Injected: {symbol} @ ${price:.2f}")
                if args.bid or args.ask:
                    print(f"  Bid: {args.bid}, Ask: {args.ask}")
            else:
                print("Injection failed")

        else:
            parser.print_help()

    finally:
        sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
