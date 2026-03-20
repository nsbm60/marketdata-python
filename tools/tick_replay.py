#!/usr/bin/env python3
"""
Tick Replay - Replay historical Alpaca data through MDS

Fetches historical bars or trades from Alpaca and replays them
through the tick injector for realistic testing.

Usage:
    python tick_replay.py AAPL NVDA --date 2024-12-30
    python tick_replay.py AAPL --date 2024-12-30 --speed 10
    python tick_replay.py --watchlist default --date 2024-12-30
    python tick_replay.py AAPL --start 2024-12-30 --end 2024-12-31 --timeframe 1min

Requires:
    - ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
    - pip install alpaca-py
"""
import zmq
import json
import sys
import argparse
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockTradesRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    print("Error: alpaca-py not installed. Run: pip install alpaca-py")
    sys.exit(1)

# Alpaca has limits on trades requests
MAX_TRADES_PER_REQUEST = 10000

MDS_CONTROL_PORT = 6007


def get_alpaca_client() -> StockHistoricalDataClient:
    """Create Alpaca client from environment variables."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("Error: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)

    return StockHistoricalDataClient(api_key, secret_key)


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Parse timeframe string like '1min', '5min', '1hour', '1day'."""
    tf_str = tf_str.lower().strip()

    if tf_str in ("1min", "1m"):
        return TimeFrame(1, TimeFrameUnit.Minute)
    elif tf_str in ("5min", "5m"):
        return TimeFrame(5, TimeFrameUnit.Minute)
    elif tf_str in ("15min", "15m"):
        return TimeFrame(15, TimeFrameUnit.Minute)
    elif tf_str in ("1hour", "1h"):
        return TimeFrame(1, TimeFrameUnit.Hour)
    elif tf_str in ("1day", "1d"):
        return TimeFrame(1, TimeFrameUnit.Day)
    else:
        print(f"Warning: Unknown timeframe '{tf_str}', using 1min")
        return TimeFrame(1, TimeFrameUnit.Minute)


def fetch_bars(client: StockHistoricalDataClient, symbols: List[str],
               start_date: str, end_date: str, timeframe: TimeFrame) -> List[Dict]:
    """Fetch historical bars from Alpaca."""
    print(f"Fetching {timeframe} bars for {symbols} from {start_date} to {end_date}...")

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        start=datetime.strptime(start_date, "%Y-%m-%d"),
        end=datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1),
        timeframe=timeframe
    )

    bars = client.get_stock_bars(request)

    # Flatten and sort by timestamp
    all_bars = []
    for symbol, symbol_bars in bars.data.items():
        for bar in symbol_bars:
            all_bars.append({
                "symbol": symbol,
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
                "vwap": float(bar.vwap) if bar.vwap else None
            })

    # Sort by timestamp
    all_bars.sort(key=lambda x: x["timestamp"])
    print(f"Fetched {len(all_bars)} bars")
    return all_bars


def fetch_trades(client: StockHistoricalDataClient, symbols: List[str],
                 start_date: str, end_date: str, limit: int = None) -> List[Dict]:
    """Fetch historical trades (tick-by-tick) from Alpaca."""
    print(f"Fetching trades for {symbols} from {start_date} to {end_date}...")
    print("(This may take a while for active symbols...)")

    request = StockTradesRequest(
        symbol_or_symbols=symbols,
        start=datetime.strptime(start_date, "%Y-%m-%d"),
        end=datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1),
        limit=limit
    )

    trades = client.get_stock_trades(request)

    # Flatten and sort by timestamp
    all_trades = []
    for symbol, symbol_trades in trades.data.items():
        for trade in symbol_trades:
            all_trades.append({
                "symbol": symbol,
                "timestamp": trade.timestamp,
                "price": float(trade.price),
                "size": int(trade.size),
                "exchange": trade.exchange,
                "conditions": trade.conditions if trade.conditions else []
            })

    # Sort by timestamp
    all_trades.sort(key=lambda x: x["timestamp"])
    print(f"Fetched {len(all_trades)} trades")
    return all_trades


def fetch_watchlist_symbols(sock) -> List[str]:
    """Fetch symbols from the active watchlist via MDS."""
    request = {"op": "get_active_watchlist"}
    sock.send_string(json.dumps(request))
    response = json.loads(sock.recv_string())

    if response.get("ok"):
        data = response.get("data", {})
        symbols = data.get("symbols", [])
        name = data.get("name", "unknown")
        print(f"Loaded watchlist '{name}' with {len(symbols)} symbols")
        return symbols
    else:
        print(f"Error fetching watchlist: {response.get('error')}")
        return []


def inject_tick(sock, symbol: str, price: float, bid: float = None, ask: float = None,
                volume: int = None) -> bool:
    """Inject a single tick via MDS control."""
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
    if volume is not None:
        request["volume"] = volume

    sock.send_string(json.dumps(request))
    response = json.loads(sock.recv_string())
    return response.get("ok", False)


def replay_bars(sock, bars: List[Dict], speed: float = 1.0, use_close: bool = True):
    """
    Replay bars through tick injector.

    Args:
        sock: ZMQ socket to MDS control
        bars: List of bar dicts sorted by timestamp
        speed: Replay speed multiplier (10 = 10x faster)
        use_close: If True, use close price; if False, replay OHLC sequence
    """
    if not bars:
        print("No bars to replay")
        return

    print(f"\nReplaying {len(bars)} bars at {speed}x speed...")
    print("Press Ctrl+C to stop\n")
    print("-" * 60)

    start_time = time.time()
    first_bar_time = bars[0]["timestamp"]
    tick_count = 0
    last_print = 0

    # Track latest price for each symbol
    latest_prices: Dict[str, float] = {}
    latest_time = None

    try:
        for i, bar in enumerate(bars):
            symbol = bar["symbol"]

            # Calculate target replay time
            bar_time = bar["timestamp"]
            elapsed_market = (bar_time - first_bar_time).total_seconds()
            target_elapsed = elapsed_market / speed

            # Wait until it's time for this bar
            actual_elapsed = time.time() - start_time
            if target_elapsed > actual_elapsed:
                time.sleep(target_elapsed - actual_elapsed)

            if use_close:
                # Simple mode: just inject close price
                price = bar["close"]
                # Estimate bid/ask from bar range
                spread = max(0.01, (bar["high"] - bar["low"]) * 0.1)
                bid = price - spread / 2
                ask = price + spread / 2

                if inject_tick(sock, symbol, price, bid, ask, bar.get("volume")):
                    tick_count += 1
                    latest_prices[symbol] = price
                    latest_time = bar_time
            else:
                # Detailed mode: replay O-H-L-C sequence
                for price in [bar["open"], bar["high"], bar["low"], bar["close"]]:
                    spread = max(0.01, price * 0.001)
                    if inject_tick(sock, symbol, price, price - spread/2, price + spread/2):
                        tick_count += 1
                    time.sleep(0.01)  # Small delay between OHLC
                latest_prices[symbol] = bar["close"]
                latest_time = bar_time

            # Progress output (throttled) - show ALL symbols
            now = time.time()
            if now - last_print >= 0.5 and latest_time:
                bar_time_str = latest_time.strftime("%H:%M:%S")
                pct = (i + 1) / len(bars) * 100
                prices_str = " | ".join(
                    f"{sym}: ${px:.2f}" for sym, px in sorted(latest_prices.items())
                )
                print(f"[{bar_time_str}] {prices_str}  ({pct:.1f}%)")
                last_print = now

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Replayed {tick_count} ticks in {elapsed:.1f}s")


def replay_trades(sock, trades: List[Dict], speed: float = 1.0):
    """
    Replay trades tick-by-tick through injector.

    Args:
        sock: ZMQ socket to MDS control
        trades: List of trade dicts sorted by timestamp
        speed: Replay speed multiplier (10 = 10x faster)
    """
    if not trades:
        print("No trades to replay")
        return

    print(f"\nReplaying {len(trades)} trades at {speed}x speed...")
    print("Press Ctrl+C to stop\n")
    print("-" * 60)

    start_time = time.time()
    first_trade_time = trades[0]["timestamp"]
    tick_count = 0
    last_print = 0

    # Track latest price for each symbol
    latest_prices: Dict[str, float] = {}
    latest_time = None

    try:
        for i, trade in enumerate(trades):
            symbol = trade["symbol"]
            price = trade["price"]

            # Calculate target replay time
            trade_time = trade["timestamp"]
            elapsed_market = (trade_time - first_trade_time).total_seconds()
            target_elapsed = elapsed_market / speed

            # Wait until it's time for this trade
            actual_elapsed = time.time() - start_time
            if target_elapsed > actual_elapsed:
                time.sleep(target_elapsed - actual_elapsed)

            # Inject the trade (no bid/ask for raw trades)
            # Estimate a tight spread around the trade price
            spread = max(0.01, price * 0.0005)  # 0.05% spread
            bid = price - spread / 2
            ask = price + spread / 2

            if inject_tick(sock, symbol, price, bid, ask, trade.get("size")):
                tick_count += 1
                latest_prices[symbol] = price
                latest_time = trade_time

            # Progress output (throttled) - show ALL symbols
            now = time.time()
            if now - last_print >= 0.5 and latest_time:
                trade_time_str = latest_time.strftime("%H:%M:%S.%f")[:-3]
                pct = (i + 1) / len(trades) * 100
                prices_str = " | ".join(
                    f"{sym}: ${px:.2f}" for sym, px in sorted(latest_prices.items())
                )
                print(f"[{trade_time_str}] {prices_str}  ({pct:.1f}%)")
                last_print = now

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    elapsed = time.time() - start_time
    tps = tick_count / elapsed if elapsed > 0 else 0
    print("-" * 60)
    print(f"Replayed {tick_count} trades in {elapsed:.1f}s ({tps:.1f} ticks/sec)")


def main():
    parser = argparse.ArgumentParser(
        description="Replay historical Alpaca data through MDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replay yesterday's data for specific symbols (using 1min bars)
    python tick_replay.py AAPL NVDA TSLA --date 2024-12-30

    # Replay at 10x speed
    python tick_replay.py AAPL --date 2024-12-30 --speed 10

    # Replay active watchlist
    python tick_replay.py --watchlist --date 2024-12-30

    # Use 5-minute bars
    python tick_replay.py AAPL --date 2024-12-30 --timeframe 5min

    # Replay actual tick-by-tick trades (most realistic)
    python tick_replay.py AAPL --date 2024-12-30 --trades --speed 10

    # Limit number of trades (for testing)
    python tick_replay.py AAPL --date 2024-12-30 --trades --limit 10000
        """
    )
    parser.add_argument("symbols", nargs="*", help="Symbols to replay")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1min",
                        help="Bar timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1min)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay speed multiplier (default: 1.0 = realtime)")
    parser.add_argument("--watchlist", action="store_true",
                        help="Use symbols from active MDS watchlist")
    parser.add_argument("--trades", action="store_true",
                        help="Replay actual tick-by-tick trades instead of bars")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of trades to fetch (only with --trades)")
    parser.add_argument("--ohlc", action="store_true",
                        help="Replay full OHLC sequence instead of just close (bars only)")
    parser.add_argument("--port", type=int, default=MDS_CONTROL_PORT,
                        help="MDS control port (default: 6007)")
    args = parser.parse_args()

    # Determine date range
    if args.date:
        start_date = args.date
        end_date = args.date
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        # Default to previous trading day
        today = datetime.now()
        # Simple heuristic: go back 1-3 days to skip weekends
        for days_back in range(1, 5):
            check = today - timedelta(days=days_back)
            if check.weekday() < 5:  # Monday=0, Friday=4
                start_date = end_date = check.strftime("%Y-%m-%d")
                print(f"Using previous trading day: {start_date}")
                break
        else:
            print("Error: Could not determine previous trading day. Use --date.")
            sys.exit(1)

    # Connect to MDS control
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    print(f"Connected to MDS control on port {args.port}")

    try:
        # Determine symbols
        if args.watchlist:
            symbols = fetch_watchlist_symbols(sock)
            if not symbols:
                print("No symbols in watchlist")
                sys.exit(1)
        elif args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            print("Error: Provide symbols or use --watchlist")
            parser.print_help()
            sys.exit(1)

        print(f"Symbols: {', '.join(symbols)}")

        # Fetch historical data
        client = get_alpaca_client()

        if args.trades:
            # Tick-by-tick trades (most realistic)
            trades = fetch_trades(client, symbols, start_date, end_date, limit=args.limit)
            if not trades:
                print("No trades returned from Alpaca")
                sys.exit(1)
            replay_trades(sock, trades, speed=args.speed)
        else:
            # Bar-based replay
            timeframe = parse_timeframe(args.timeframe)
            bars = fetch_bars(client, symbols, start_date, end_date, timeframe)
            if not bars:
                print("No bars returned from Alpaca")
                sys.exit(1)
            replay_bars(sock, bars, speed=args.speed, use_close=not args.ohlc)

    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
