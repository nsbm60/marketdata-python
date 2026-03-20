# MarketData Python

Python tools and utilities for the MarketData system.

## Structure

```
tools/          - ZMQ utilities, testing scripts
ml/             - Machine learning (future)
notebooks/      - Jupyter notebooks for exploration
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tools

### report_sniffer.py
Monitor CalcServer report topics.

```bash
python tools/report_sniffer.py                  # All reports
python tools/report_sniffer.py watchlist        # Watchlist reports only
python tools/report_sniffer.py --raw            # Raw JSON output
```

### tick_injector.py
Inject synthetic ticks for testing without live market data.

```bash
python tools/tick_injector.py AAPL 185.50                        # Single tick
python tools/tick_injector.py AAPL 185.50 --bid 185.48 --ask 185.52
python tools/tick_injector.py --simulate AAPL NVDA TSLA          # Continuous
```

### tick_replay.py
Replay historical Alpaca data for realistic testing.

Requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables.

```bash
# Bar-based replay (default: 1min bars)
python tools/tick_replay.py AAPL NVDA --date 2024-12-30          # Specific date
python tools/tick_replay.py AAPL --date 2024-12-30 --speed 10    # 10x speed
python tools/tick_replay.py --watchlist --date 2024-12-30        # Active watchlist
python tools/tick_replay.py AAPL --timeframe 5min --speed 60     # 5min bars at 60x

# Tick-by-tick trades (most realistic)
python tools/tick_replay.py AAPL --date 2024-12-30 --trades      # Every trade
python tools/tick_replay.py AAPL --trades --speed 100            # 100x speed
python tools/tick_replay.py AAPL --trades --limit 50000          # Cap at 50k trades
```

Note: Active symbols can have 100k+ trades per day. Use `--limit` or higher `--speed` for faster testing.

### zmq_subscriber.py
Basic ZMQ subscriber for equity quotes.

### sub_dump.py
Dump all ZMQ traffic for debugging.

### ib_monitor.py
Monitor IB Gateway messages.

## Ports

| Service | PubSub | Control |
|---------|--------|---------|
| MDS     | 6006   | 6007    |
| IB      | 6010   | 6011    |
| Calc    | 6020   | 6021    |
