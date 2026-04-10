#!/usr/bin/env python3
"""
chart_indicators.py

Plot candlestick chart with EMA ribbon and ATR for a symbol/timeframe/date.
Queries stock_bar and indicator tables from ClickHouse.
Outputs a standalone HTML file.

Usage:
    python tools/chart_indicators.py --symbol NVDA --timeframe 5m
    python tools/chart_indicators.py --symbol NVDA --timeframe 5m --date 2026-04-09
    python tools/chart_indicators.py --symbol NVDA --timeframe 1m --date 2026-04-09 --session regular

Requirements:
    pip install plotly clickhouse-connect

Environment variables:
    CLICKHOUSE_USER     (default: default)
    CLICKHOUSE_PASSWORD (default: Aector99)
    CLICKHOUSE_DATABASE (default: trading)
"""

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import clickhouse_connect
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discovery.service_locator import ServiceLocator

load_dotenv()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Chart EMA ribbon + ATR for a symbol/timeframe")
parser.add_argument("--symbol",    default="NVDA",  help="Ticker symbol (default: NVDA)")
parser.add_argument("--timeframe", default="5m",    help="Timeframe (default: 5m)")
parser.add_argument("--date",      default=None,    help="Session date YYYY-MM-DD (default: today)")
parser.add_argument("--session",   default="all", choices=["regular", "pre", "post", "all"],
                    help="Session filter (default: all)")
parser.add_argument("--out",       default=None,    help="Output HTML file (default: auto)")
args = parser.parse_args()

symbol    = args.symbol.upper()
timeframe = args.timeframe
session_date = args.date or date.today().isoformat()
session_filter = args.session
out_path = args.out or f"chart_{symbol}_{timeframe}_{session_date}.html"

print(f"Charting {symbol} {timeframe} {session_date} session={session_filter}")

# ---------------------------------------------------------------------------
# Connect to ClickHouse
# ---------------------------------------------------------------------------
print("Discovering ClickHouse...")
endpoint = ServiceLocator.wait_for_service(
    service_name=ServiceLocator.CLICKHOUSE,
    timeout_sec=30,
)

ch = clickhouse_connect.get_client(
    host     = endpoint.host,
    port     = endpoint.port,
    username = os.environ.get("CLICKHOUSE_USER",     "default"),
    password = os.environ.get("CLICKHOUSE_PASSWORD", "Aector99"),
    database = os.environ.get("CLICKHOUSE_DATABASE", "trading"),
)

# ---------------------------------------------------------------------------
# Query bars
# ---------------------------------------------------------------------------
session_clause = "" if session_filter == "all" else f"AND session = '{session_filter}'"

bars_sql = f"""
SELECT
    toDateTime(ts, 'UTC')   AS ts,
    open, high, low, close, volume
FROM stock_bar FINAL
WHERE symbol   = '{symbol}'
  AND period   = '{timeframe}'
  AND toDate(toTimezone(ts, 'America/New_York')) = '{session_date}'
  {session_clause}
ORDER BY ts
"""

bars_result = ch.query(bars_sql)
bars = bars_result.result_rows

if not bars:
    print(f"No bars found for {symbol} {timeframe} {session_date}")
    sys.exit(1)

print(f"Loaded {len(bars)} bars")

ts_list    = [r[0] for r in bars]
open_list  = [r[1] for r in bars]
high_list  = [r[2] for r in bars]
low_list   = [r[3] for r in bars]
close_list = [r[4] for r in bars]

# ---------------------------------------------------------------------------
# Query indicators
# ---------------------------------------------------------------------------
indicators_sql = f"""
SELECT
    toDateTime(ts, 'UTC')   AS ts,
    indicator,
    value,
    warm
FROM indicator FINAL
WHERE symbol    = '{symbol}'
  AND timeframe = '{timeframe}'
  AND session   = '{session_date}'
  AND warm      = true
ORDER BY ts, indicator
"""

ind_result = ch.query(indicators_sql)
ind_rows = ind_result.result_rows

if not ind_rows:
    print(f"No indicator data found for {symbol} {timeframe} {session_date}")
    sys.exit(1)

print(f"Loaded {len(ind_rows)} indicator rows")

# Pivot indicators into per-series dicts
from collections import defaultdict
ind_data = defaultdict(lambda: {"ts": [], "value": [], "warm": []})
for ts, indicator, value, warm in ind_rows:
    ind_data[indicator]["ts"].append(ts)
    ind_data[indicator]["value"].append(value if not (value != value) else None)  # NaN → None
    ind_data[indicator]["warm"].append(warm)

# EMA periods and colors
ema_periods = sorted([k for k in ind_data.keys() if k.startswith("ema")],
                     key=lambda x: int(x[3:]))
ema_colors = {
    "ema10": "#ff6b6b",
    "ema15": "#ffa94d",
    "ema20": "#ffe066",
    "ema25": "#69db7c",
    "ema30": "#74c0fc",
}

# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.75, 0.25],
    subplot_titles=[f"{symbol} {timeframe} — {session_date}", "ATR(14)"]
)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=ts_list,
    open=open_list, high=high_list,
    low=low_list,   close=close_list,
    name="Price",
    increasing_line_color="#69db7c",
    decreasing_line_color="#ff6b6b",
    increasing_fillcolor="#69db7c",
    decreasing_fillcolor="#ff6b6b",
    line=dict(width=1),
), row=1, col=1)

# EMA ribbon
for ema_key in ema_periods:
    d = ind_data[ema_key]
    color = ema_colors.get(ema_key, "#aaa")
    period = ema_key[3:]
    fig.add_trace(go.Scatter(
        x=d["ts"],
        y=d["value"],
        name=f"EMA({period})",
        line=dict(color=color, width=1.5),
        opacity=0.85,
    ), row=1, col=1)

# ATR
if "atr" in ind_data:
    d = ind_data["atr"]
    fig.add_trace(go.Scatter(
        x=d["ts"],
        y=d["value"],
        name="ATR(14)",
        line=dict(color="#a9e34b", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(169, 227, 75, 0.15)",
    ), row=2, col=1)

# Layout
fig.update_layout(
    title=f"{symbol} {timeframe} — EMA Ribbon + ATR(14) — {session_date}",
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#16213e",
    font=dict(color="#e0e0e0", size=11),
    xaxis_rangeslider_visible=False,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1,
        bgcolor="rgba(0,0,0,0.3)",
    ),
    height=800,
    margin=dict(l=60, r=40, t=80, b=40),
)

fig.update_xaxes(
    gridcolor="#2a2a4a",
    showgrid=True,
)
fig.update_yaxes(
    gridcolor="#2a2a4a",
    showgrid=True,
)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Chart written to {out_path}")
print(f"Open in browser: open {out_path}")
