"""
tools/chart_day.py

Render a single day's bars with breakout candidates overlaid.
Opens in browser via Plotly.

Usage:
    PYTHONPATH=. python tools/chart_day.py \
        --symbol NVDA \
        --date 2026-04-10 \
        --timeframe 5m
"""

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml.shared.clickhouse import get_ch_client
from ml.shared.utils import utc_dt

TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "10m": 10, "15m": 15, "20m": 20, "30m": 30, "60m": 60,
}

MT = ZoneInfo("America/Denver")


def to_mt(dt: datetime) -> datetime:
    """Convert UTC datetime to Mountain Time."""
    return dt.astimezone(MT)


def marker_color(mfe, mae):
    if mfe is None or mae is None:
        return "#888888"
    if mfe > 1.0 and mae < 1.0:
        return "#00bcd4"
    if mfe > 1.0:
        return "#ff9800"
    return "#ef5350"


def query_bars(ch, symbol, timeframe, trading_date):
    result = ch.query(
        "SELECT ts, open, high, low, close, volume, session "
        "FROM trading.stock_bar FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND period = %(timeframe)s "
        "  AND trading_date = %(date)s "
        "ORDER BY ts",
        parameters={"symbol": symbol, "timeframe": timeframe, "date": trading_date},
    )
    return [
        {
            "ts": utc_dt(r[0]),
            "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4]),
            "volume": int(r[5]),
            "session": r[6],
        }
        for r in result.result_rows
    ]


def query_candidates(ch, symbol, timeframe, trading_date):
    result = ch.query(
        "SELECT bc.ts, bc.direction, bc.price, bc.level_price, bc.level_age_min, "
        "       bc.score, bc.atr, "
        "       bo.max_favorable_excursion, bo.max_adverse_excursion, "
        "       bo.return_atr_eod, "
        "       bo.return_atr_pivot, bo.exit_bars_pivot, "
        "       bo.return_atr_ema30, bo.exit_bars_ema30 "
        "FROM trading.breakout_candidate bc FINAL "
        "LEFT JOIN trading.breakout_outcome bo FINAL "
        "    ON bc.symbol = bo.symbol "
        "    AND bc.timeframe = bo.timeframe "
        "    AND bc.ts = bo.ts "
        "WHERE bc.symbol = %(symbol)s "
        "  AND bc.timeframe = %(timeframe)s "
        "  AND bc.session = %(date)s "
        "ORDER BY bc.ts",
        parameters={"symbol": symbol, "timeframe": timeframe, "date": trading_date},
    )
    return [
        {
            "ts": utc_dt(r[0]),
            "direction": r[1],
            "price": float(r[2]),
            "level_price": float(r[3]),
            "level_age_min": int(r[4]),
            "score": float(r[5]),
            "atr": float(r[6]),
            "mfe": float(r[7]) if r[7] is not None else None,
            "mae": float(r[8]) if r[8] is not None else None,
            "return_atr_eod": float(r[9]) if r[9] is not None else None,
            "return_atr_pivot": float(r[10]) if r[10] is not None else None,
            "exit_bars_pivot": int(r[11]) if r[11] is not None else None,
            "return_atr_ema30": float(r[12]) if r[12] is not None else None,
            "exit_bars_ema30": int(r[13]) if r[13] is not None else None,
        }
        for r in result.result_rows
    ]


def query_emas(ch, symbol, timeframe, trading_date):
    """Fetch warm EMA ribbon values for the day."""
    result = ch.query(
        "SELECT ts, indicator, value "
        "FROM trading.indicator FINAL "
        "WHERE symbol = %(symbol)s "
        "  AND timeframe = %(tf)s "
        "  AND trading_date = %(date)s "
        "  AND indicator IN ('ema10', 'ema15', 'ema20', 'ema25', 'ema30') "
        "  AND warm = true "
        "ORDER BY ts",
        parameters={"symbol": symbol, "tf": timeframe, "date": trading_date},
    )
    ema_data = {}
    for r in result.result_rows:
        name = r[1]
        if name not in ema_data:
            ema_data[name] = {"ts": [], "value": []}
        ema_data[name]["ts"].append(utc_dt(r[0]))
        ema_data[name]["value"].append(float(r[2]))
    return ema_data


def find_exit_ts(bars, bar_ts_index, candidate_ts, exit_bars):
    """Find the exit bar timestamp given candidate ts and exit_bars count."""
    if exit_bars is None:
        return None
    cand_idx = bar_ts_index.get(candidate_ts)
    if cand_idx is None:
        return None
    entry_idx = cand_idx + 1  # entry bar is AFTER candidate bar
    exit_idx = min(entry_idx + exit_bars - 1, len(bars) - 1)
    return bars[exit_idx]["ts"]


EMA_STYLES = {
    "ema10": ("#00e5ff", 1), "ema15": ("#40c4ff", 1),
    "ema20": ("#80d8ff", 1), "ema25": ("#ffab40", 1),
    "ema30": ("#ff6d00", 2),
}


def build_hover(c):
    lines = [
        f"<b>{c['direction'].upper()}</b> score={c['score']:.2f}",
        f"level={c['level_price']:.2f}  age={c['level_age_min']}min",
    ]
    if c["mfe"] is not None:
        lines.append(f"MFE={c['mfe']:.2f}  MAE={c['mae']:.2f}")
        lines.append(f"EOD={c['return_atr_eod']:.2f}")
    else:
        lines.append("no outcome data")
    return "<br>".join(lines)


def build_figure(bars, candidates, emas, symbol, timeframe, trading_date):
    """Build Plotly figure from pre-queried data. Timestamps converted to MT."""
    if not bars:
        return go.Figure().update_layout(
            title=f"{symbol} {timeframe} — {trading_date} (no data)",
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="#e0e0e0"),
        )

    tf_min = TIMEFRAME_MINUTES.get(timeframe, 5)

    # Convert all timestamps to Mountain Time
    for b in bars:
        b["ts"] = to_mt(b["ts"])
    for c in candidates:
        c["ts"] = to_mt(c["ts"])
    for name in emas:
        emas[name]["ts"] = [to_mt(t) for t in emas[name]["ts"]]

    # Build bar timestamp index
    bar_ts_list = [b["ts"] for b in bars]
    bar_ts_index = {ts: i for i, ts in enumerate(bar_ts_list)}

    # ATR for marker offset
    avg_range = sum(b["high"] - b["low"] for b in bars) / len(bars)
    offset = avg_range * 0.5

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.8, 0.2], vertical_spacing=0.03,
    )

    # Candlesticks by session
    session_styles = {
        "regular": ("#26a69a", "#ef5350", 1.0, "Regular"),
        "pre":     ("#4fc3f7", "#90a4ae", 0.6, "Pre-market"),
        "post":    ("#ce93d8", "#90a4ae", 0.6, "Post-market"),
    }
    for session_key, (up_color, down_color, opacity, label) in session_styles.items():
        session_bars = [b for b in bars if b["session"] == session_key]
        if not session_bars:
            continue
        fig.add_trace(go.Candlestick(
            x=[b["ts"] for b in session_bars],
            open=[b["open"] for b in session_bars],
            high=[b["high"] for b in session_bars],
            low=[b["low"] for b in session_bars],
            close=[b["close"] for b in session_bars],
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            opacity=opacity,
            name=label,
        ), row=1, col=1)

    # Volume
    vol_colors = ["rgba(38,166,154,0.5)" if b["close"] >= b["open"] else "rgba(239,83,80,0.5)" for b in bars]
    fig.add_trace(go.Bar(
        x=bar_ts_list,
        y=[b["volume"] for b in bars],
        marker_color=vol_colors,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    # EMA ribbon
    for name, (color, width) in EMA_STYLES.items():
        if name in emas:
            fig.add_trace(go.Scatter(
                x=emas[name]["ts"], y=emas[name]["value"],
                mode="lines", line=dict(color=color, width=width),
                name=name.upper(), showlegend=True,
            ), row=1, col=1)

    # Candidates: markers + level lines + exit markers
    for c in candidates:
        color = marker_color(c["mfe"], c["mae"])
        is_long = c["direction"] == "long"

        # Find the bar at candidate ts
        bar_idx = bar_ts_index.get(c["ts"])
        if bar_idx is None:
            continue
        bar = bars[bar_idx]

        # Marker position
        marker_y = bar["low"] - offset if is_long else bar["high"] + offset
        marker_symbol = "triangle-up" if is_long else "triangle-down"

        fig.add_trace(go.Scatter(
            x=[c["ts"]],
            y=[marker_y],
            mode="markers",
            marker=dict(symbol=marker_symbol, size=14, color=color, line=dict(width=1, color="#ffffff")),
            hovertext=build_hover(c),
            hoverinfo="text",
            showlegend=False,
        ), row=1, col=1)

        # Level line: from level_age_min bars back to candidate ts
        lookback_bars = max(1, c["level_age_min"] // tf_min)
        line_start_idx = max(0, bar_idx - lookback_bars)
        line_start_ts = bar_ts_list[line_start_idx]

        fig.add_trace(go.Scatter(
            x=[line_start_ts, c["ts"]],
            y=[c["level_price"], c["level_price"]],
            mode="lines",
            line=dict(color=color, width=1.5, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        # Exit markers
        exit_defs = [
            ("Pivot", c.get("exit_bars_pivot"), c.get("return_atr_pivot"),
             "diamond", "#ab47bc", 10),
            ("EMA30", c.get("exit_bars_ema30"), c.get("return_atr_ema30"),
             "square", "#ff6d00", 10),
            ("EOD", len(bars) - (bar_idx + 1), c.get("return_atr_eod"),
             "x", "#ffffff", 8),
        ]
        for exit_type, exit_bars, return_val, sym, ecolor, esize in exit_defs:
            exit_ts = find_exit_ts(bars, bar_ts_index, c["ts"], exit_bars)
            if exit_ts is None:
                continue
            exit_bar_idx = bar_ts_index.get(exit_ts)
            if exit_bar_idx is None:
                continue
            exit_bar = bars[exit_bar_idx]
            ey = exit_bar["high"] + offset if is_long else exit_bar["low"] - offset
            ret_str = f"{return_val:+.2f} ATR" if return_val is not None else "n/a"
            fig.add_trace(go.Scatter(
                x=[exit_ts], y=[ey],
                mode="markers",
                marker=dict(symbol=sym, size=esize, color=ecolor,
                            line=dict(width=1, color="#ffffff")),
                hovertext=f"{exit_type} exit: {ret_str}",
                hoverinfo="text",
                showlegend=False,
            ), row=1, col=1)

    # Layout
    fig.update_layout(
        title=f"{symbol} {timeframe} — {trading_date}",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=60, r=30, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="#2a2a4a")
    fig.update_yaxes(gridcolor="#2a2a4a")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Chart a single day with breakout candidates")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", required=True)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    trading_date = date.fromisoformat(args.date)
    timeframe = args.timeframe

    ch = get_ch_client()
    bars = query_bars(ch, symbol, timeframe, trading_date)
    candidates = query_candidates(ch, symbol, timeframe, trading_date)
    emas = query_emas(ch, symbol, timeframe, trading_date)

    print(f"{symbol} {timeframe} {trading_date}: {len(bars)} bars, {len(candidates)} candidates")

    fig = build_figure(bars, candidates, emas, symbol, timeframe, trading_date)
    fig.show()


if __name__ == "__main__":
    main()
