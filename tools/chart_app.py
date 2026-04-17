"""
tools/chart_app.py

Dash app for browsing breakout candidates by symbol/date/timeframe.

Usage:
    PYTHONPATH=. python tools/chart_app.py --port 8050
"""

import argparse
from datetime import date

from dash import Dash, dcc, html, Input, Output, State, ctx

from ml.shared.clickhouse import get_ch_client
from ml.shared.config import fetch_symbol_list
from discovery.service_locator import ServiceLocator
from tools.chart_day import (
    query_bars, query_candidates, query_emas, build_figure,
    TIMEFRAME_MINUTES,
)

# Shared ClickHouse client
ch = get_ch_client()


def get_symbols():
    """Fetch trading universe symbols from MDS, fallback to breakout_candidate."""
    try:
        md = ServiceLocator.wait_for_service(ServiceLocator.MARKET_DATA, timeout_sec=5)
        return fetch_symbol_list(md.router, "trading_universe")
    except Exception:
        result = ch.query(
            "SELECT DISTINCT symbol FROM trading.breakout_candidate FINAL "
            "ORDER BY symbol"
        )
        return [r[0] for r in result.result_rows]


def get_dates_for_symbol(symbol):
    """Fetch trading dates with candidates for a symbol, newest first."""
    result = ch.query(
        "SELECT DISTINCT session FROM trading.breakout_candidate FINAL "
        "WHERE symbol = %(symbol)s "
        "ORDER BY session DESC "
        "LIMIT 90",
        parameters={"symbol": symbol},
    )
    return [r[0] if isinstance(r[0], date) else date.fromisoformat(str(r[0]))
            for r in result.result_rows]


app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#16213e", "minHeight": "100vh",
           "fontFamily": "Inter, sans-serif", "color": "#e0e0e0"},
    children=[
        # Controls bar
        html.Div(
            style={"display": "flex", "gap": "12px", "padding": "12px 16px",
                   "alignItems": "center", "borderBottom": "1px solid #2a2a4a"},
            children=[
                html.Label("Symbol", style={"fontSize": 12}),
                dcc.Dropdown(
                    id="symbol-dropdown",
                    options=[],
                    value=None,
                    style={"width": 120, "color": "#000"},
                ),
                html.Label("Timeframe", style={"fontSize": 12}),
                dcc.Dropdown(
                    id="timeframe-dropdown",
                    options=[{"label": tf, "value": tf}
                             for tf in TIMEFRAME_MINUTES.keys()],
                    value="5m",
                    style={"width": 100, "color": "#000"},
                ),
                html.Label("Date", style={"fontSize": 12}),
                html.Button("\u25c0", id="prev-date", n_clicks=0,
                    style={"padding": "4px 10px", "cursor": "pointer",
                           "backgroundColor": "#2a2a4a", "border": "1px solid #3a3a5a",
                           "color": "#e0e0e0", "borderRadius": 4, "fontSize": 14}),
                dcc.Dropdown(
                    id="date-dropdown",
                    options=[],
                    value=None,
                    style={"width": 160, "color": "#000"},
                ),
                html.Button("\u25b6", id="next-date", n_clicks=0,
                    style={"padding": "4px 10px", "cursor": "pointer",
                           "backgroundColor": "#2a2a4a", "border": "1px solid #3a3a5a",
                           "color": "#e0e0e0", "borderRadius": 4, "fontSize": 14}),
                html.Span(id="status", style={"fontSize": 12, "marginLeft": "auto"}),
            ],
        ),
        # Chart
        dcc.Graph(
            id="chart",
            style={"height": "calc(100vh - 60px)"},
            config={"displayModeBar": True, "scrollZoom": True},
        ),
    ],
)


@app.callback(
    Output("symbol-dropdown", "options"),
    Output("symbol-dropdown", "value"),
    Input("symbol-dropdown", "id"),
)
def populate_symbols(_):
    symbols = get_symbols()
    options = [{"label": s, "value": s} for s in symbols]
    default = symbols[0] if symbols else None
    return options, default


@app.callback(
    Output("date-dropdown", "options"),
    Output("date-dropdown", "value"),
    Input("symbol-dropdown", "value"),
)
def populate_dates(symbol):
    if not symbol:
        return [], None
    dates = get_dates_for_symbol(symbol)
    options = [{"label": d.isoformat(), "value": d.isoformat()} for d in dates]
    default = dates[0].isoformat() if dates else None
    return options, default


@app.callback(
    Output("date-dropdown", "value", allow_duplicate=True),
    Input("prev-date", "n_clicks"),
    Input("next-date", "n_clicks"),
    State("date-dropdown", "value"),
    State("date-dropdown", "options"),
    prevent_initial_call=True,
)
def navigate_date(prev_clicks, next_clicks, current_date, options):
    if not current_date or not options:
        return current_date
    dates = [o["value"] for o in options]  # newest first
    try:
        idx = dates.index(current_date)
    except ValueError:
        return current_date
    if ctx.triggered_id == "prev-date":
        idx = min(idx + 1, len(dates) - 1)  # older = higher index (newest first)
    elif ctx.triggered_id == "next-date":
        idx = max(idx - 1, 0)  # newer = lower index
    return dates[idx]


@app.callback(
    Output("chart", "figure"),
    Output("status", "children"),
    Input("symbol-dropdown", "value"),
    Input("timeframe-dropdown", "value"),
    Input("date-dropdown", "value"),
)
def update_chart(symbol, timeframe, date_str):
    if not symbol or not timeframe or not date_str:
        empty = build_figure([], [], {}, "", "", "")
        return empty, ""

    trading_date = date.fromisoformat(date_str)
    bars = query_bars(ch, symbol, timeframe, trading_date)
    candidates = query_candidates(ch, symbol, timeframe, trading_date)
    emas = query_emas(ch, symbol, timeframe, trading_date)

    fig = build_figure(bars, candidates, emas, symbol, timeframe, trading_date)
    fig.update_layout(height=None)  # let CSS control height

    status = f"{len(bars)} bars, {len(candidates)} candidates"
    return fig, status


def main():
    parser = argparse.ArgumentParser(description="Breakout candidate chart browser")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
