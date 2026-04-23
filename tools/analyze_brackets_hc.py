"""
Analyze bracket outcomes on HC predictions per timeframe.
"""
import pandas as pd
from datetime import date
from ml.shared.clickhouse import get_ch_client
from ml.shared.utils import utc_dt

ch = get_ch_client()

for tf in ['1m', '5m', '10m', '15m']:
    preds = pd.read_csv(f'data/reports/breakout/{tf}/predictions.csv')
    preds['ts'] = pd.to_datetime(preds['ts'], utc=True)

    result = ch.query(
        "SELECT bc.symbol, bc.ts, bc.timeframe, "
        "       bo.bracket_exit_type, bo.bracket_return_atr "
        "FROM trading.breakout_candidate bc FINAL "
        "JOIN trading.breakout_outcome bo FINAL "
        "    ON bc.symbol = bo.symbol "
        "    AND bc.timeframe = bo.timeframe "
        "    AND bc.ts = bo.ts "
        "WHERE bc.timeframe = %(tf)s",
        parameters={"tf": tf}
    )
    outcomes = pd.DataFrame(result.result_rows, columns=[
        'symbol', 'ts', 'timeframe', 'bracket_exit_type', 'bracket_return_atr'
    ])
    outcomes['ts'] = outcomes['ts'].apply(utc_dt)

    merged = preds.merge(outcomes, on=['symbol', 'ts', 'timeframe'])
    hc = merged[merged['is_hc'] == 1]
    total_hc = len(hc)

    print(f"\n=== {tf} ===")
    print(f"Total HC predictions: {total_hc}")
    print(f"Avg bracket return: {hc['bracket_return_atr'].mean():.3f} ATR/trade")
    print()

    dist = hc.groupby('bracket_exit_type').agg(
        count=('bracket_exit_type', 'size'),
        avg_return=('bracket_return_atr', 'mean')
    ).reset_index()
    dist['pct'] = (dist['count'] / total_hc * 100).round(1)
    print(dist.to_string(index=False))
