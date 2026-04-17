#!/bin/bash
set -e
TIMEFRAMES="1m 5m 10m 15m 20m 30m 60m"

for TF in $TIMEFRAMES; do
    echo "=== Processing $TF ==="

    PYTHONPATH=. python tools/build_breakout_features.py \
        --timeframe $TF

    PYTHONPATH=. python -m ml.models.breakout.validate \
        --features data/features/breakout/features_${TF}.csv \
        --timeframe $TF \
        --fold-size 50

    PYTHONPATH=. python -m ml.models.breakout.feature_importance \
        --features data/features/breakout/features_${TF}.csv \
        --timeframe $TF
done

echo "=== All timeframes complete ==="
