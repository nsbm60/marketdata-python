#!/bin/bash
set -e
CONFIGS="config/models/1m.yaml config/models/5m.yaml config/models/10m.yaml config/models/15m.yaml config/models/20m.yaml"

for CONFIG in $CONFIGS; do
    TF=$(grep timeframe $CONFIG | awk '{print $2}')
    echo "=== Processing $TF ==="

    PYTHONPATH=. python tools/label_outcomes.py --config $CONFIG --force

    PYTHONPATH=. python tools/build_breakout_features.py --config $CONFIG

    PYTHONPATH=. python -m ml.models.breakout.validate \
        --features data/features/breakout/features_${TF}.csv \
        --config $CONFIG

    PYTHONPATH=. python -m ml.models.breakout.feature_importance \
        --features data/features/breakout/features_${TF}.csv \
        --config $CONFIG
done

echo "=== All timeframes complete ==="
