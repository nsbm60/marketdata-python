"""
ml/shared/features.py

Feature list for session direction models.
"""

# Features available by 10:30am on the session date -- no lookahead
FEATURES = [
    # Gap / prior session context (known at 9:30)
    "gap_pct",
    "prior_day_range_pct",
    "atr20",

    # Window price structure (known at 9:30 + window minutes)
    "w_range_pct",
    "w_range_atr",
    "w_vwap_dev",

    # First-15-minute sub-features
    "f15_range_ratio",
    "f15_vol_ratio",

    # Volume
    "w_vol_ratio",

    # Sweep signal
    "sweep_signal",
    "sweep_direction",

    # Confirmation signal features
    "reversal_progress",
    "close_position",

    # Regime features (rolling, lagged -- no lookahead)
    "rolling_reversal_rate",
    "rolling_high_set_rate",
    "directional_bias",
    "gap_regime_alignment",

    # Correlation / market regime features (rolling 60-session, lagged)
    # Measure how macro-driven vs idiosyncratic the current environment is.
    # corr_regime_dev excluded -- too many nulls in early history.
    "target_qqq_corr",
    "target_smh_corr",
    "target_qqq_beta",
]

# Columns deliberately excluded and why:
#   fh_high_is_session_high / fh_low_is_session_low -- used to compute the label
#   fh_range_abs  -- redundant with fh_range_pct and fh_range_atr
#   avg_vol_20    -- raw volume number, already normalized into w_vol_ratio
#   corr_regime_dev -- 75 nulls from extended lookback; add back when dataset grows
#   date          -- not a feature
#   label / label_name -- target
