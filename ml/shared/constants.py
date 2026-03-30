"""
ml/shared/constants.py

Shared constants used across ML modules.
"""

from zoneinfo import ZoneInfo

# New York timezone for market hours
NY = ZoneInfo("America/New_York")

# Valid prediction windows in minutes
VALID_WINDOWS = [15, 30, 45, 60]

# Session label names (4-class)
LABEL_NAMES = {
    0: "trend",
    1: "containment",
    2: "reversal",
    3: "double_sweep",
}

# Binary label names
BINARY_NAMES = {
    0: "non_reversal",
    1: "reversal",
}

# Directional label names (for reversal sessions only)
DIRECTIONAL_NAMES = {
    0: "buy_the_dip",    # first-hour LOW is session low -- stock dropped early, recovered
    1: "fade_the_high",  # first-hour HIGH is session high -- stock ran up early, faded
}

# Column names for labels
LABEL_COL = "label"
BINARY_LABEL_COL = "binary_label"
DIRECTIONAL_LABEL_COL = "directional_label"
