"""
explore.py

Exploratory analysis for model 2 (level_hold) parameter determination.

Analyzes historical data to determine:
1. Band width -- how close price needs to get to first-hour extreme
2. N bars for label -- how many bars to observe after touch

Usage:
    python ml/models/level_hold/explore.py --symbols NVDA AMD
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ml.shared.clickhouse import get_ch_client
from ml.shared.paths import features_path, report_path, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# SQL to get intraday bars after 10:30am for a session
INTRADAY_SQL = """
SELECT ts, open, high, low, close, volume
FROM stock_bar FINAL
WHERE symbol       = %(symbol)s
  AND period       = '1m'
  AND session      = 1
  AND toDate(toTimezone(ts, 'America/New_York')) = %(session_date)s
  AND (toHour(ts) > 10 OR (toHour(ts) = 10 AND toMinute(ts) >= 30))
ORDER BY ts
"""

# SQL to get actual labels for model 1 correctness filtering
LABELS_SQL = """
SELECT session_date, directional_label
FROM session_labels
WHERE symbol         = %(symbol)s
  AND window_minutes = 60
  AND directional_label IS NOT NULL
ORDER BY session_date
"""


def load_model1_predictions(symbol: str, window: int = 60) -> pd.DataFrame:
    """Load model 1 predictions from session_direction CSV."""
    path = features_path(symbol, window, prefix="session_direction")
    if not path.exists():
        log.error(f"Model 1 predictions not found: {path}")
        log.error("Run: python ml/models/session_direction/score_historical.py --symbols {symbol}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["session_date"])
    return df


def load_actual_labels(ch_client, symbol: str) -> pd.DataFrame:
    """Load actual directional labels from ClickHouse."""
    result = ch_client.query(LABELS_SQL, parameters={"symbol": symbol})
    df = pd.DataFrame(result.result_rows, columns=["session_date", "actual_label"])
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df


def get_correct_sessions(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Join predictions with actuals and filter to sessions where model 1 was correct.
    Returns merged DataFrame with both predicted and actual labels.
    """
    predictions["session_date"] = pd.to_datetime(predictions["session_date"])
    merged = predictions.merge(actuals, on="session_date", how="inner")
    correct = merged[merged["label"] == merged["actual_label"]].copy()
    return correct


def fetch_intraday_bars(ch_client, symbol: str, session_date) -> pd.DataFrame:
    """Fetch intraday bars after 10:30am for a single session."""
    if isinstance(session_date, pd.Timestamp):
        session_date = session_date.date()
    result = ch_client.query(INTRADAY_SQL, parameters={
        "symbol": symbol,
        "session_date": session_date.isoformat(),
    })
    if not result.result_rows:
        return pd.DataFrame()
    df = pd.DataFrame(result.result_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def analyze_distance_distribution(ch_client, symbol: str, correct_sessions: pd.DataFrame) -> dict:
    """
    Analysis 1: Distance to extreme distribution.

    For every correct session, find the closest price got to the predicted extreme.
    """
    distances_fade = []  # fade_the_high sessions
    distances_dip = []   # buy_the_dip sessions

    for _, row in correct_sessions.iterrows():
        bars = fetch_intraday_bars(ch_client, symbol, row["session_date"])
        if bars.empty:
            continue

        fh_high = row["first_hour_high"]
        fh_low = row["first_hour_low"]
        fh_range = row["first_hour_range"]

        if fh_range <= 0:
            continue

        if row["label"] == 1:  # fade_the_high
            # Distance = how close bar_high got to first_hour_high
            # Positive = didn't reach, Negative = breached
            closest = (bars["high"].max() - fh_high) / fh_range
            distances_fade.append(closest)
        else:  # buy_the_dip
            # Distance = how close bar_low got to first_hour_low
            closest = (fh_low - bars["low"].min()) / fh_range
            distances_dip.append(closest)

    return {
        "fade_the_high": np.array(distances_fade),
        "buy_the_dip": np.array(distances_dip),
    }


def analyze_time_to_resolution(ch_client, symbol: str, correct_sessions: pd.DataFrame,
                                band_pct: float = 0.15) -> dict:
    """
    Analysis 2: Time to resolution.

    For sessions where price came within band_pct of the extreme,
    measure how many bars until outcome becomes clear.
    """
    hold_resolution_times = []
    break_resolution_times = []

    for _, row in correct_sessions.iterrows():
        bars = fetch_intraday_bars(ch_client, symbol, row["session_date"])
        if bars.empty or len(bars) < 5:
            continue

        fh_high = row["first_hour_high"]
        fh_low = row["first_hour_low"]
        fh_range = row["first_hour_range"]

        if fh_range <= 0:
            continue

        band_threshold = fh_range * band_pct
        decisive_threshold = fh_range * 0.05

        if row["label"] == 1:  # fade_the_high
            extreme = fh_high
            # Find bars where price came within band of the high
            for i, bar in bars.iterrows():
                if bar["high"] >= extreme - band_threshold:
                    # Found a touch - now count bars to resolution
                    touch_idx = bars.index.get_loc(i)
                    subsequent = bars.iloc[touch_idx:]

                    # Did it break (go above extreme)?
                    broke = (subsequent["high"] > extreme).any()

                    # Find resolution - price moves decisively away
                    for j in range(1, len(subsequent)):
                        distance_from_extreme = extreme - subsequent.iloc[j]["high"]
                        if distance_from_extreme > decisive_threshold:
                            if broke:
                                break_resolution_times.append(j)
                            else:
                                hold_resolution_times.append(j)
                            break
                    break  # Only analyze first touch per session
        else:  # buy_the_dip
            extreme = fh_low
            for i, bar in bars.iterrows():
                if bar["low"] <= extreme + band_threshold:
                    touch_idx = bars.index.get_loc(i)
                    subsequent = bars.iloc[touch_idx:]

                    broke = (subsequent["low"] < extreme).any()

                    for j in range(1, len(subsequent)):
                        distance_from_extreme = subsequent.iloc[j]["low"] - extreme
                        if distance_from_extreme > decisive_threshold:
                            if broke:
                                break_resolution_times.append(j)
                            else:
                                hold_resolution_times.append(j)
                            break
                    break

    return {
        "hold": np.array(hold_resolution_times),
        "break": np.array(break_resolution_times),
    }


def analyze_approach_profile(ch_client, symbol: str, correct_sessions: pd.DataFrame,
                              band_pct: float = 0.15) -> dict:
    """
    Analysis 3: Candle approach profile.

    Characterize the 10 bars immediately before the touch.
    """
    hold_body_ratios = []
    break_body_ratios = []
    hold_colors = []
    break_colors = []
    hold_momentum = []
    break_momentum = []

    n_bars = 10

    for _, row in correct_sessions.iterrows():
        bars = fetch_intraday_bars(ch_client, symbol, row["session_date"])
        if bars.empty or len(bars) < n_bars + 5:
            continue

        fh_high = row["first_hour_high"]
        fh_low = row["first_hour_low"]
        fh_range = row["first_hour_range"]

        if fh_range <= 0:
            continue

        band_threshold = fh_range * band_pct

        if row["label"] == 1:  # fade_the_high
            extreme = fh_high
            for i, bar in bars.iterrows():
                if bar["high"] >= extreme - band_threshold:
                    touch_idx = bars.index.get_loc(i)
                    if touch_idx < n_bars:
                        break  # Not enough bars before touch

                    approach_bars = bars.iloc[touch_idx - n_bars:touch_idx]

                    # Body ratio per bar
                    body = abs(approach_bars["close"] - approach_bars["open"])
                    bar_range = approach_bars["high"] - approach_bars["low"]
                    body_ratio = (body / bar_range.replace(0, np.nan)).fillna(0).values

                    # Color: 1=green, 0=red
                    colors = (approach_bars["close"] > approach_bars["open"]).astype(int).values

                    # Momentum: price change over approach
                    momentum = (approach_bars["close"].iloc[-1] - approach_bars["close"].iloc[0]) / fh_range

                    # Did it break?
                    subsequent = bars.iloc[touch_idx:]
                    broke = (subsequent["high"] > extreme).any()

                    if broke:
                        break_body_ratios.append(body_ratio)
                        break_colors.append(colors)
                        break_momentum.append(momentum)
                    else:
                        hold_body_ratios.append(body_ratio)
                        hold_colors.append(colors)
                        hold_momentum.append(momentum)
                    break
        else:  # buy_the_dip
            extreme = fh_low
            for i, bar in bars.iterrows():
                if bar["low"] <= extreme + band_threshold:
                    touch_idx = bars.index.get_loc(i)
                    if touch_idx < n_bars:
                        break

                    approach_bars = bars.iloc[touch_idx - n_bars:touch_idx]

                    body = abs(approach_bars["close"] - approach_bars["open"])
                    bar_range = approach_bars["high"] - approach_bars["low"]
                    body_ratio = (body / bar_range.replace(0, np.nan)).fillna(0).values

                    colors = (approach_bars["close"] > approach_bars["open"]).astype(int).values

                    momentum = (approach_bars["close"].iloc[-1] - approach_bars["close"].iloc[0]) / fh_range

                    subsequent = bars.iloc[touch_idx:]
                    broke = (subsequent["low"] < extreme).any()

                    if broke:
                        break_body_ratios.append(body_ratio)
                        break_colors.append(colors)
                        break_momentum.append(momentum)
                    else:
                        hold_body_ratios.append(body_ratio)
                        hold_colors.append(colors)
                        hold_momentum.append(momentum)
                    break

    return {
        "hold_body_ratios": np.array(hold_body_ratios) if hold_body_ratios else np.array([]).reshape(0, n_bars),
        "break_body_ratios": np.array(break_body_ratios) if break_body_ratios else np.array([]).reshape(0, n_bars),
        "hold_colors": np.array(hold_colors) if hold_colors else np.array([]).reshape(0, n_bars),
        "break_colors": np.array(break_colors) if break_colors else np.array([]).reshape(0, n_bars),
        "hold_momentum": np.array(hold_momentum),
        "break_momentum": np.array(break_momentum),
    }


def analyze_revisit_frequency(ch_client, symbol: str, correct_sessions: pd.DataFrame,
                               band_pct: float = 0.15) -> dict:
    """
    Analysis 4: Revisit frequency.

    How often does price revisit the first-hour extreme after 10:30am?
    """
    revisit_counts = []
    first_revisit_times = []  # minutes after 10:30am

    for _, row in correct_sessions.iterrows():
        bars = fetch_intraday_bars(ch_client, symbol, row["session_date"])
        if bars.empty:
            revisit_counts.append(0)
            continue

        fh_high = row["first_hour_high"]
        fh_low = row["first_hour_low"]
        fh_range = row["first_hour_range"]

        if fh_range <= 0:
            revisit_counts.append(0)
            continue

        band_threshold = fh_range * band_pct
        count = 0
        first_time = None
        in_zone = False

        if row["label"] == 1:  # fade_the_high
            extreme = fh_high
            for _, bar in bars.iterrows():
                near_extreme = bar["high"] >= extreme - band_threshold
                if near_extreme and not in_zone:
                    count += 1
                    if first_time is None:
                        # Minutes since 10:30am
                        first_time = (bar["ts"].hour - 10) * 60 + bar["ts"].minute - 30
                    in_zone = True
                elif not near_extreme:
                    in_zone = False
        else:  # buy_the_dip
            extreme = fh_low
            for _, bar in bars.iterrows():
                near_extreme = bar["low"] <= extreme + band_threshold
                if near_extreme and not in_zone:
                    count += 1
                    if first_time is None:
                        first_time = (bar["ts"].hour - 10) * 60 + bar["ts"].minute - 30
                    in_zone = True
                elif not near_extreme:
                    in_zone = False

        revisit_counts.append(count)
        if first_time is not None:
            first_revisit_times.append(first_time)

    return {
        "counts": np.array(revisit_counts),
        "first_times": np.array(first_revisit_times),
    }


def plot_distance_distribution(distances: dict, symbol: str):
    """Plot distance distribution histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, data) in zip(axes, distances.items()):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{label} (n=0)")
            continue

        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Extreme level")
        ax.axvline(-0.05, color="orange", linestyle=":", alpha=0.7, label="-5%")
        ax.axvline(-0.10, color="orange", linestyle=":", alpha=0.7, label="-10%")
        ax.axvline(-0.15, color="orange", linestyle=":", alpha=0.7, label="-15%")
        ax.set_xlabel("Distance (% of first-hour range)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{label} (n={len(data)})")
        ax.legend(fontsize=8)

    fig.suptitle(f"{symbol} - Distance to Extreme Distribution", fontsize=14)
    plt.tight_layout()

    ensure_dirs(symbol)
    out_path = report_path(symbol, "distance_distribution", "png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_time_to_resolution(resolution: dict, symbol: str):
    """Plot time to resolution histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (outcome, data) in zip(axes, resolution.items()):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{outcome.upper()} outcomes (n=0)")
            continue

        ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.percentile(data, 80), color="red", linestyle="--",
                   label=f"80th pctl: {np.percentile(data, 80):.0f} bars")
        ax.axvline(np.percentile(data, 90), color="orange", linestyle="--",
                   label=f"90th pctl: {np.percentile(data, 90):.0f} bars")
        ax.set_xlabel("Bars to resolution")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{outcome.upper()} outcomes (n={len(data)})")
        ax.legend(fontsize=8)

    fig.suptitle(f"{symbol} - Time to Resolution", fontsize=14)
    plt.tight_layout()

    out_path = report_path(symbol, "time_to_resolution", "png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_approach_profiles(profiles: dict, symbol: str):
    """Plot approach profile charts."""
    n_bars = 10
    x = range(1, n_bars + 1)

    # Body ratio profile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if profiles["hold_body_ratios"].size > 0:
        hold_mean = profiles["hold_body_ratios"].mean(axis=0)
        hold_std = profiles["hold_body_ratios"].std(axis=0)
        axes[0].plot(x, hold_mean, "g-", linewidth=2, label="Hold")
        axes[0].fill_between(x, hold_mean - hold_std, hold_mean + hold_std, alpha=0.2, color="green")

    if profiles["break_body_ratios"].size > 0:
        break_mean = profiles["break_body_ratios"].mean(axis=0)
        break_std = profiles["break_body_ratios"].std(axis=0)
        axes[0].plot(x, break_mean, "r-", linewidth=2, label="Break")
        axes[0].fill_between(x, break_mean - break_std, break_mean + break_std, alpha=0.2, color="red")

    axes[0].set_xlabel("Bars before touch")
    axes[0].set_ylabel("Body ratio (body/range)")
    axes[0].set_title("Body Ratio Profile")
    axes[0].legend()
    axes[0].set_xticks(x)

    # Color sequence profile
    if profiles["hold_colors"].size > 0:
        hold_green_pct = profiles["hold_colors"].mean(axis=0) * 100
        axes[1].plot(x, hold_green_pct, "g-", linewidth=2, label="Hold")

    if profiles["break_colors"].size > 0:
        break_green_pct = profiles["break_colors"].mean(axis=0) * 100
        axes[1].plot(x, break_green_pct, "r-", linewidth=2, label="Break")

    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Bars before touch")
    axes[1].set_ylabel("% Green candles")
    axes[1].set_title("Candle Color Profile")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 100)

    fig.suptitle(f"{symbol} - Approach Profiles", fontsize=14)
    plt.tight_layout()

    out_path = report_path(symbol, "approach_profile", "png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def plot_revisit_frequency(revisits: dict, symbol: str):
    """Plot revisit frequency charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = revisits["counts"]
    first_times = revisits["first_times"]

    # Revisit count distribution
    if len(counts) > 0:
        count_bins = [0, 1, 2, 3, 4, 100]
        count_labels = ["0", "1", "2", "3", "4+"]
        hist, _ = np.histogram(counts, bins=count_bins)
        axes[0].bar(count_labels, hist, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Number of revisits")
        axes[0].set_ylabel("Sessions")
        axes[0].set_title(f"Revisit Count Distribution (n={len(counts)})")

        for i, v in enumerate(hist):
            pct = 100 * v / len(counts)
            axes[0].text(i, v + 1, f"{pct:.1f}%", ha="center", fontsize=9)

    # Time of first revisit
    if len(first_times) > 0:
        axes[1].hist(first_times, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(np.median(first_times), color="red", linestyle="--",
                        label=f"Median: {np.median(first_times):.0f} min")
        axes[1].set_xlabel("Minutes after 10:30am")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Time to First Revisit (n={len(first_times)})")
        axes[1].legend()

    fig.suptitle(f"{symbol} - Revisit Frequency", fontsize=14)
    plt.tight_layout()

    out_path = report_path(symbol, "revisit_frequency", "png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved: {out_path}")


def print_summary(symbol: str, distances: dict, resolution: dict,
                  profiles: dict, revisits: dict):
    """Print analysis summary."""
    print()
    print("=" * 60)
    print(f"EXPLORATORY ANALYSIS SUMMARY: {symbol}")
    print("=" * 60)

    # Distance analysis
    all_distances = np.concatenate([distances["fade_the_high"], distances["buy_the_dip"]])
    if len(all_distances) > 0:
        print("\nDistance to extreme (% of correct sessions):")
        for thresh in [0.05, 0.10, 0.15, 0.20]:
            pct = 100 * (all_distances >= -thresh).mean()
            print(f"  Within {thresh*100:4.0f}% of range:  {pct:5.1f}%")

    # Resolution time
    print("\nResolution time:")
    for outcome in ["hold", "break"]:
        data = resolution[outcome]
        if len(data) > 0:
            p80 = np.percentile(data, 80)
            p90 = np.percentile(data, 90)
            print(f"  {outcome.upper():5} -- 80th percentile: {p80:2.0f} bars")
            print(f"  {outcome.upper():5} -- 90th percentile: {p90:2.0f} bars")
        else:
            print(f"  {outcome.upper():5} -- no data")

    # Revisit frequency
    counts = revisits["counts"]
    if len(counts) > 0:
        print("\nRevisit frequency:")
        print(f"  0 revisits:  {100*(counts == 0).mean():5.1f}% of sessions")
        print(f"  1 revisit:   {100*(counts == 1).mean():5.1f}% of sessions")
        print(f"  2+ revisits: {100*(counts >= 2).mean():5.1f}% of sessions")
        print(f"  Avg revisits per session: {counts.mean():.2f}")

    # Approach profile
    print("\nApproach profile:")
    if profiles["hold_momentum"].size > 0:
        print(f"  Hold sessions:  avg final body ratio = {profiles['hold_body_ratios'][:, -1].mean():.2f}, "
              f"avg momentum = {profiles['hold_momentum'].mean():.3f}")
    if profiles["break_momentum"].size > 0:
        print(f"  Break sessions: avg final body ratio = {profiles['break_body_ratios'][:, -1].mean():.2f}, "
              f"avg momentum = {profiles['break_momentum'].mean():.3f}")

    # Suggested parameters
    print("\nSUGGESTED PARAMETERS (confirm by reviewing plots):")
    # Band width: use threshold where ~50-70% of sessions have revisits
    if len(counts) > 0:
        revisit_rate = (counts > 0).mean()
        if revisit_rate > 0.5:
            print(f"  Band width:        15% of first_hour_range")
        else:
            print(f"  Band width:        20% of first_hour_range (low revisit rate: {revisit_rate:.1%})")

    # N bars: use 80th percentile of resolution times
    all_resolution = np.concatenate([resolution["hold"], resolution["break"]])
    if len(all_resolution) > 0:
        suggested_n = int(np.percentile(all_resolution, 80))
        print(f"  N bars for label:  {suggested_n} bars")

    print("=" * 60)


def run_analysis(symbol: str):
    """Run full exploratory analysis for a symbol."""
    log.info(f"Starting analysis for {symbol}")

    # Load data
    predictions = load_model1_predictions(symbol)
    if predictions.empty:
        return

    log.info(f"Loaded {len(predictions)} model 1 predictions")

    ch_client = get_ch_client()
    actuals = load_actual_labels(ch_client, symbol)
    log.info(f"Loaded {len(actuals)} actual labels")

    correct = get_correct_sessions(predictions, actuals)
    log.info(f"Found {len(correct)} correct model 1 sessions")

    if len(correct) < 50:
        log.warning(f"Only {len(correct)} correct sessions -- results may be unreliable")

    # Run analyses
    log.info("Analysis 1: Distance distribution...")
    distances = analyze_distance_distribution(ch_client, symbol, correct)

    log.info("Analysis 2: Time to resolution...")
    resolution = analyze_time_to_resolution(ch_client, symbol, correct)

    log.info("Analysis 3: Approach profiles...")
    profiles = analyze_approach_profile(ch_client, symbol, correct)

    log.info("Analysis 4: Revisit frequency...")
    revisits = analyze_revisit_frequency(ch_client, symbol, correct)

    # Generate plots
    log.info("Generating plots...")
    plot_distance_distribution(distances, symbol)
    plot_time_to_resolution(resolution, symbol)
    plot_approach_profiles(profiles, symbol)
    plot_revisit_frequency(revisits, symbol)

    # Print summary
    print_summary(symbol, distances, resolution, profiles, revisits)

    ch_client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory analysis for model 2 parameter determination"
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbols to analyze")
    args = parser.parse_args()

    for symbol in [s.upper() for s in args.symbols]:
        run_analysis(symbol)

    log.info("Done.")


if __name__ == "__main__":
    main()
