"""Plot slope-slowdown labels on a contiguous OHLCV sample.

Run from the repository root:

    python crypto/results/temp/plot_slope_slowdown.py

Edit the configuration block below to explore different definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crypto.data import load_ohlcv  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = REPO_ROOT / "data/crypto/BTCUSDT_15m.csv"
PRICE_COLUMN = "high"

SLOPE_LOOKBACK = 5
HORIZON = 3
SLOPE_SLOWDOWN_THRESHOLD = 0.0003  # 0.03% per candle
SLOPE_MIN_INITIAL = 0.0002  # 0.02% per candle

# "top": rising trend slows down; "bottom": falling trend recovers.
LABEL_SIDE = "top"

N_BARS = 1000
RANDOM_SEED = 42

# Set an integer to select an exact starting row. Keep None for random start.
START_ROW: int | None = None

OUTPUT_PATH = Path(__file__).with_name("slope_slowdown_preview.png")
FIGURE_SIZE = (20, 10)
IMAGE_DPI = 180


def rolling_log_ols_slope(close: pd.Series, window: int) -> pd.Series:
    """Return the OLS log-price slope as an approximate return per candle."""
    log_close = np.log(close.astype(float).to_numpy())
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = np.square(x_centered).sum()
    values = np.full(len(log_close), np.nan, dtype=float)

    for end in range(window - 1, len(log_close)):
        sample = log_close[end - window + 1 : end + 1]
        beta = np.dot(x_centered, sample - sample.mean()) / denominator
        values[end] = np.expm1(beta)

    return pd.Series(values, index=close.index)


def build_labels(
    initial_slope: pd.Series,
    expanded_slope: pd.Series,
) -> pd.Series:
    side = LABEL_SIDE.strip().lower()
    if side == "top":
        slowdown = initial_slope - expanded_slope
        return (initial_slope > SLOPE_MIN_INITIAL) & (
            slowdown > SLOPE_SLOWDOWN_THRESHOLD
        )
    if side == "bottom":
        recovery = expanded_slope - initial_slope
        return (initial_slope < -SLOPE_MIN_INITIAL) & (
            recovery > SLOPE_SLOWDOWN_THRESHOLD
        )
    raise ValueError("LABEL_SIDE must be 'top' or 'bottom'.")


def choose_start(valid: pd.Series, frame_length: int) -> int:
    valid_positions = np.flatnonzero(valid.to_numpy())
    if len(valid_positions) == 0:
        raise ValueError("No valid slope rows were produced.")

    first_start = int(valid_positions[0])
    last_start = int(valid_positions[-1]) - frame_length + 1
    if last_start < first_start:
        raise ValueError(f"Not enough valid rows for N_BARS={frame_length}.")

    if START_ROW is not None:
        start = int(START_ROW)
        if not first_start <= start <= last_start:
            raise ValueError(
                f"START_ROW must be between {first_start} and {last_start}."
            )
        return start

    rng = np.random.default_rng(RANDOM_SEED)
    return int(rng.integers(first_start, last_start + 1))


def main() -> None:
    df = load_ohlcv(DATA_PATH).copy()
    price = df[PRICE_COLUMN].astype(float)

    initial_slope = rolling_log_ols_slope(price, SLOPE_LOOKBACK)
    expanded_at_end = rolling_log_ols_slope(
        price,
        SLOPE_LOOKBACK + HORIZON,
    )
    # The expanded window ending at t+H is mapped back to decision row t.
    expanded_at_t = expanded_at_end.shift(-HORIZON)
    labels = build_labels(initial_slope, expanded_at_t)
    valid = initial_slope.notna() & expanded_at_t.notna()

    start = choose_start(valid, N_BARS)
    end = start + N_BARS

    sample = df.iloc[start:end]
    sample_initial = initial_slope.iloc[start:end]
    sample_expanded = expanded_at_t.iloc[start:end]
    sample_labels = labels.iloc[start:end]
    x_values = sample.index

    marked_x = np.asarray(x_values)[sample_labels.to_numpy()]
    marked_price = sample.loc[sample_labels, PRICE_COLUMN].astype(float).to_numpy()

    figure, (price_axis, slope_axis) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08},
    )

    price_axis.plot(
        x_values,
        sample[PRICE_COLUMN].astype(float),
        color="#243447",
        linewidth=1.0,
        label=PRICE_COLUMN.title(),
    )
    price_axis.scatter(
        marked_x,
        marked_price,
        color="#d62728",
        s=24,
        zorder=4,
        label=f"Label 1 (n={int(sample_labels.sum())})",
    )
    price_axis.set_ylabel(f"BTCUSDT {PRICE_COLUMN}")
    price_axis.grid(alpha=0.20)
    price_axis.legend(loc="upper left")
    price_axis.set_title(
        f"Slope slowdown labels on {N_BARS:,} consecutive candles\n"
        f"price={PRICE_COLUMN} | side={LABEL_SIDE} | "
        f"lookback={SLOPE_LOOKBACK} | H={HORIZON} | "
        f"min initial={SLOPE_MIN_INITIAL * 100:.3f}%/bar | "
        f"slowdown={SLOPE_SLOWDOWN_THRESHOLD * 100:.3f}%/bar"
    )

    slope_axis.plot(
        x_values,
        sample_initial * 100,
        color="#177245",
        linewidth=0.9,
        label=f"Initial OLS slope ({SLOPE_LOOKBACK} bars)",
    )
    slope_axis.plot(
        x_values,
        sample_expanded * 100,
        color="#3465a4",
        linewidth=0.9,
        label=f"Expanded OLS slope ({SLOPE_LOOKBACK + HORIZON} bars)",
    )
    initial_line = SLOPE_MIN_INITIAL if LABEL_SIDE.lower() == "top" else -SLOPE_MIN_INITIAL
    slope_axis.axhline(
        initial_line * 100,
        color="#177245",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    slope_axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    slope_axis.scatter(
        marked_x,
        (sample_initial[sample_labels] * 100).to_numpy(),
        color="#d62728",
        s=18,
        zorder=4,
    )
    slope_axis.set_ylabel("OLS slope (%/bar)")
    slope_axis.set_xlabel("Candle time")
    slope_axis.grid(alpha=0.20)
    slope_axis.legend(loc="upper left", ncol=2)

    figure.autofmt_xdate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=IMAGE_DPI, bbox_inches="tight")
    plt.close(figure)

    label_count = int(sample_labels.sum())
    print(f"Saved: {OUTPUT_PATH.resolve()}")
    print(f"Rows: {start}..{end - 1}")
    print(f"Time: {x_values[0]} .. {x_values[-1]}")
    print(
        f"Label 1: {label_count}/{N_BARS} "
        f"({label_count / N_BARS * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
