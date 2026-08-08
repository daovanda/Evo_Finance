"""Plot random 1-minute candles and highlight simple slope accumulation zones.

Definition for a window ending at candle t:

    abs(slope(close[t-3:t])) <= 10% * abs(slope(previous_window))

where ``previous_window`` can be any of the three four-candle windows ending
at t-1, t-2, or t-3. Slopes are ordinary least-squares slopes of log(close),
measured as log-return per candle. Every window is shifted by one candle. A
qualifying current window highlights all four of its candles.

PowerShell:
    python -m temp.plot_1m_slope_accumulation

Override examples:
    python -m temp.plot_1m_slope_accumulation --seed 7
    python -m temp.plot_1m_slope_accumulation --ratio 0.15 --bars 150
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import random
import secrets
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


DATA_PATH = Path("data/crypto/BTCUSDT_1m.csv")
OUTPUT_PATH = Path("temp/output/BTCUSDT_1m_slope_accumulation.png")
SAMPLE_BARS = 100
SLOPE_WINDOW = 3
SLOPE_RATIO = 0.10
PREVIOUS_WINDOWS = 3
RANDOM_SEED: int | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.plot_1m_slope_accumulation")


def _read_random_rows(
    path: Path,
    count: int,
    seed: int,
    max_attempts: int = 20,
) -> pd.DataFrame:
    """Read an approximately uniform random contiguous block without full CSV load."""
    if not path.exists():
        raise FileNotFoundError(path)
    rng = random.Random(int(seed))
    file_size = path.stat().st_size
    if file_size <= 0:
        raise ValueError(f"Empty CSV: {path}")

    with path.open("rb") as handle:
        header = handle.readline()
        header_end = handle.tell()
        if not header:
            raise ValueError(f"CSV has no header: {path}")

        for _ in range(max_attempts):
            offset = rng.randrange(header_end, max(header_end + 1, file_size))
            handle.seek(offset)
            if offset > header_end:
                handle.readline()  # Discard the partial row at the random offset.
            rows = []
            for _ in range(count):
                line = handle.readline()
                if not line:
                    break
                rows.append(line)
            if len(rows) != count:
                continue

            payload = header + b"".join(rows)
            frame = pd.read_csv(
                io.BytesIO(payload),
                usecols=["date", "open", "high", "low", "close"],
                parse_dates=["date"],
            )
            if len(frame) != count:
                continue
            frame = frame.sort_values("date").reset_index(drop=True)
            for column in ("open", "high", "low", "close"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[["open", "high", "low", "close"]].notna().all().all():
                return frame

    raise RuntimeError(
        f"Could not read {count} contiguous rows from {path} after "
        f"{max_attempts} attempts."
    )


def _ols_log_slope(values: np.ndarray) -> float:
    logs = np.log(np.asarray(values, dtype=float))
    x = np.arange(len(logs), dtype=float)
    x -= x.mean()
    return float(np.dot(x, logs - logs.mean()) / np.dot(x, x))


def detect_zones(
    close: np.ndarray,
    window: int,
    ratio: float,
    previous_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(window) < 2:
        raise ValueError("window must be at least 2.")
    if int(previous_windows) < 1:
        raise ValueError("previous_windows must be at least 1.")
    if not math.isfinite(float(ratio)) or float(ratio) < 0.0:
        raise ValueError("ratio must be finite and non-negative.")

    values = np.asarray(close, dtype=float)
    slopes = np.full(len(values), np.nan, dtype=float)
    slope_ratios = np.full(len(values), np.nan, dtype=float)
    qualifying_ends = np.zeros(len(values), dtype=bool)

    first_end = int(window) + int(previous_windows) - 1
    for end in range(first_end, len(values)):
        current = values[end - window + 1 : end + 1]
        current_slope = _ols_log_slope(current)
        previous_slopes = [
            _ols_log_slope(
                values[
                    end - offset - window + 1 : end - offset + 1
                ]
            )
            for offset in range(1, int(previous_windows) + 1)
        ]
        max_previous_abs = max(abs(value) for value in previous_slopes)
        slopes[end] = current_slope
        if max_previous_abs > 0.0:
            slope_ratios[end] = abs(current_slope) / max_previous_abs
        elif abs(current_slope) == 0.0:
            slope_ratios[end] = 0.0
        else:
            slope_ratios[end] = np.inf
        qualifying_ends[end] = (
            abs(current_slope) <= ratio * max_previous_abs
        )

    highlighted = np.zeros(len(values), dtype=bool)
    for end in np.flatnonzero(qualifying_ends):
        highlighted[end - window + 1 : end + 1] = True
    return highlighted, qualifying_ends, slope_ratios


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    spans: list[tuple[int, int]] = []
    start = previous = int(positions[0])
    for value in positions[1:]:
        current = int(value)
        if current != previous + 1:
            spans.append((start, previous))
            start = current
        previous = current
    spans.append((start, previous))
    return spans


def plot_candles(
    frame: pd.DataFrame,
    highlighted: np.ndarray,
    qualifying_ends: np.ndarray,
    slope_ratios: np.ndarray,
    *,
    display_start: int,
    window: int,
    previous_windows: int,
    ratio: float,
    seed: int,
    output: Path,
) -> None:
    display = frame.iloc[display_start:].reset_index(drop=True)
    display_highlighted = highlighted[display_start:]
    display_ends = qualifying_ends[display_start:]
    display_ratios = slope_ratios[display_start:]
    x = np.arange(len(display), dtype=float)

    fig, (price_ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(20, 9),
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.08},
        sharex=True,
    )
    fig.patch.set_facecolor("#11161d")
    for axis in (price_ax, ratio_ax):
        axis.set_facecolor("#11161d")
        axis.grid(True, color="#39424e", alpha=0.45, linewidth=0.6)
        axis.tick_params(colors="#d7dde5")
        for spine in axis.spines.values():
            spine.set_color("#596371")

    for start, end in _contiguous_spans(display_highlighted):
        price_ax.axvspan(
            start - 0.5,
            end + 0.5,
            color="#ff3b30",
            alpha=0.18,
            linewidth=0,
        )
        ratio_ax.axvspan(
            start - 0.5,
            end + 0.5,
            color="#ff3b30",
            alpha=0.12,
            linewidth=0,
        )

    candle_width = 0.62
    for index, row in display.iterrows():
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        rising = close_price >= open_price
        color = "#29d17d" if rising else "#f4f7fb"
        price_ax.vlines(index, low_price, high_price, color=color, linewidth=0.8)
        body_low = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 1e-9)
        price_ax.add_patch(
            Rectangle(
                (index - candle_width / 2.0, body_low),
                candle_width,
                body_height,
                facecolor=color if rising else "#f4f7fb",
                edgecolor=color,
                linewidth=0.8,
            )
        )

    finite_ratio = np.where(np.isfinite(display_ratios), display_ratios, np.nan)
    ratio_ax.plot(x, finite_ratio, color="#62a8ff", linewidth=1.1)
    ratio_ax.axhline(
        ratio,
        color="#ffcc4d",
        linestyle="--",
        linewidth=1.0,
        label=f"threshold = {ratio:.2f}",
    )
    ratio_ax.scatter(
        x[display_ends],
        finite_ratio[display_ends],
        color="#ff3b30",
        s=20,
        zorder=3,
    )
    ratio_ax.set_ylim(
        0.0,
        max(
            ratio * 2.5,
            float(np.nanpercentile(finite_ratio, 90))
            if np.isfinite(finite_ratio).any()
            else 1.0,
        ),
    )
    ratio_ax.set_ylabel(
        f"|slope{window}| /\nmax previous {previous_windows}",
        color="#d7dde5",
    )
    ratio_ax.legend(loc="upper right", frameon=False, labelcolor="#d7dde5")

    tick_positions = np.unique(
        np.linspace(0, len(display) - 1, 11, dtype=int)
    )
    tick_labels = [
        pd.Timestamp(display.iloc[position]["date"]).strftime("%m-%d %H:%M")
        for position in tick_positions
    ]
    ratio_ax.set_xticks(tick_positions)
    ratio_ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    start_time = pd.Timestamp(display["date"].iloc[0])
    end_time = pd.Timestamp(display["date"].iloc[-1])
    zone_windows = int(display_ends.sum())
    highlighted_bars = int(display_highlighted.sum())
    price_ax.set_title(
        "BTCUSDT 1m simple slope accumulation\n"
        f"{start_time} to {end_time} | seed={seed} | "
        f"abs(current slope{window}) <= {ratio:.0%} of "
        f"any of previous {previous_windows} slope{window} windows | "
        f"qualifying windows={zone_windows}, highlighted bars={highlighted_bars}",
        color="#f4f7fb",
        fontsize=12,
        pad=12,
    )
    price_ax.set_ylabel("Price", color="#d7dde5")
    price_ax.legend(
        handles=[
            Patch(
                facecolor="#ff3b30",
                alpha=0.25,
                label="Accumulation zone",
            )
        ],
        loc="upper left",
        frameon=False,
        labelcolor="#d7dde5",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--bars", type=int, default=SAMPLE_BARS)
    parser.add_argument("--window", type=int, default=SLOPE_WINDOW)
    parser.add_argument(
        "--previous-windows",
        type=int,
        default=PREVIOUS_WINDOWS,
    )
    parser.add_argument("--ratio", type=float, default=SLOPE_RATIO)
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            "Optional reproducible random seed. When omitted, a new seed is "
            "generated for every run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bars <= 0:
        raise ValueError("bars must be positive.")
    effective_seed = (
        int(args.seed) if args.seed is not None else secrets.randbits(32)
    )
    context_bars = int(args.window) + int(args.previous_windows) - 1
    frame = _read_random_rows(
        path=args.data,
        count=int(args.bars) + context_bars,
        seed=effective_seed,
    )
    highlighted, qualifying_ends, slope_ratios = detect_zones(
        close=frame["close"].to_numpy(dtype=float),
        window=int(args.window),
        ratio=float(args.ratio),
        previous_windows=int(args.previous_windows),
    )
    plot_candles(
        frame,
        highlighted,
        qualifying_ends,
        slope_ratios,
        display_start=context_bars,
        window=int(args.window),
        previous_windows=int(args.previous_windows),
        ratio=float(args.ratio),
        seed=effective_seed,
        output=args.output,
    )
    displayed_highlighted = highlighted[context_bars:]
    displayed_ends = qualifying_ends[context_bars:]
    logger.info(
        "Saved %s | sample=%s..%s | qualifying_windows=%d | "
        "highlighted_bars=%d/%d",
        args.output,
        frame["date"].iloc[context_bars],
        frame["date"].iloc[-1],
        int(displayed_ends.sum()),
        int(displayed_highlighted.sum()),
        int(args.bars),
    )


if __name__ == "__main__":
    main()
