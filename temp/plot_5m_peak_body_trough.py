"""Label and visualize BTC 5-minute peak-to-trough bear swings.

The label scheme follows the VNINDEX ZigZag example:

    3 = peak
    2 = bear body (bars strictly between a valid peak and trough)
    1 = trough
    0 = all remaining bars

Pivots are computed on the full close-price history. A random contiguous sample
is selected only after labeling so pivots near the sample boundaries are not
created from truncated data.

Example:
    python -m temp.plot_5m_peak_body_trough \
      --data data/crypto/BTCUSDT_5m.csv \
      --sample-bars 2000 \
      --zigzag-tolerance 0.003 \
      --min-drop 0.005 \
      --min-bars 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_NAMES = {0: "Other", 1: "Trough", 2: "Bear body", 3: "Peak"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label BTC 5m ZigZag peaks, bear bodies and troughs."
    )
    parser.add_argument("--data", default="data/crypto/BTCUSDT_5m.csv")
    parser.add_argument("--out-dir", default="temp/output")
    parser.add_argument("--sample-bars", type=int, default=2000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible random seed. Omit for a new sample each run.",
    )
    parser.add_argument(
        "--zigzag-tolerance",
        type=float,
        default=0.003,
        help="Reversal needed to confirm a pivot (0.003 = 0.30%%).",
    )
    parser.add_argument(
        "--min-drop",
        type=float,
        default=0.005,
        help="Minimum peak-to-trough decline (0.005 = 0.50%%).",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=3,
        help="Minimum bars from peak to trough.",
    )
    parser.add_argument(
        "--bars-per-panel",
        type=int,
        default=500,
        help="Number of bars in each plot panel.",
    )
    return parser.parse_args()


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = (
        df.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"No valid OHLC rows found in {path}")
    return df


def find_zigzag(close: pd.Series, tolerance: float) -> list[tuple[int, float, str]]:
    """Return confirmed pivots as ``(row_index, close_price, kind)``.

    A peak is confirmed only after close falls by ``tolerance`` from the latest
    running high. A trough is confirmed only after close rises by ``tolerance``
    from the latest running low. The unfinished final leg is intentionally not
    emitted because it has not yet received reversal confirmation.
    """
    if tolerance <= 0:
        raise ValueError("zigzag_tolerance must be greater than zero")

    prices = close.to_numpy(dtype=float)
    if prices.size < 2:
        return []

    pivots: list[tuple[int, float, str]] = []
    direction: str | None = None
    anchor_price = float(prices[0])
    candidate_idx = 0
    candidate_price = anchor_price

    for i in range(1, len(prices)):
        price = float(prices[i])

        if direction is None:
            relative_move = price / anchor_price - 1.0
            if relative_move >= tolerance:
                direction = "up"
                candidate_idx, candidate_price = i, price
            elif relative_move <= -tolerance:
                direction = "down"
                candidate_idx, candidate_price = i, price
            continue

        if direction == "up":
            if price >= candidate_price:
                candidate_idx, candidate_price = i, price
            elif price <= candidate_price * (1.0 - tolerance):
                pivots.append((candidate_idx, candidate_price, "peak"))
                direction = "down"
                candidate_idx, candidate_price = i, price
        else:
            if price <= candidate_price:
                candidate_idx, candidate_price = i, price
            elif price >= candidate_price * (1.0 + tolerance):
                pivots.append((candidate_idx, candidate_price, "trough"))
                direction = "up"
                candidate_idx, candidate_price = i, price

    return pivots


def find_valid_swings(
    pivots: list[tuple[int, float, str]],
    min_bars: int,
    min_drop: float,
) -> list[dict[str, int | float]]:
    if min_bars < 1:
        raise ValueError("min_bars must be at least 1")
    if min_drop < 0:
        raise ValueError("min_drop cannot be negative")

    swings: list[dict[str, int | float]] = []
    for peak, trough in zip(pivots, pivots[1:]):
        if peak[2] != "peak" or trough[2] != "trough":
            continue
        bars = trough[0] - peak[0]
        drop = (peak[1] - trough[1]) / peak[1]
        if bars >= min_bars and drop >= min_drop:
            swings.append(
                {
                    "peak_idx": peak[0],
                    "peak_price": peak[1],
                    "trough_idx": trough[0],
                    "trough_price": trough[1],
                    "bars": bars,
                    "drop": drop,
                }
            )
    return swings


def assign_labels(length: int, swings: list[dict[str, int | float]]) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int8)
    for swing in swings:
        peak_idx = int(swing["peak_idx"])
        trough_idx = int(swing["trough_idx"])
        labels[peak_idx] = 3
        labels[peak_idx + 1 : trough_idx] = 2
        labels[trough_idx] = 1
    return labels


def choose_sample(df: pd.DataFrame, sample_bars: int, seed: int | None) -> pd.DataFrame:
    if sample_bars < 1:
        raise ValueError("sample_bars must be greater than zero")
    if sample_bars > len(df):
        raise ValueError(f"sample_bars={sample_bars} exceeds available rows={len(df)}")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(df) - sample_bars + 1))
    sample = df.iloc[start : start + sample_bars].copy()
    sample["source_idx"] = np.arange(start, start + sample_bars)
    return sample.reset_index(drop=True)


def _shade_bear_runs(ax: plt.Axes, panel: pd.DataFrame) -> None:
    mask = panel["label"].eq(2).to_numpy()
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for left, right in changes.reshape(-1, 2):
        first = panel["date"].iloc[left]
        last = panel["date"].iloc[right - 1]
        ax.axvspan(first, last + pd.Timedelta(minutes=5), color="#ef4444", alpha=0.13)


def plot_sample(
    sample: pd.DataFrame,
    output_path: Path,
    tolerance: float,
    min_drop: float,
    min_bars: int,
    bars_per_panel: int,
) -> None:
    if bars_per_panel < 1:
        raise ValueError("bars_per_panel must be greater than zero")
    panel_count = int(np.ceil(len(sample) / bars_per_panel))
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(20, max(4.0 * panel_count, 6.0)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for panel_idx, ax in enumerate(axes):
        left = panel_idx * bars_per_panel
        panel = sample.iloc[left : left + bars_per_panel]
        _shade_bear_runs(ax, panel)
        ax.plot(panel["date"], panel["close"], color="#172033", linewidth=1.05)

        body = panel[panel["label"].eq(2)]
        peak = panel[panel["label"].eq(3)]
        trough = panel[panel["label"].eq(1)]
        ax.scatter(body["date"], body["close"], s=7, color="#ef4444", alpha=0.65)
        ax.scatter(
            peak["date"], peak["close"], marker="v", s=75,
            color="#f59e0b", edgecolor="white", linewidth=0.6, zorder=4,
        )
        ax.scatter(
            trough["date"], trough["close"], marker="^", s=75,
            color="#16a34a", edgecolor="white", linewidth=0.6, zorder=4,
        )
        ax.set_ylabel("BTCUSDT")
        ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.65)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlim(panel["date"].iloc[0], panel["date"].iloc[-1])

    start = sample["date"].iloc[0]
    end = sample["date"].iloc[-1]
    label_counts = sample["label"].value_counts()
    fig.suptitle(
        "BTCUSDT 5m - ZigZag peak / bear body / trough\n"
        f"{start} to {end} | bars={len(sample):,} | reversal={tolerance:.2%} | "
        f"min drop={min_drop:.2%} | min bars={min_bars} | "
        f"peaks={label_counts.get(3, 0)} troughs={label_counts.get(1, 0)}",
        fontsize=14,
    )
    fig.legend(
        handles=[
            plt.Line2D([], [], color="#172033", label="Close"),
            plt.Line2D([], [], marker="v", linestyle="", color="#f59e0b", label="3 Peak"),
            plt.Line2D([], [], marker="o", linestyle="", color="#ef4444", label="2 Bear body"),
            plt.Line2D([], [], marker="^", linestyle="", color="#16a34a", label="1 Trough"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    pivots = find_zigzag(df["close"], args.zigzag_tolerance)
    swings = find_valid_swings(pivots, args.min_bars, args.min_drop)
    df["label"] = assign_labels(len(df), swings)

    sample = choose_sample(df, args.sample_bars, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_tag = sample["date"].iloc[0].strftime("%Y%m%d_%H%M")
    end_tag = sample["date"].iloc[-1].strftime("%Y%m%d_%H%M")
    stem = f"btc_5m_peak_body_trough_{start_tag}_{end_tag}"
    image_path = out_dir / f"{stem}.png"
    csv_path = out_dir / f"{stem}.csv"

    plot_sample(
        sample,
        image_path,
        args.zigzag_tolerance,
        args.min_drop,
        args.min_bars,
        args.bars_per_panel,
    )
    sample.to_csv(csv_path, index=False)

    counts = sample["label"].value_counts().reindex(range(4), fill_value=0)
    print(f"Full rows: {len(df):,} | confirmed pivots: {len(pivots):,} | valid bear swings: {len(swings):,}")
    print(f"Random sample: {sample['date'].iloc[0]} -> {sample['date'].iloc[-1]}")
    for label in range(4):
        print(f"  label {label} ({LABEL_NAMES[label]}): {counts[label]:,} ({counts[label] / len(sample):.2%})")
    print(f"Saved image: {image_path}")
    print(f"Saved labels: {csv_path}")


if __name__ == "__main__":
    main()
