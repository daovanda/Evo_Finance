"""Label and visualize BTC 5-minute trough-to-peak bull swings.

Label scheme:

    3 = peak
    2 = bull body (bars strictly between a valid trough and peak)
    1 = trough
    0 = all remaining bars

ZigZag pivots are calculated on the complete close-price history before a
random contiguous sample is selected. This prevents artificial pivots at the
sample boundaries.

PowerShell:
    python -m temp.plot_5m_trough_body_peak `
      --data data/crypto/BTCUSDT_5m.csv `
      --sample-bars 2000 `
      --zigzag-tolerance 0.003 `
      --min-rise 0.005 `
      --min-bars 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temp.plot_5m_peak_body_trough import choose_sample, find_zigzag, load_data


LABEL_NAMES = {0: "Other", 1: "Trough", 2: "Bull body", 3: "Peak"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label BTC 5m ZigZag troughs, bull bodies and peaks."
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
        "--min-rise",
        type=float,
        default=0.005,
        help="Minimum trough-to-peak rise (0.005 = 0.50%%).",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=3,
        help="Minimum bars from trough to peak.",
    )
    parser.add_argument(
        "--bars-per-panel",
        type=int,
        default=500,
        help="Number of bars in each plot panel.",
    )
    return parser.parse_args()


def find_valid_bull_swings(
    pivots: list[tuple[int, float, str]],
    min_bars: int,
    min_rise: float,
) -> list[dict[str, int | float]]:
    if min_bars < 1:
        raise ValueError("min_bars must be at least 1")
    if min_rise < 0:
        raise ValueError("min_rise cannot be negative")

    swings: list[dict[str, int | float]] = []
    for trough, peak in zip(pivots, pivots[1:]):
        if trough[2] != "trough" or peak[2] != "peak":
            continue
        bars = int(peak[0] - trough[0])
        rise = float((peak[1] - trough[1]) / trough[1])
        if bars >= int(min_bars) and rise >= float(min_rise):
            swings.append(
                {
                    "trough_idx": trough[0],
                    "trough_price": trough[1],
                    "peak_idx": peak[0],
                    "peak_price": peak[1],
                    "bars": bars,
                    "rise": rise,
                }
            )
    return swings


def assign_bull_labels(
    length: int,
    swings: list[dict[str, int | float]],
) -> np.ndarray:
    labels = np.zeros(int(length), dtype=np.int8)
    for swing in swings:
        trough_idx = int(swing["trough_idx"])
        peak_idx = int(swing["peak_idx"])
        labels[trough_idx] = 1
        labels[trough_idx + 1 : peak_idx] = 2
        labels[peak_idx] = 3
    return labels


def _shade_bull_runs(ax: plt.Axes, panel: pd.DataFrame) -> None:
    mask = panel["label"].eq(2).to_numpy()
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for left, right in changes.reshape(-1, 2):
        first = panel["date"].iloc[left]
        last = panel["date"].iloc[right - 1]
        ax.axvspan(
            first,
            last + pd.Timedelta(minutes=5),
            color="#22c55e",
            alpha=0.14,
        )


def plot_sample(
    sample: pd.DataFrame,
    output_path: Path,
    tolerance: float,
    min_rise: float,
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
        _shade_bull_runs(ax, panel)
        ax.plot(panel["date"], panel["close"], color="#172033", linewidth=1.05)

        body = panel[panel["label"].eq(2)]
        peak = panel[panel["label"].eq(3)]
        trough = panel[panel["label"].eq(1)]
        ax.scatter(body["date"], body["close"], s=7, color="#16a34a", alpha=0.7)
        ax.scatter(
            peak["date"],
            peak["close"],
            marker="v",
            s=75,
            color="#f59e0b",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.scatter(
            trough["date"],
            trough["close"],
            marker="^",
            s=75,
            color="#2563eb",
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.set_ylabel("BTCUSDT")
        ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.65)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlim(panel["date"].iloc[0], panel["date"].iloc[-1])

    start = sample["date"].iloc[0]
    end = sample["date"].iloc[-1]
    label_counts = sample["label"].value_counts()
    fig.suptitle(
        "BTCUSDT 5m - ZigZag trough / bull body / peak\n"
        f"{start} to {end} | bars={len(sample):,} | reversal={tolerance:.2%} | "
        f"min rise={min_rise:.2%} | min bars={min_bars} | "
        f"troughs={label_counts.get(1, 0)} peaks={label_counts.get(3, 0)}",
        fontsize=14,
    )
    fig.legend(
        handles=[
            plt.Line2D([], [], color="#172033", label="Close"),
            plt.Line2D(
                [], [], marker="^", linestyle="", color="#2563eb", label="1 Trough"
            ),
            plt.Line2D(
                [], [], marker="o", linestyle="", color="#16a34a", label="2 Bull body"
            ),
            plt.Line2D(
                [], [], marker="v", linestyle="", color="#f59e0b", label="3 Peak"
            ),
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
    swings = find_valid_bull_swings(pivots, args.min_bars, args.min_rise)
    df["label"] = assign_bull_labels(len(df), swings)

    sample = choose_sample(df, args.sample_bars, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_tag = sample["date"].iloc[0].strftime("%Y%m%d_%H%M")
    end_tag = sample["date"].iloc[-1].strftime("%Y%m%d_%H%M")
    stem = f"btc_5m_trough_body_peak_{start_tag}_{end_tag}"
    image_path = out_dir / f"{stem}.png"
    csv_path = out_dir / f"{stem}.csv"

    plot_sample(
        sample,
        image_path,
        args.zigzag_tolerance,
        args.min_rise,
        args.min_bars,
        args.bars_per_panel,
    )
    sample.to_csv(csv_path, index=False)

    counts = sample["label"].value_counts().reindex(range(4), fill_value=0)
    print(
        f"Full rows: {len(df):,} | confirmed pivots: {len(pivots):,} | "
        f"valid bull swings: {len(swings):,}"
    )
    print(f"Random sample: {sample['date'].iloc[0]} -> {sample['date'].iloc[-1]}")
    for label in range(4):
        print(
            f"  label {label} ({LABEL_NAMES[label]}): {counts[label]:,} "
            f"({counts[label] / len(sample):.2%})"
        )
    print(f"Saved image: {image_path}")
    print(f"Saved labels: {csv_path}")


if __name__ == "__main__":
    main()
