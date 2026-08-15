"""Backtest MA3 slope-reversal score bands as fixed-horizon Short signals.

At signal candle ``t``:

* require the observable MA3 slope at ``t`` to be positive;
* enter Short at ``open(t + 1)``;
* keep a Short TP active from H1 onward;
* inspect MA3 slope at ``t + 2``: if it is not negative, exit at close H3;
* otherwise hold until TP or the first later close whose MA3 slope is positive;
* allow independent overlapping trades.

Score-band cutoffs are fitted on eligible Final Val rows and applied unchanged
to Test. This keeps Test genuinely out of sample.

PowerShell:
    python -m temp.backtest_5m_ma_slope_reversal_short_h3 `
      --archive crypto/results/crypto_btc_5m_long_ma_slope_reversal_fs2_top20_seed1_resume_seed2_2h.json `
      --rank 1 `
      --band-step 0.05 `
      --take-profit 0.002 `
      --trade-cost 0.0002 `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.backtest import ModelSpec, _archive_horizons, _load_rank_entry
from crypto.data import load_ohlcv
from temp.plot_5m_ma_slope_reversal_signals import (
    DEFAULT_ARCHIVE,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA,
    DEFAULT_OUT_DIR,
    _ma3_slope_positive,
    _metadata,
    _train_or_load,
)


OUTPUT_NAME = "ma3_reversal_short_h3_bands.png"


def _simulate_short_paths(
    raw: pd.DataFrame,
    signal_index: pd.Index,
    *,
    take_profit: float,
) -> pd.DataFrame:
    """Simulate each signal in timestamp order using only observable exits."""
    raw_index = raw.index
    opens = pd.to_numeric(raw["open"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(raw["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(raw["close"], errors="coerce").to_numpy(float)
    close_series = pd.to_numeric(raw["close"], errors="coerce")
    ma3 = close_series.rolling(config.MA_SLOPE_FAST_WINDOW).mean()
    slope = (
        ma3 - ma3.shift(config.MA_SLOPE_FAST_SHIFT)
    ).reindex(raw_index).to_numpy(float)
    positions = raw_index.get_indexer(signal_index)
    records: list[tuple[object, ...]] = []

    for signal_time, signal_position in zip(signal_index, positions, strict=True):
        if signal_position < 0 or signal_position + 3 >= len(raw):
            continue
        entry_position = int(signal_position + 1)
        decision_position = int(signal_position + 2)
        entry_price = float(opens[entry_position])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        tp_price = entry_price * (1.0 - float(take_profit))
        slope_at_h2 = float(slope[decision_position])
        reversed_at_h2 = bool(np.isfinite(slope_at_h2) and slope_at_h2 < 0.0)

        if reversed_at_h2:
            exit_limit = len(raw) - 1
            close_exit_type = "end_close"
            for position in range(decision_position + 1, len(raw)):
                if np.isfinite(slope[position]) and slope[position] > 0.0:
                    exit_limit = position
                    close_exit_type = "slope_positive_close"
                    break
        else:
            exit_limit = int(signal_position + 3)
            close_exit_type = "no_reversal_close_h3"

        exit_position = exit_limit
        exit_price = float(closes[exit_position])
        exit_type = close_exit_type
        gross_return = 1.0 - exit_price / entry_price
        for position in range(entry_position, exit_limit + 1):
            if float(lows[position]) <= tp_price:
                exit_position = position
                exit_price = tp_price
                exit_type = "tp"
                gross_return = float(take_profit)
                break

        if not np.isfinite(gross_return):
            continue
        records.append(
            (
                signal_time,
                raw_index[entry_position],
                raw_index[exit_position],
                entry_price,
                exit_price,
                bool(reversed_at_h2),
                exit_type,
                int(exit_position - entry_position + 1),
                float(gross_return),
            )
        )
    return pd.DataFrame.from_records(
        records,
        index="signal_time",
        columns=[
            "signal_time",
            "entry_time",
            "exit_time",
            "entry_open_h1",
            "exit_price",
            "reversed_at_h2",
            "exit_type",
            "holding_bars",
            "gross_short_return",
        ],
    )


def _eligible_split(
    signals: object,
    raw: pd.DataFrame,
    slope_positive: pd.Series,
    *,
    take_profit: float,
) -> pd.DataFrame:
    data = signals.data[["pred", "label"]].copy()
    data["ma3_slope_positive"] = (
        slope_positive.reindex(data.index).fillna(False).astype(bool)
    )
    data = data.loc[data["ma3_slope_positive"]].dropna(subset=["pred"])
    outcomes = _simulate_short_paths(
        raw,
        data.index,
        take_profit=take_profit,
    )
    return data.join(outcomes, how="inner")


def _band_rows(
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    band_step: float,
    trade_cost: float,
) -> list[dict[str, float | int | str]]:
    fractions = np.arange(band_step, 1.0 + band_step / 2.0, band_step)
    val_sorted = val.sort_values("pred", ascending=False)
    cutoffs: list[float] = []
    for fraction in fractions:
        count = min(len(val_sorted), max(1, int(np.ceil(len(val_sorted) * fraction))))
        cutoffs.append(float(val_sorted.iloc[count - 1]["pred"]))

    rows: list[dict[str, float | int | str]] = []
    for split_name, data in (("val", val), ("test", test)):
        previous = pd.Index([])
        for fraction, cutoff in zip(fractions, cutoffs, strict=True):
            cumulative_index = pd.Index(data.index[data["pred"].ge(cutoff)])
            band_index = cumulative_index.difference(previous, sort=False)
            band = data.reindex(band_index).dropna(subset=["gross_short_return"])
            cumulative = data.reindex(cumulative_index).dropna(
                subset=["gross_short_return"]
            )
            gross = band["gross_short_return"]
            net = gross - float(trade_cost)
            cumulative_net = cumulative["gross_short_return"] - float(trade_cost)
            start = fraction - band_step
            rows.append(
                {
                    "split": split_name,
                    "score band": f"top {100.0 * start:.0f}-{100.0 * fraction:.0f}%",
                    "n": len(band),
                    "gross mean": float(gross.mean()),
                    "E[net]": float(net.mean()),
                    "net win rate": float(net.gt(0.0).mean()),
                    "gross median": float(gross.median()),
                    "gross Q10": float(gross.quantile(0.10)),
                    "gross Q90": float(gross.quantile(0.90)),
                    "P(gross loss > 0.10%)": float(gross.lt(-0.001).mean()),
                    "TP hit rate": float(band["exit_type"].eq("tp").mean()),
                    "no-reversal close H3": float(
                        band["exit_type"].eq("no_reversal_close_h3").mean()
                    ),
                    "slope-positive exit": float(
                        band["exit_type"].eq("slope_positive_close").mean()
                    ),
                    "holding bars mean": float(band["holding_bars"].mean()),
                    "cumulative n": len(cumulative),
                    "cumulative E[net]": float(cumulative_net.mean()),
                    "Val cutoff": cutoff,
                }
            )
            previous = cumulative_index
    return rows


def _render_table(rows: list[dict[str, float | int | str]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    columns = [
        "split",
        "score band",
        "n",
        "gross mean",
        "E[net]",
        "net win rate",
        "gross median",
        "gross Q10",
        "gross Q90",
        "P(gross loss > 0.10%)",
        "TP hit rate",
        "no-reversal close H3",
        "slope-positive exit",
        "holding bars mean",
        "cumulative n",
        "cumulative E[net]",
    ]
    display = frame[columns].copy()
    for column in (
        "gross mean",
        "E[net]",
        "net win rate",
        "gross median",
        "gross Q10",
        "gross Q90",
        "P(gross loss > 0.10%)",
        "TP hit rate",
        "no-reversal close H3",
        "slope-positive exit",
        "cumulative E[net]",
    ):
        display[column] = display[column].map(
            lambda value: f"{100.0 * value:+.3f}%"
            if column not in (
                "net win rate",
                "P(gross loss > 0.10%)",
                "TP hit rate",
                "no-reversal close H3",
                "slope-positive exit",
            )
            else f"{100.0 * value:.2f}%"
        )
    display["n"] = display["n"].map(lambda value: f"{int(value):,}")
    display["cumulative n"] = display["cumulative n"].map(
        lambda value: f"{int(value):,}"
    )
    display["holding bars mean"] = display["holding bars mean"].map(
        lambda value: f"{float(value):.2f}"
    )

    fig_height = max(8.0, 0.36 * len(display) + 2.2)
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#fff7ed")
    ax.set_title(
        "MA3 slope-reversal score bands: Short open H1 | TP or state exit\n"
        "At H2: no reversal -> close H3; reversal -> hold until MA3 slope > 0",
        fontsize=14,
        pad=18,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--band-step", type=float, default=0.05)
    parser.add_argument("--take-profit", type=float, default=0.002)
    parser.add_argument("--trade-cost", type=float, default=config.TRADE_COST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-model-cache", action="store_true")
    parser.add_argument("--no-model-cache", action="store_true")
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive.")
    if not 0.0 < args.band_step <= 1.0:
        parser.error("--band-step must be in (0, 1].")
    if not np.isfinite(args.take_profit) or args.take_profit <= 0.0:
        parser.error("--take-profit must be finite and positive.")
    if not np.isfinite(args.trade_cost) or args.trade_cost < 0.0:
        parser.error("--trade-cost must be finite and non-negative.")
    # _train_or_load expects this field; all predictions are retained before
    # this file creates its own Val-derived score bands.
    args.top_fraction = 1.0
    return args


def main() -> None:
    args = parse_args()
    metadata = _metadata(args.archive)
    horizons = _archive_horizons(
        args.archive,
        fallback=[1],
        label="MA slope-reversal archive",
    )
    entry = _load_rank_entry(args.archive, args.rank)
    spec = ModelSpec(
        archive_path=args.archive,
        rank=args.rank,
        label_mode="ma_slope_reversal",
        label_threshold=float(metadata.get("label_threshold", 0.0)),
        top_fraction=1.0,
        label_direction="long",
    )
    raw = load_ohlcv(args.data)
    bundle = _train_or_load(
        args=args,
        raw=raw,
        metadata=metadata,
        spec=spec,
        entry=entry,
        horizons=horizons,
    )
    slope_positive = _ma3_slope_positive(raw)
    val = _eligible_split(
        bundle.val,
        raw,
        slope_positive,
        take_profit=args.take_profit,
    )
    test = _eligible_split(
        bundle.test,
        raw,
        slope_positive,
        take_profit=args.take_profit,
    )
    rows = _band_rows(
        val,
        test,
        band_step=args.band_step,
        trade_cost=args.trade_cost,
    )
    frame = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(
            "\n=== MA3 reversal Short | "
            f"TP={100.0 * args.take_profit:.3f}% | state-dependent exit ==="
        )
        print(frame.to_string(index=False))
    output = args.out_dir / OUTPUT_NAME
    _render_table(rows, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
