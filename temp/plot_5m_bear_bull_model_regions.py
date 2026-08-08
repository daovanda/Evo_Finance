"""Plot Bear/Bull model selections on a random BTCUSDT 5-minute segment.

Both archive ranks are retrained on the final Train split. The top-fraction
cutoff is learned on Final Val and then applied unchanged to Test, matching
``crypto.backtest``. A random contiguous sample is drawn wholly from either
Val or Test so it never crosses the purged split boundary.

PowerShell:
    python -m temp.plot_5m_bear_bull_model_regions `
      --bear-archive crypto/results/crypto_btc_5m_bear_top40_seed1_8h.json `
      --bull-archive crypto/results/crypto_btc_5m_bull_top40_seed1_8h.json `
      --top-fraction 0.40 `
      --sample-bars 2000 `
      --take-profit 0.004 `
      --stop-loss 0.002 `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import (
    BundleSignals,
    ModelSpec,
    SplitSignals,
    _archive_horizons,
    _cached_feature_space,
    _load_rank_entry,
    _quality_train_index,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv
from temp.backtest_5m_long_mfe_fixed_tp_sl_h3 import (
    _bundle_with_top_fraction,
    _load_trained_bundle_cache,
    _save_trained_bundle_cache,
    _trained_bundle_cache_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.plot_5m_bear_bull_model_regions")

DEFAULT_BEAR_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_bear_top40_seed1_8h.json"
)
DEFAULT_BULL_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_bull_top40_seed1_8h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_CACHE_DIR = Path("temp/model")

STATE_NAMES = {
    0: "No model",
    1: "Bear only",
    2: "Bull only",
    3: "Bear + Bull",
}
STATE_COLORS = {
    1: "#ef4444",
    2: "#22c55e",
    3: "#6b7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bear-archive", default=str(DEFAULT_BEAR_ARCHIVE))
    parser.add_argument("--bull-archive", default=str(DEFAULT_BULL_ARCHIVE))
    parser.add_argument("--bear-rank", type=int, default=1)
    parser.add_argument("--bull-rank", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.40)
    parser.add_argument("--sample-bars", type=int, default=2000)
    parser.add_argument("--bars-per-panel", type=int, default=500)
    parser.add_argument("--take-profit", type=float, default=0.004)
    parser.add_argument("--stop-loss", type=float, default=0.002)
    parser.add_argument("--trade-cost", type=float, default=0.00016)
    parser.add_argument(
        "--same-candle-policy",
        choices=("stop_first", "tp_first"),
        default="stop_first",
        help="Barrier ordering when both TP and SL are touched in one candle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible seed; omit for a new random segment each run.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument(
        "--rebuild-model-cache",
        action="store_true",
        help="Ignore matching cached predictions and retrain both archive ranks.",
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Do not read or write persistent trained-model predictions.",
    )
    return parser.parse_args()


def _archive_metadata(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Archive metadata is missing or invalid: {path}")
    return metadata


def _make_spec(path: Path, rank: int, mode: str, top_fraction: float) -> ModelSpec:
    metadata = _archive_metadata(path)
    archived_mode = config.canonical_label_mode(str(metadata.get("label_mode", "")))
    if archived_mode != mode:
        raise ValueError(
            f"Expected {mode!r} archive, got mode={archived_mode!r}: {path}"
        )
    return ModelSpec(
        archive_path=path,
        rank=int(rank),
        label_mode=archived_mode,
        label_threshold=float(metadata.get("label_threshold", 0.0)),
        top_fraction=float(top_fraction),
        label_direction=config.canonical_label_direction(
            str(metadata.get("label_direction", "long"))
        ),
    )


def _train_or_load_bundle(
    *,
    spec: ModelSpec,
    entry: dict[str, object],
    horizons: list[int],
    raw_df: pd.DataFrame,
    data_path: Path,
    feature_space: object | None,
    args: argparse.Namespace,
    purge_bars: int,
    preloaded_bundle: BundleSignals | None = None,
) -> BundleSignals:
    cache_path = _trained_bundle_cache_path(
        cache_dir=Path(args.model_cache_dir),
        archive_path=spec.archive_path,
        data_path=data_path,
        raw_df=raw_df,
        spec=spec,
        horizons=horizons,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )
    bundle = preloaded_bundle
    if (
        bundle is None
        and not args.no_model_cache
        and not args.rebuild_model_cache
    ):
        bundle = _load_trained_bundle_cache(cache_path)
    if bundle is None:
        if feature_space is None:
            raise RuntimeError("Feature space is required to train an uncached model.")
        bundle = _train_spec_bundle(
            spec=spec,
            entry=entry,
            raw_df=raw_df,
            feature_space=feature_space,
            horizons=horizons,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            purge_bars=purge_bars,
        )
        if not args.no_model_cache:
            _save_trained_bundle_cache(cache_path, bundle)
    return _bundle_with_top_fraction(bundle, spec.top_fraction)


def _aligned_split_frame(
    split: str,
    bear: SplitSignals,
    bull: SplitSignals,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    common = bear.data.index.intersection(bull.data.index).intersection(raw_df.index)
    frame = pd.DataFrame(index=common)
    frame["open"] = raw_df["open"].reindex(common)
    frame["high"] = raw_df["high"].reindex(common)
    frame["low"] = raw_df["low"].reindex(common)
    frame["close"] = raw_df["close"].reindex(common)
    frame["bear_score"] = bear.data["pred"].reindex(common)
    frame["bull_score"] = bull.data["pred"].reindex(common)
    frame["bear_selected"] = frame.index.isin(bear.selected_index)
    frame["bull_selected"] = frame.index.isin(bull.selected_index)
    frame["state"] = (
        frame["bear_selected"].astype(np.int8)
        + 2 * frame["bull_selected"].astype(np.int8)
    )
    frame["split"] = split
    return frame.dropna(subset=["open", "high", "low", "close"])


def _choose_sample(
    split_frames: dict[str, pd.DataFrame],
    sample_bars: int,
    seed: int | None,
) -> pd.DataFrame:
    if sample_bars < 1:
        raise ValueError("sample-bars must be greater than zero")
    eligible = {
        name: frame
        for name, frame in split_frames.items()
        if len(frame) >= sample_bars
    }
    if not eligible:
        sizes = {name: len(frame) for name, frame in split_frames.items()}
        raise ValueError(
            f"sample-bars={sample_bars} exceeds every available split: {sizes}"
        )

    names = list(eligible)
    start_counts = np.asarray(
        [len(eligible[name]) - sample_bars + 1 for name in names], dtype=float
    )
    rng = np.random.default_rng(seed)
    split = str(rng.choice(names, p=start_counts / start_counts.sum()))
    frame = eligible[split]
    start = int(rng.integers(0, len(frame) - sample_bars + 1))
    return frame.iloc[start : start + sample_bars].copy()


def _shade_state_runs(ax: plt.Axes, panel: pd.DataFrame) -> None:
    states = panel["state"].to_numpy(dtype=np.int8)
    if not len(states):
        return
    boundaries = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = int(states[left])
        if state == 0:
            continue
        first = panel.index[left]
        last = panel.index[right - 1] + pd.Timedelta(minutes=5)
        ax.axvspan(first, last, color=STATE_COLORS[state], alpha=0.18, linewidth=0)


def _plot_sample(
    sample: pd.DataFrame,
    output_path: Path,
    bars_per_panel: int,
    bear_threshold: float,
    bull_threshold: float,
    top_fraction: float,
) -> None:
    if bars_per_panel < 1:
        raise ValueError("bars-per-panel must be greater than zero")
    panel_count = int(np.ceil(len(sample) / bars_per_panel))
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(21, max(4.0 * panel_count, 6.5)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for panel_index, ax in enumerate(axes):
        left = panel_index * bars_per_panel
        panel = sample.iloc[left : left + bars_per_panel]
        _shade_state_runs(ax, panel)
        ax.plot(panel.index, panel["close"], color="#111827", linewidth=1.05)
        ax.set_ylabel("BTCUSDT")
        ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.65)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlim(panel.index[0], panel.index[-1] + pd.Timedelta(minutes=5))

    counts = sample["state"].value_counts().reindex(range(4), fill_value=0)
    start = sample.index[0]
    end = sample.index[-1]
    split = str(sample["split"].iloc[0]).upper()
    fig.suptitle(
        "BTCUSDT 5m - Bear/Bull model-selected regions\n"
        f"{split} | {start} to {end} | bars={len(sample):,} | top={top_fraction:.0%} | "
        f"Bear cutoff={bear_threshold:.6f} | Bull cutoff={bull_threshold:.6f}\n"
        f"Bear only={counts[1]:,} ({counts[1] / len(sample):.1%}) | "
        f"Bull only={counts[2]:,} ({counts[2] / len(sample):.1%}) | "
        f"Both={counts[3]:,} ({counts[3] / len(sample):.1%})",
        fontsize=13,
    )
    axes[0].legend(
        handles=[
            plt.Line2D([], [], color="#111827", label="Close"),
            Patch(facecolor=STATE_COLORS[1], alpha=0.35, label="Bear only"),
            Patch(facecolor=STATE_COLORS[2], alpha=0.35, label="Bull only"),
            Patch(facecolor=STATE_COLORS[3], alpha=0.35, label="Bear + Bull"),
        ],
        loc="upper left",
        ncol=4,
        frameon=True,
        framealpha=0.88,
        fontsize=9,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _simulate_directional_trades(
    sample: pd.DataFrame,
    raw_df: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
) -> pd.DataFrame:
    """Open Short on Bear-only and Long on Bull-only at next-bar open."""
    if take_profit <= 0.0 or stop_loss <= 0.0:
        raise ValueError("take-profit and stop-loss must both be greater than zero")
    if trade_cost < 0.0:
        raise ValueError("trade-cost cannot be negative")

    raw_index = raw_df.index
    raw_open = raw_df["open"].to_numpy(dtype=float)
    raw_high = raw_df["high"].to_numpy(dtype=float)
    raw_low = raw_df["low"].to_numpy(dtype=float)
    raw_close = raw_df["close"].to_numpy(dtype=float)
    records: list[tuple[object, ...]] = []
    signals = sample[sample["state"].isin((1, 2))]
    signal_positions = raw_index.get_indexer(signals.index)
    signal_states = signals["state"].to_numpy(dtype=np.int8)
    for signal_time, source_position, state in zip(
        signals.index, signal_positions, signal_states, strict=True
    ):
        if source_position < 0 or source_position + 1 >= len(raw_df):
            continue
        entry_position = int(source_position + 1)
        entry_time = raw_index[entry_position]
        entry_price = raw_open[entry_position]
        direction = "short" if int(state) == 1 else "long"
        sign = -1.0 if direction == "short" else 1.0
        tp_price = entry_price * (
            1.0 - take_profit if direction == "short" else 1.0 + take_profit
        )
        sl_price = entry_price * (
            1.0 + stop_loss if direction == "short" else 1.0 - stop_loss
        )

        exit_position = len(raw_df) - 1
        exit_time = raw_index[exit_position]
        exit_price = raw_close[exit_position]
        gross_return = sign * (exit_price / entry_price - 1.0)
        outcome = "end_close"
        for position in range(entry_position, len(raw_df)):
            high = raw_high[position]
            low = raw_low[position]
            if direction == "long":
                tp_hit = high >= tp_price
                sl_hit = low <= sl_price
            else:
                tp_hit = low <= tp_price
                sl_hit = high >= sl_price
            if not tp_hit and not sl_hit:
                continue

            exit_position = position
            exit_time = raw_index[position]
            if tp_hit and sl_hit:
                outcome = same_candle_policy
                tp_wins = same_candle_policy == "tp_first"
            else:
                tp_wins = bool(tp_hit)
                outcome = "tp" if tp_wins else "sl"
            exit_price = tp_price if tp_wins else sl_price
            gross_return = take_profit if tp_wins else -stop_loss
            break

        records.append(
            (
                signal_time,
                entry_time,
                exit_time,
                direction,
                entry_price,
                exit_price,
                outcome,
                float(gross_return),
                float(gross_return - trade_cost),
                int(exit_position - entry_position + 1),
            )
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "signal_time",
            "entry_time",
            "exit_time",
            "direction",
            "entry_price",
            "exit_price",
            "outcome",
            "gross_return",
            "net_return",
            "holding_bars",
        ],
    )


def _plot_strategy(
    signal_frame: pd.DataFrame,
    raw_df: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: Path,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
) -> None:
    if trades.empty:
        raise ValueError("The random segment contains no Bear-only/Bull-only trades.")

    display_end = max(
        signal_frame.index[-1], pd.Timestamp(trades["exit_time"].max())
    )
    price = raw_df.loc[signal_frame.index[0] : display_end, "close"]
    ordered = trades.sort_values(["exit_time", "entry_time"]).copy()
    ordered["cumulative_gross"] = ordered["gross_return"].cumsum()
    ordered["cumulative_net"] = ordered["net_return"].cumsum()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(21, 10),
        gridspec_kw={"height_ratios": [2.1, 1.0]},
        sharex=True,
        constrained_layout=True,
    )
    price_ax, equity_ax = axes
    price_ax.plot(price.index, price, color="#111827", linewidth=1.0, label="Close")

    for direction, color, marker, label in (
        ("long", "#16a34a", "^", "Long entry (Bull only)"),
        ("short", "#dc2626", "v", "Short entry (Bear only)"),
    ):
        group = trades[trades["direction"].eq(direction)]
        price_ax.scatter(
            group["entry_time"],
            group["entry_price"],
            s=18,
            marker=marker,
            color=color,
            alpha=0.7,
            label=label,
            zorder=3,
        )
    wins = trades[trades["gross_return"].gt(0.0)]
    losses = trades[trades["gross_return"].le(0.0)]
    price_ax.scatter(
        wins["exit_time"], wins["exit_price"], s=12, marker="o",
        color="#2563eb", alpha=0.6, label="TP exit", zorder=3,
    )
    price_ax.scatter(
        losses["exit_time"], losses["exit_price"], s=12, marker="x",
        color="#7c3aed", alpha=0.65, label="SL/end exit", zorder=3,
    )
    price_ax.set_ylabel("BTCUSDT")
    price_ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.65)
    price_ax.legend(loc="upper left", ncol=5, fontsize=8, framealpha=0.9)

    equity_ax.step(
        ordered["exit_time"], ordered["cumulative_gross"] * 100.0,
        where="post", color="#2563eb", linewidth=1.2, label="Cumulative gross",
    )
    equity_ax.step(
        ordered["exit_time"], ordered["cumulative_net"] * 100.0,
        where="post", color="#111827", linewidth=1.35, label="Cumulative net",
    )
    equity_ax.axhline(0.0, color="#64748b", linewidth=0.8)
    equity_ax.set_ylabel("Cumulative return (%)")
    equity_ax.set_xlabel("Exit time")
    equity_ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.65)
    equity_ax.legend(loc="upper left", framealpha=0.9)
    equity_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    gross_mean = float(trades["gross_return"].mean())
    net_mean = float(trades["net_return"].mean())
    win_rate = float(trades["gross_return"].gt(0.0).mean())
    long_count = int(trades["direction"].eq("long").sum())
    short_count = int(trades["direction"].eq("short").sum())
    fig.suptitle(
        "Full Final Val + Test | Bear-only -> Short | Bull-only -> Long | "
        "next-bar open entry\n"
        f"TP={take_profit:.2%} | SL={stop_loss:.2%} | cost={trade_cost:.3%} | "
        f"same candle={same_candle_policy} | independent overlapping trades\n"
        f"n={len(trades):,} (Long={long_count:,}, Short={short_count:,}) | "
        f"win={win_rate:.2%} | gross mean={gross_mean:.3%} | net mean={net_mean:.3%}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top-fraction must be in (0, 1]")

    bear_path = Path(args.bear_archive)
    bull_path = Path(args.bull_archive)
    data_path = Path(args.data)
    bear_spec = _make_spec(bear_path, args.bear_rank, "bear", args.top_fraction)
    bull_spec = _make_spec(bull_path, args.bull_rank, "bull", args.top_fraction)
    bear_horizons = _archive_horizons(bear_path, fallback=[1], label="Bear archive")
    bull_horizons = _archive_horizons(bull_path, fallback=[1], label="Bull archive")
    all_horizons = sorted(set(bear_horizons + bull_horizons))
    purge_bars = config.purge_bars_for_horizons(all_horizons)

    raw_df = load_ohlcv(data_path)
    bear_entry = _load_rank_entry(bear_path, bear_spec.rank)
    bull_entry = _load_rank_entry(bull_path, bull_spec.rank)

    def cache_path_for(spec: ModelSpec, horizons: list[int]) -> Path:
        return _trained_bundle_cache_path(
            cache_dir=Path(args.model_cache_dir),
            archive_path=spec.archive_path,
            data_path=data_path,
            raw_df=raw_df,
            spec=spec,
            horizons=horizons,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            purge_bars=purge_bars,
        )

    bear_cached = None
    bull_cached = None
    if not args.no_model_cache and not args.rebuild_model_cache:
        bear_cached = _load_trained_bundle_cache(
            cache_path_for(bear_spec, bear_horizons)
        )
        bull_cached = _load_trained_bundle_cache(
            cache_path_for(bull_spec, bull_horizons)
        )

    feature_space = None
    if bear_cached is None or bull_cached is None:
        quality_index = _quality_train_index(
            raw_df=raw_df,
            spec=bear_spec,
            horizons=all_horizons,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            purge_bars=purge_bars,
        )
        feature_space = _cached_feature_space(
            raw_df=raw_df,
            data_path=data_path,
            required_windows=_required_windows_for_entries([bear_entry, bull_entry]),
            quality_index=quality_index,
        )
    bear_bundle = _train_or_load_bundle(
        spec=bear_spec,
        entry=bear_entry,
        horizons=bear_horizons,
        raw_df=raw_df,
        data_path=data_path,
        feature_space=feature_space,
        args=args,
        purge_bars=purge_bars,
        preloaded_bundle=bear_cached,
    )
    bull_bundle = _train_or_load_bundle(
        spec=bull_spec,
        entry=bull_entry,
        horizons=bull_horizons,
        raw_df=raw_df,
        data_path=data_path,
        feature_space=feature_space,
        args=args,
        purge_bars=purge_bars,
        preloaded_bundle=bull_cached,
    )

    split_frames = {
        "val": _aligned_split_frame("val", bear_bundle.val, bull_bundle.val, raw_df),
        "test": _aligned_split_frame(
            "test", bear_bundle.test, bull_bundle.test, raw_df
        ),
    }
    sample = _choose_sample(split_frames, args.sample_bars, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = str(sample["split"].iloc[0])
    start_tag = sample.index[0].strftime("%Y%m%d_%H%M")
    end_tag = sample.index[-1].strftime("%Y%m%d_%H%M")
    output_path = out_dir / (
        f"btc_5m_bear_bull_models_top{args.top_fraction * 100:.0f}_"
        f"{split}_{start_tag}_{end_tag}.png"
    )
    strategy_path = out_dir / (
        f"btc_5m_bear_bull_strategy_top{args.top_fraction * 100:.0f}_"
        f"tp{args.take_profit * 100:.3f}_sl{args.stop_loss * 100:.3f}_"
        "full_val_test.png"
    )
    _plot_sample(
        sample=sample,
        output_path=output_path,
        bars_per_panel=args.bars_per_panel,
        bear_threshold=bear_bundle.val.pred_threshold,
        bull_threshold=bull_bundle.val.pred_threshold,
        top_fraction=args.top_fraction,
    )
    strategy_signals = (
        pd.concat([split_frames["val"], split_frames["test"]], axis=0)
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
        .sort_index()
    )
    trades = _simulate_directional_trades(
        sample=strategy_signals,
        raw_df=raw_df,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        trade_cost=args.trade_cost,
        same_candle_policy=args.same_candle_policy,
    )
    _plot_strategy(
        signal_frame=strategy_signals,
        raw_df=raw_df,
        trades=trades,
        output_path=strategy_path,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        trade_cost=args.trade_cost,
        same_candle_policy=args.same_candle_policy,
    )

    counts = sample["state"].value_counts().reindex(range(4), fill_value=0)
    print(
        f"Random {split.upper()} sample: {sample.index[0]} -> {sample.index[-1]} "
        f"({len(sample):,} bars)"
    )
    print(
        f"Top fraction: {args.top_fraction:.2%} | "
        f"Bear Val cutoff: {bear_bundle.val.pred_threshold:.8f} | "
        f"Bull Val cutoff: {bull_bundle.val.pred_threshold:.8f}"
    )
    for state in range(4):
        print(
            f"  {STATE_NAMES[state]}: {counts[state]:,} "
            f"({counts[state] / len(sample):.2%})"
        )
    print(f"Saved image: {output_path}")
    print(
        f"Strategy trades: {len(trades):,} | "
        f"gross mean={trades['gross_return'].mean():.4%} | "
        f"net mean={trades['net_return'].mean():.4%} | "
        f"win rate={trades['gross_return'].gt(0.0).mean():.2%}"
    )
    print(f"Saved strategy image: {strategy_path}")


if __name__ == "__main__":
    main()
