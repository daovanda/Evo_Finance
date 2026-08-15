"""Plot Rank-1 MA3 slope-reversal selections on BTCUSDT 5-minute candles.

The model cutoff is fitted on Final Val and applied unchanged to Test, matching
``crypto.backtest``. Only selected rows whose observable MA3 slope is positive
are highlighted. Highlighting is done on signal candle ``t``; the corresponding
trade entry candle would be ``H1 = t + 1``.

PowerShell:
    python -m temp.plot_5m_ma_slope_reversal_signals `
      --archive crypto/results/crypto_btc_5m_long_ma_slope_reversal_fs2_top20_seed1_resume_seed2_2h.json `
      --rank 1 `
      --top-fraction 0.10 `
      --panels 4 `
      --bars-per-panel 500 `
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import (
    BundleSignals,
    ModelSpec,
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
logger = logging.getLogger("temp.plot_5m_ma_slope_reversal_signals")

DEFAULT_ARCHIVE = Path(
    "crypto/results/"
    "crypto_btc_5m_long_ma_slope_reversal_fs2_top20_seed1_resume_seed2_2h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_CACHE_DIR = Path("temp/model")


def _metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid archive metadata: {path}")
    if config.canonical_label_mode(metadata.get("label_mode")) != "ma_slope_reversal":
        raise ValueError("Archive must use label_mode=ma_slope_reversal.")
    return metadata


def _split_value(metadata: dict[str, object], key: str, fallback: object) -> object:
    policy = metadata.get("split_policy", {})
    if not isinstance(policy, dict):
        return fallback
    value = policy.get(key)
    return fallback if value is None else value


def _train_or_load(
    *,
    args: argparse.Namespace,
    raw: pd.DataFrame,
    metadata: dict[str, object],
    spec: ModelSpec,
    entry: dict[str, object],
    horizons: list[int],
) -> BundleSignals:
    val_start = str(_split_value(metadata, "val_start", config.VAL_START))
    test_start = str(_split_value(metadata, "test_start", config.TEST_START))
    test_end = _split_value(metadata, "test_end", config.TEST_END)
    purge_bars = config.purge_bars_for_horizons(horizons)
    cache_path = _trained_bundle_cache_path(
        cache_dir=Path(args.model_cache_dir),
        archive_path=spec.archive_path,
        data_path=Path(args.data),
        raw_df=raw,
        spec=spec,
        horizons=horizons,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    bundle = None
    if not args.no_model_cache and not args.rebuild_model_cache:
        bundle = _load_trained_bundle_cache(cache_path)
    if bundle is None:
        quality_index = _quality_train_index(
            raw_df=raw,
            spec=spec,
            horizons=horizons,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
        )
        feature_space = _cached_feature_space(
            raw_df=raw,
            data_path=Path(args.data),
            required_windows=_required_windows_for_entries([entry]),
            quality_index=quality_index,
        )
        bundle = _train_spec_bundle(
            spec=spec,
            entry=entry,
            raw_df=raw,
            feature_space=feature_space,
            horizons=horizons,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
        )
        if not args.no_model_cache:
            _save_trained_bundle_cache(cache_path, bundle)
            logger.info("Saved model prediction cache: %s", cache_path)
    else:
        logger.info("Loaded model prediction cache: %s", cache_path)
    return _bundle_with_top_fraction(bundle, float(args.top_fraction))


def _ma3_slope_positive(raw: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(raw["close"], errors="coerce")
    ma3 = close.rolling(int(config.MA_SLOPE_FAST_WINDOW)).mean()
    slope = ma3 - ma3.shift(int(config.MA_SLOPE_FAST_SHIFT))
    return slope.gt(0.0).where(slope.notna())


def _split_frames(bundle: BundleSignals, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slope_positive = _ma3_slope_positive(raw)
    frames: dict[str, pd.DataFrame] = {}
    for split_name, signals in (("val", bundle.val), ("test", bundle.test)):
        index = signals.data.index.intersection(raw.index)
        frame = raw.reindex(index)[["open", "high", "low", "close"]].copy()
        frame["score"] = signals.data["pred"].reindex(index)
        frame["selected"] = frame.index.isin(signals.selected_index)
        frame["ma3_slope_positive"] = (
            slope_positive.reindex(index).fillna(False).astype(bool)
        )
        frame["highlight"] = frame["selected"] & frame["ma3_slope_positive"]
        frame["split"] = split_name
        frames[split_name] = frame.dropna(subset=["open", "high", "low", "close"])
    return frames


def _sample_panels(
    frames: dict[str, pd.DataFrame],
    *,
    panels: int,
    bars_per_panel: int,
    seed: int | None,
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, int]] = []
    for split_name, frame in frames.items():
        for position in np.flatnonzero(frame["highlight"].to_numpy(bool)):
            candidates.append((split_name, int(position)))
    if not candidates:
        raise ValueError("No top-fraction selections with MA3 slope > 0 were found.")
    replace = len(candidates) < panels
    picks = rng.choice(len(candidates), size=panels, replace=replace)
    samples: list[pd.DataFrame] = []
    for picked in np.atleast_1d(picks):
        split_name, center = candidates[int(picked)]
        frame = frames[split_name]
        left = max(0, min(center - bars_per_panel // 2, len(frame) - bars_per_panel))
        sample = frame.iloc[left : left + bars_per_panel].copy()
        if len(sample) < bars_per_panel:
            raise ValueError(f"Split {split_name} has fewer than {bars_per_panel} rows.")
        samples.append(sample)
    return samples


def _draw_candles(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x = mdates.date2num(pd.DatetimeIndex(frame.index).to_pydatetime())
    width = 4.0 / (24.0 * 60.0)
    for position, (_, row) in enumerate(frame.iterrows()):
        highlighted = bool(row["highlight"])
        face = "#facc15" if highlighted else "#ffffff"
        edge = "#a16207" if highlighted else "#475569"
        ax.vlines(x[position], row["low"], row["high"], color=edge, linewidth=0.65)
        lower = min(float(row["open"]), float(row["close"]))
        height = abs(float(row["close"]) - float(row["open"]))
        if height == 0.0:
            ax.hlines(lower, x[position] - width / 2, x[position] + width / 2,
                      color=edge, linewidth=0.8)
        else:
            ax.add_patch(
                Rectangle(
                    (x[position] - width / 2, lower),
                    width,
                    height,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.65,
                )
            )
    ax.set_xlim(x[0] - width, x[-1] + width)
    ax.xaxis_date()


def _render(
    samples: list[pd.DataFrame],
    *,
    output: Path,
    rank: int,
    top_fraction: float,
    val_cutoff: float,
    test_cutoff: float,
) -> None:
    fig, axes = plt.subplots(
        len(samples),
        1,
        figsize=(22, max(4.0 * len(samples), 6.5)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for panel_number, (ax, frame) in enumerate(zip(axes, samples, strict=True), 1):
        _draw_candles(ax, frame)
        split = str(frame["split"].iloc[0]).upper()
        selected = int(frame["highlight"].sum())
        ax.set_title(
            f"Panel {panel_number} | {split} | {frame.index[0]} to {frame.index[-1]} | "
            f"yellow signals={selected}",
            fontsize=10,
        )
        ax.set_ylabel("BTCUSDT")
        ax.grid(color="#cbd5e1", linewidth=0.5, alpha=0.55)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.legend(
            handles=[
                Patch(facecolor="#ffffff", edgecolor="#475569", label="regular candle"),
                Patch(
                    facecolor="#facc15",
                    edgecolor="#a16207",
                    label="top selection + MA3 slope > 0",
                ),
            ],
            loc="upper left",
            fontsize=8,
        )
    axes[-1].set_xlabel("Signal candle time t")
    fig.suptitle(
        f"BTCUSDT 5m MA3 slope-reversal | Rank {rank} | top {top_fraction:.0%}\n"
        f"Val cutoff={val_cutoff:.6f} | Test applied cutoff={test_cutoff:.6f} | "
        "yellow marks signal candle t (entry H1=t+1)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--bars-per-panel", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-model-cache", action="store_true")
    parser.add_argument("--no-model-cache", action="store_true")
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive.")
    if not 0.0 < args.top_fraction <= 1.0:
        parser.error("--top-fraction must be in (0, 1].")
    if args.panels < 1 or args.bars_per_panel < 10:
        parser.error("--panels must be positive and --bars-per-panel at least 10.")
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
        top_fraction=args.top_fraction,
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
    frames = _split_frames(bundle, raw)
    samples = _sample_panels(
        frames,
        panels=args.panels,
        bars_per_panel=args.bars_per_panel,
        seed=args.seed,
    )
    top_tag = f"{100.0 * args.top_fraction:.0f}".zfill(2)
    output = args.out_dir / f"ma3_reversal_top{top_tag}_candles.png"
    _render(
        samples,
        output=output,
        rank=args.rank,
        top_fraction=args.top_fraction,
        val_cutoff=bundle.val.pred_threshold,
        test_cutoff=bundle.test.pred_threshold,
    )
    total_selected = sum(int(frame["highlight"].sum()) for frame in frames.values())
    print(
        f"Valid top selections (Val + Test, MA3 slope > 0): {total_selected:,}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
