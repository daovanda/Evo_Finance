"""Backtest a 5-minute Long MFE signal with fixed TP/SL through H3.

Flow:
1. Train one selected Long MFE archive rank.
2. Learn its top-fraction cutoff on Final Val and apply it unchanged to Test.
3. Without the 1m trigger filter, enter at open H1. With the filter enabled,
   enter exactly at the configured trigger touched during H1 minute 1. Since
   minute-1 OHLC has no intrabar ordering, TP/SL evaluation starts at minute 2.
   If minute 2 opens beyond SL, exit at that open instead of the better SL.
4. Keep one fixed TP and SL active through H1, H2, and H3.
5. If neither barrier is reached, exit at close H3.

When TP and SL are both reached in the same 5-minute candle, OHLC cannot reveal
their order. The default is conservative ``stop_first``; use
``--same-candle-policy tp_first`` for the optimistic sensitivity case.

The trained LightGBM model, Val/Test predictions, Val threshold, and selected
indices are cached under ``temp/model``. An unchanged model/data/config run
loads this cache before feature construction and training.

PowerShell:
    python -m temp.backtest_5m_long_mfe_fixed_tp_sl_h3 `
      --archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --rank 1 `
      --top-fraction 0.40 `
      --take-profit 0.001 `
      --stop-loss 0.001 `
      --trade-cost 0.00016 `
      --same-candle-policy stop_first `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output

One-minute trigger-entry variant:
    python -m temp.backtest_5m_long_mfe_fixed_tp_sl_h3 `
      --archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --rank 1 `
      --top-fraction 0.40 `
      --filter-take-profit 0.001 `
      --take-profit 0.001 `
      --stop-loss 0.001 `
      --trade-cost 0.00016 `
      --same-candle-policy tp_first `
      --next-1m-tp-filter `
      --data data/crypto/BTCUSDT_5m.csv `
      --data-1m data/crypto/BTCUSDT_1m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import (
    BundleSignals,
    ModelSpec,
    _archive_horizons,
    _cached_feature_space,
    _combine_horizons,
    _load_rank_entry,
    _quality_train_index,
    _split_signals,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_5m_long_mfe_fixed_tp_sl_h3")


DEFAULT_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_DATA_1M = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_MODEL_CACHE_DIR = Path("temp/model")
DEFAULT_RANK = 1
DEFAULT_TOP_FRACTION = 0.40
DEFAULT_TAKE_PROFIT = 0.001
DEFAULT_FILTER_TAKE_PROFIT: float | None = None
DEFAULT_STOP_LOSS = 0.001
DEFAULT_TRADE_COST = 0.00016
DEFAULT_OPT_TP_START = 0.00025
DEFAULT_OPT_TP_END = 0.005
DEFAULT_OPT_TP_STEP = 0.00025
DEFAULT_OPT_SL_START = 0.00025
DEFAULT_OPT_SL_END = 0.005
DEFAULT_OPT_SL_STEP = 0.00025


def load_archive_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Archive metadata must be an object: {path}")
    return dict(metadata)


def make_price_path(
    raw_df: pd.DataFrame,
    horizon: int,
    direction: str = "long",
) -> pd.DataFrame:
    entry_open = pd.to_numeric(raw_df["open"], errors="coerce").shift(-1)
    selected_direction = config.canonical_label_direction(direction)
    result = pd.DataFrame(index=raw_df.index)
    result["entry_open"] = entry_open
    for step in range(1, int(horizon) + 1):
        if selected_direction == "short":
            future_open = pd.to_numeric(
                raw_df["open"], errors="coerce"
            ).shift(-step)
            future_low = pd.to_numeric(
                raw_df["low"], errors="coerce"
            ).shift(-step)
            future_high = pd.to_numeric(
                raw_df["high"], errors="coerce"
            ).shift(-step)
            future_close = pd.to_numeric(
                raw_df["close"], errors="coerce"
            ).shift(-step)
            result[f"open_h{step}"] = 1.0 - future_open.div(entry_open)
            # Normalize Short so high_h is favorable and low_h is adverse.
            result[f"high_h{step}"] = 1.0 - future_low.div(entry_open)
            result[f"low_h{step}"] = 1.0 - future_high.div(entry_open)
            result[f"close_h{step}"] = 1.0 - future_close.div(entry_open)
        else:
            for column in ("open", "high", "low", "close"):
                price = pd.to_numeric(
                    raw_df[column], errors="coerce"
                ).shift(-step)
                result[f"{column}_h{step}"] = price.div(entry_open).sub(1.0)
    return result.replace([np.inf, -np.inf], np.nan)


def load_one_minute_ohlc(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    frame = frame.set_index("date").sort_index()
    frame = frame.loc[
        (frame.index >= pd.Timestamp(start))
        & (frame.index <= pd.Timestamp(end))
    ].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if frame.index.has_duplicates:
        frame = frame.loc[~frame.index.duplicated(keep="last")]
    return frame


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trained_bundle_cache_path(
    cache_dir: Path,
    archive_path: Path,
    data_path: Path,
    raw_df: pd.DataFrame,
    spec: ModelSpec,
    horizons: list[int],
    val_start: str,
    test_start: str,
    test_end: str | None,
    purge_bars: int,
    include_top_fraction: bool = False,
) -> Path:
    data_stat = data_path.stat()
    project_root = Path(__file__).resolve().parents[1]
    source_paths = [
        project_root / "crypto/config.py",
        project_root / "crypto/data.py",
        project_root / "crypto/analyze.py",
        project_root / "crypto/backtest.py",
    ]
    payload = {
        "schema": 1 if include_top_fraction else 2,
        "archive": str(archive_path.resolve()),
        "archive_sha256": _sha256_file(archive_path),
        "rank": int(spec.rank),
        "label_mode": spec.label_mode,
        "label_direction": spec.label_direction,
        "label_threshold": float(spec.label_threshold),
        "exit_after_k": spec.exit_after_k,
        "horizons": [int(value) for value in horizons],
        "data": str(data_path.resolve()),
        "data_size": int(data_stat.st_size),
        "data_mtime_ns": int(data_stat.st_mtime_ns),
        "data_rows": int(len(raw_df)),
        "data_start": str(raw_df.index.min()),
        "data_end": str(raw_df.index.max()),
        "val_start": str(val_start),
        "test_start": str(test_start),
        "test_end": None if test_end is None else str(test_end),
        "purge_bars": int(purge_bars),
        "training_sources": {
            str(path): _sha256_file(path)
            for path in source_paths
            if path.exists()
        },
    }
    if include_top_fraction:
        payload["top_fraction"] = float(spec.top_fraction)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:20]
    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in archive_path.stem
    )
    return cache_dir / f"{safe_stem}_r{spec.rank:02d}_{key}.pkl"


def _load_trained_bundle_cache(path: Path) -> BundleSignals | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception as exc:
        logger.warning("Ignoring unreadable model cache %s: %s", path, exc)
        return None
    if not isinstance(bundle, BundleSignals):
        logger.warning("Ignoring incompatible model cache: %s", path)
        return None
    logger.info("Loaded trained model signals from cache: %s", path)
    return bundle


def _save_trained_bundle_cache(path: Path, bundle: BundleSignals) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    logger.info("Saved trained model signals to cache: %s", path)


def _bundle_with_top_fraction(
    bundle: BundleSignals,
    top_fraction: float,
) -> BundleSignals:
    fraction = float(top_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")

    source_val_horizons = bundle.val_horizons or (bundle.val,)
    source_test_horizons = bundle.test_horizons or (bundle.test,)
    if len(source_val_horizons) != len(source_test_horizons):
        raise ValueError("Cached Val/Test horizon counts do not match.")

    val_horizons = []
    test_horizons = []
    for val_source, test_source in zip(
        source_val_horizons,
        source_test_horizons,
        strict=True,
    ):
        val = _split_signals(
            split="val",
            label=val_source.data["label"],
            pred=val_source.data["pred"],
            top_fraction=fraction,
        )
        test = _split_signals(
            split="test",
            label=test_source.data["label"],
            pred=test_source.data["pred"],
            top_fraction=fraction,
            pred_threshold=val.pred_threshold,
        )
        val_horizons.append(val)
        test_horizons.append(test)

    if len(val_horizons) == 1:
        val = val_horizons[0]
        test = test_horizons[0]
    else:
        val = _combine_horizons(
            split="val",
            split_results=val_horizons,
            top_fraction=fraction,
        )
        test = _combine_horizons(
            split="test",
            split_results=test_horizons,
            top_fraction=fraction,
        )

    logger.info(
        "Applied top fraction %.2f%% from cached predictions | "
        "val_threshold=%.8f | val_selected=%d | test_selected=%d",
        fraction * 100.0,
        val.pred_threshold,
        len(val.selected_index),
        len(test.selected_index),
    )
    return BundleSignals(
        label=bundle.label,
        val=val,
        test=test,
        val_horizons=tuple(val_horizons),
        test_horizons=tuple(test_horizons),
        models=bundle.models,
    )


def simulate_fixed_tp_sl(
    selected_path: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
) -> pd.DataFrame:
    """Simulate fixed barriers in chronological H1-H3 order."""
    minute_columns = [
        column
        for minute_offset in range(2, 6)
        for column in (f"high_m1_{minute_offset}", f"low_m1_{minute_offset}")
    ]
    minute_columns.append("open_m1_2")
    has_minute_path = all(column in selected_path.columns for column in minute_columns)
    required = ["high_h2", "low_h2", "high_h3", "low_h3", "close_h3"]
    required += minute_columns if has_minute_path else ["high_h1", "low_h1"]
    result = selected_path.dropna(subset=required).copy()
    tp = float(take_profit)
    sl = float(stop_loss)
    policy = str(same_candle_policy).strip().lower()
    if tp <= 0.0 or sl <= 0.0:
        raise ValueError("take_profit and stop_loss must be positive.")
    if policy not in {"stop_first", "tp_first"}:
        raise ValueError("same_candle_policy must be stop_first or tp_first.")

    n = len(result)
    active = np.ones(n, dtype=bool)
    gross = np.full(n, np.nan, dtype=float)
    exit_h = np.full(n, 3, dtype=int)
    outcome = np.full(n, "close_h3", dtype=object)

    execution_steps: list[tuple[str, str, str | None, int]] = []
    if has_minute_path:
        execution_steps.extend(
            (
                f"high_m1_{minute_offset}",
                f"low_m1_{minute_offset}",
                "open_m1_2" if minute_offset == 2 else None,
                1,
            )
            for minute_offset in range(2, 6)
        )
    else:
        execution_steps.append(("high_h1", "low_h1", None, 1))
    execution_steps.extend(
        [
            ("high_h2", "low_h2", None, 2),
            ("high_h3", "low_h3", None, 3),
        ]
    )

    for high_column, low_column, open_column, step in execution_steps:
        high = pd.to_numeric(result[high_column], errors="coerce").to_numpy()
        low = pd.to_numeric(result[low_column], errors="coerce").to_numpy()
        if open_column is not None:
            open_return = pd.to_numeric(
                result[open_column], errors="coerce"
            ).to_numpy()
            gap_sl_exit = active & (open_return <= -sl)
            gross[gap_sl_exit] = open_return[gap_sl_exit]
            exit_h[gap_sl_exit] = step
            outcome[gap_sl_exit] = "sl_open_m1_2"
            active[gap_sl_exit] = False
        tp_hit = high >= tp
        sl_hit = low <= -sl
        if policy == "stop_first":
            sl_exit = active & sl_hit
            tp_exit = active & ~sl_hit & tp_hit
        else:
            tp_exit = active & tp_hit
            sl_exit = active & ~tp_hit & sl_hit

        gross[sl_exit] = -sl
        exit_h[sl_exit] = step
        outcome[sl_exit] = f"sl_h{step}"
        active[sl_exit] = False

        gross[tp_exit] = tp
        exit_h[tp_exit] = step
        outcome[tp_exit] = f"tp_h{step}"
        active[tp_exit] = False

    close_h3 = pd.to_numeric(result["close_h3"], errors="coerce").to_numpy()
    gross[active] = close_h3[active]
    result["outcome"] = outcome
    result["exit_h"] = exit_h
    result["gross_return"] = gross
    result["net_return"] = gross - float(trade_cost)
    return result


def simulate_next_1m_tp_oracle(
    selected_path: pd.DataFrame,
    minute: pd.DataFrame,
    filter_take_profit: float,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
    direction: str = "long",
) -> pd.DataFrame:
    """Enter at the minute-1 trigger; execute barriers from minute 2."""
    filter_tp = float(filter_take_profit)
    tp = float(take_profit)
    sl = float(stop_loss)
    policy = str(same_candle_policy).strip().lower()
    if policy not in {"stop_first", "tp_first"}:
        raise ValueError("same_candle_policy must be stop_first or tp_first.")
    if filter_tp <= 0.0:
        raise ValueError("filter_take_profit must be positive.")

    signals = pd.DatetimeIndex(selected_path.index)
    entry_times = signals + pd.Timedelta(minutes=5)
    offsets = np.arange(5, dtype="timedelta64[m]")
    lookup_values = (
        entry_times.to_numpy(dtype="datetime64[ns]")[:, None] + offsets[None, :]
    )
    lookup_index = pd.DatetimeIndex(lookup_values.reshape(-1))
    aligned = minute.reindex(lookup_index)
    shape = (len(signals), 5)
    open_values = aligned["open"].to_numpy(dtype=float).reshape(shape)
    high_values = aligned["high"].to_numpy(dtype=float).reshape(shape)
    low_values = aligned["low"].to_numpy(dtype=float).reshape(shape)
    complete = (
        np.isfinite(open_values).all(axis=1)
        & np.isfinite(high_values).all(axis=1)
        & np.isfinite(low_values).all(axis=1)
    )
    if not bool(complete.all()):
        missing = list(signals[~complete][:5])
        raise ValueError(
            "Missing one or more 1m candles inside H1 for selected signals: "
            f"{missing}"
        )

    entry_open = pd.to_numeric(
        selected_path["entry_open"], errors="coerce"
    ).to_numpy(dtype=float)
    minute_open = open_values[:, 0]
    if not bool(
        np.isclose(entry_open, minute_open, rtol=1e-10, atol=1e-8).all()
    ):
        raise ValueError(
            "The first H1 1m open does not match the 5m H1 entry open."
        )

    is_short = config.canonical_label_direction(direction) == "short"
    if is_short:
        trigger_touch = 1.0 - low_values[:, 0] / entry_open
        trigger_factor = 1.0 - filter_tp
    else:
        trigger_touch = high_values[:, 0] / entry_open - 1.0
        trigger_factor = 1.0 + filter_tp
    keep = trigger_touch >= filter_tp
    result = selected_path.iloc[np.flatnonzero(keep)].copy()
    kept_open_h1 = entry_open[keep]
    trigger_entry = kept_open_h1 * trigger_factor

    result["open_h1_original"] = kept_open_h1
    result["trigger_entry"] = trigger_entry
    result["entry_open"] = trigger_entry

    # Rebase every future 5m return from open H1 to the executable trigger.
    for step in range(1, 4):
        for column in ("open", "high", "low", "close"):
            name = f"{column}_h{step}"
            if name not in result.columns:
                continue
            old_return = pd.to_numeric(result[name], errors="coerce")
            if is_short:
                result[name] = 1.0 - (1.0 - old_return) / trigger_factor
            else:
                result[name] = (1.0 + old_return) / trigger_factor - 1.0

    kept_high_values = high_values[keep]
    kept_low_values = low_values[keep]
    kept_open_values = open_values[keep]
    if is_short:
        minute_open_return = 1.0 - kept_open_values / trigger_entry[:, None]
        minute_high_return = 1.0 - kept_low_values / trigger_entry[:, None]
        minute_low_return = 1.0 - kept_high_values / trigger_entry[:, None]
    else:
        minute_open_return = kept_open_values / trigger_entry[:, None] - 1.0
        minute_high_return = kept_high_values / trigger_entry[:, None] - 1.0
        minute_low_return = kept_low_values / trigger_entry[:, None] - 1.0

    # Minute 1 establishes the trigger fill. Its OHLC cannot reveal whether an
    # adverse extreme happened before or after entry, so execution starts at
    # minute 2 rather than falsely treating a pre-trigger low/high as a stop.
    for minute_offset in range(5):
        number = minute_offset + 1
        result[f"open_m1_{number}"] = minute_open_return[:, minute_offset]
        result[f"high_m1_{number}"] = minute_high_return[:, minute_offset]
        result[f"low_m1_{number}"] = minute_low_return[:, minute_offset]
    result["high_h1"] = np.max(minute_high_return[:, 1:5], axis=1)
    result["low_h1"] = np.min(minute_low_return[:, 1:5], axis=1)

    return simulate_fixed_tp_sl(
        selected_path=result,
        take_profit=tp,
        stop_loss=sl,
        trade_cost=float(trade_cost),
        same_candle_policy=policy,
    )


def _parameter_grid(start: float, end: float, step: float) -> np.ndarray:
    start_value = float(start)
    end_value = float(end)
    step_value = float(step)
    if start_value <= 0.0 or end_value < start_value or step_value <= 0.0:
        raise ValueError(
            "Optimization grid requires 0 < start <= end and step > 0."
        )
    count = int(np.floor((end_value - start_value) / step_value + 1e-12))
    values = start_value + np.arange(count + 1, dtype=float) * step_value
    if values[-1] < end_value - 1e-12:
        values = np.append(values, end_value)
    return values


def optimize_tp_sl_on_val(
    val_execution_path: pd.DataFrame,
    trade_cost: float,
    same_candle_policy: str,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for take_profit in tp_values:
        for stop_loss in sl_values:
            simulation = simulate_fixed_tp_sl(
                selected_path=val_execution_path,
                take_profit=float(take_profit),
                stop_loss=float(stop_loss),
                trade_cost=float(trade_cost),
                same_candle_policy=same_candle_policy,
            )
            rows.append(
                {
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "gross_mean": float(simulation["gross_return"].mean()),
                    "net_mean": float(simulation["net_return"].mean()),
                    "win_rate": float(
                        (simulation["net_return"] > 0.0).mean()
                    ),
                }
            )
    sweep = pd.DataFrame(rows)
    if sweep.empty or not np.isfinite(sweep["net_mean"]).any():
        raise ValueError("TP/SL optimization produced no finite Val result.")
    best = sweep.sort_values(
        ["net_mean", "win_rate", "take_profit", "stop_loss"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    return (
        float(best["take_profit"]),
        float(best["stop_loss"]),
        sweep,
    )


def summarize_split(
    split: str,
    simulation: pd.DataFrame,
    available_rows: int,
    prediction_threshold: float,
    base_signal_count: int | None = None,
) -> dict[str, Any]:
    outcomes = simulation["outcome"].astype(str)
    n = len(simulation)
    tp_mask = outcomes.str.startswith("tp_")
    sl_mask = outcomes.str.startswith("sl_")
    close_mask = outcomes.eq("close_h3")
    min_high_h2_h3 = simulation[
        ["high_h2", "high_h3"]
    ].apply(pd.to_numeric, errors="coerce").min(axis=1, skipna=False)
    close_h3 = pd.to_numeric(simulation["close_h3"], errors="coerce")
    return {
        "split": split,
        "available_rows": int(available_rows),
        "signals": n,
        "base_signals": (
            int(base_signal_count) if base_signal_count is not None else n
        ),
        "oracle_keep_rate": (
            float(n / base_signal_count)
            if base_signal_count
            else (1.0 if n else 0.0)
        ),
        "selected_rate": float(n / available_rows) if available_rows else 0.0,
        "prediction_threshold": float(prediction_threshold),
        "tp_count": int(tp_mask.sum()),
        "tp_rate": float(tp_mask.mean()) if n else 0.0,
        "tp_min_high_h2_h3": (
            float(min_high_h2_h3.loc[tp_mask].min())
            if bool(tp_mask.any())
            else np.nan
        ),
        "tp_close_h3_mean": (
            float(close_h3.loc[tp_mask].mean())
            if bool(tp_mask.any())
            else np.nan
        ),
        "sl_count": int(sl_mask.sum()),
        "sl_rate": float(sl_mask.mean()) if n else 0.0,
        "close_h3_count": int(close_mask.sum()),
        "close_h3_rate": float(close_mask.mean()) if n else 0.0,
        "close_h3_mean": (
            float(simulation.loc[close_mask, "gross_return"].mean())
            if bool(close_mask.any())
            else np.nan
        ),
        "gross_mean": float(simulation["gross_return"].mean()) if n else 0.0,
        "net_mean": float(simulation["net_return"].mean()) if n else 0.0,
        "win_rate": float((simulation["net_return"] > 0.0).mean()) if n else 0.0,
    }


def main_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "split": row["split"],
                "signals": f"{int(row['signals']):,}",
                "base signals": f"{int(row['base_signals']):,}",
                "1m kept/base": f"{float(row['oracle_keep_rate']):.2%}",
                "selected": f"{float(row['selected_rate']):.2%}",
                "TP": (
                    f"{int(row['tp_count']):,} "
                    f"({float(row['tp_rate']):.2%})"
                ),
                "TP min(high H2,H3)": (
                    f"{float(row['tp_min_high_h2_h3']):+.3%}"
                ),
                "TP group mean close H3": (
                    f"{float(row['tp_close_h3_mean']):+.3%}"
                ),
                "SL": (
                    f"{int(row['sl_count']):,} "
                    f"({float(row['sl_rate']):.2%})"
                ),
                "close H3": (
                    f"{int(row['close_h3_count']):,} "
                    f"({float(row['close_h3_rate']):.2%}) "
                    f"mean={float(row['close_h3_mean']):+.3%}"
                ),
                "gross mean": f"{float(row['gross_mean']):+.3%}",
                "E[net]": f"{float(row['net_mean']):+.3%}",
                "win rate": f"{float(row['win_rate']):.2%}",
            }
        )
    return pd.DataFrame(rows)


def optimized_table(
    summary: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
) -> pd.DataFrame:
    frame = main_table(summary)
    frame.insert(1, "Val-opt TP", f"{float(take_profit):.3%}")
    frame.insert(2, "Val-opt SL", f"{float(stop_loss):.3%}")
    return frame


def draw_report(
    summary: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    optimized_summary: pd.DataFrame,
    optimized_simulations: dict[str, pd.DataFrame],
    optimized_take_profit: float,
    optimized_stop_loss: float,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(27, 21),
        gridspec_kw={"height_ratios": [1.0, 2.8, 1.0, 2.8, 1.5]},
        constrained_layout=True,
    )
    frame = main_table(summary)
    axes[0].axis("off")
    table = axes[0].table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
    axes[0].set_title("Fixed TP/SL strategy", fontsize=11, pad=8)

    colors = {"val": "#2563eb", "test": "#dc2626"}
    for split, frame in simulations.items():
        ordered = frame.sort_index()
        cumulative = ordered["net_return"].cumsum()
        axes[1].plot(
            ordered.index,
            cumulative * 100.0,
            color=colors[split],
            linewidth=1.1,
            label=(
                f"{split.upper()} n={len(ordered):,} | "
                f"end={float(cumulative.iloc[-1]) * 100.0:+.2f}%"
            ),
        )
    axes[1].axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    axes[1].set_title("Fixed TP/SL cumulative net return")
    axes[1].set_ylabel("Percentage points")
    axes[1].grid(True, alpha=0.5)
    axes[1].legend(frameon=False)

    optimized_frame = optimized_table(
        optimized_summary,
        optimized_take_profit,
        optimized_stop_loss,
    )
    axes[2].axis("off")
    optimized_artist = axes[2].table(
        cellText=optimized_frame.values,
        colLabels=optimized_frame.columns,
        cellLoc="center",
        loc="center",
    )
    optimized_artist.auto_set_font_size(False)
    optimized_artist.set_fontsize(7.2)
    optimized_artist.scale(1.0, 1.55)
    for (row, _), cell in optimized_artist.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
    axes[2].set_title(
        "Val-optimized TP/SL strategy (parameters applied unchanged to Test)",
        fontsize=11,
        pad=8,
    )

    for split, frame in optimized_simulations.items():
        ordered = frame.sort_index()
        cumulative = ordered["net_return"].cumsum()
        axes[3].plot(
            ordered.index,
            cumulative * 100.0,
            color=colors[split],
            linewidth=1.1,
            label=(
                f"{split.upper()} n={len(ordered):,} | "
                f"end={float(cumulative.iloc[-1]) * 100.0:+.2f}%"
            ),
        )
    axes[3].axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    axes[3].set_title(
        "Val-optimized TP/SL cumulative net return | "
        f"TP={optimized_take_profit:.3%}, SL={optimized_stop_loss:.3%}"
    )
    axes[3].set_ylabel("Percentage points")
    axes[3].grid(True, alpha=0.5)
    axes[3].legend(frameon=False)

    combined = pd.concat(simulations.values()).sort_index()
    daily = pd.Series(1, index=combined.index).resample("D").sum()
    axes[4].bar(daily.index, daily.to_numpy(), width=0.9, color="#f59e0b")
    axes[4].set_title("Signals per day")
    axes[4].set_ylabel("Signals")
    axes[4].grid(True, axis="y", alpha=0.5)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    archive_path = Path(args.archive)
    data_path = Path(args.data)
    filter_take_profit = (
        float(args.take_profit)
        if args.filter_take_profit is None
        else float(args.filter_take_profit)
    )
    metadata = load_archive_metadata(archive_path)
    mode = config.canonical_label_mode(metadata.get("label_mode"))
    direction = config.canonical_label_direction(
        metadata.get("label_direction")
    )
    expected_direction = config.canonical_label_direction(
        args.strategy_direction
    )
    horizons = _archive_horizons(
        archive_path,
        fallback=[3],
        label=f"5m {expected_direction.title()} MFE fixed TP/SL",
    )
    if mode != "mfe" or direction != expected_direction or horizons != [3]:
        raise ValueError(
            "Archive must use mode=mfe, "
            f"direction={expected_direction}, horizons=[3]; got "
            f"mode={mode}, direction={direction}, horizons={horizons}."
        )

    spec = ModelSpec(
        archive_path=archive_path,
        rank=int(args.rank),
        label_mode=mode,
        label_threshold=float(metadata["label_threshold"]),
        top_fraction=float(args.top_fraction),
        label_direction=direction,
    )
    raw_df = load_ohlcv(data_path)
    purge_bars = config.purge_bars_for_horizons(horizons)
    model_cache_path = _trained_bundle_cache_path(
        cache_dir=Path(args.model_cache_dir),
        archive_path=archive_path,
        data_path=data_path,
        raw_df=raw_df,
        spec=spec,
        horizons=horizons,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
    )
    legacy_model_cache_path = _trained_bundle_cache_path(
        cache_dir=Path(args.model_cache_dir),
        archive_path=archive_path,
        data_path=data_path,
        raw_df=raw_df,
        spec=spec,
        horizons=horizons,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=purge_bars,
        include_top_fraction=True,
    )
    bundle = None
    if not args.no_model_cache and not args.rebuild_model_cache:
        bundle = _load_trained_bundle_cache(model_cache_path)
        if bundle is None:
            bundle = _load_trained_bundle_cache(legacy_model_cache_path)
            if bundle is not None:
                _save_trained_bundle_cache(model_cache_path, bundle)
    if bundle is None:
        entry = _load_rank_entry(archive_path, spec.rank)
        quality_index = _quality_train_index(
            raw_df=raw_df,
            spec=spec,
            horizons=horizons,
            val_start=args.val_start,
            test_start=args.test_start,
            test_end=args.test_end,
            purge_bars=purge_bars,
        )
        feature_space = _cached_feature_space(
            raw_df=raw_df,
            data_path=data_path,
            required_windows=_required_windows_for_entries([entry]),
            quality_index=quality_index,
        )
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
            _save_trained_bundle_cache(model_cache_path, bundle)
    bundle = _bundle_with_top_fraction(bundle, spec.top_fraction)
    path = make_price_path(raw_df, horizon=3, direction=direction)
    minute: pd.DataFrame | None = None
    if args.next_1m_tp_filter:
        selected_indexes = [
            pd.Timestamp(index)
            for signals in (bundle.val, bundle.test)
            for index in signals.selected_index
        ]
        if selected_indexes:
            minute = load_one_minute_ohlc(
                args.data_1m,
                start=min(selected_indexes) + pd.Timedelta(minutes=5),
                end=max(selected_indexes) + pd.Timedelta(minutes=9),
            )

    simulations: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for split, signals in (("val", bundle.val), ("test", bundle.test)):
        selected_path = path.reindex(pd.Index(signals.selected_index))
        if args.next_1m_tp_filter:
            if minute is None:
                simulation = selected_path.iloc[0:0].copy()
                simulation["outcome"] = pd.Series(dtype=object)
                simulation["exit_h"] = pd.Series(dtype=int)
                simulation["gross_return"] = pd.Series(dtype=float)
                simulation["net_return"] = pd.Series(dtype=float)
            else:
                simulation = simulate_next_1m_tp_oracle(
                    selected_path=selected_path,
                    minute=minute,
                    filter_take_profit=filter_take_profit,
                    take_profit=float(args.take_profit),
                    stop_loss=float(args.stop_loss),
                    trade_cost=float(args.trade_cost),
                    same_candle_policy=args.same_candle_policy,
                    direction=direction,
                )
        else:
            simulation = simulate_fixed_tp_sl(
                selected_path=selected_path,
                take_profit=float(args.take_profit),
                stop_loss=float(args.stop_loss),
                trade_cost=float(args.trade_cost),
                same_candle_policy=args.same_candle_policy,
            )
        simulations[split] = simulation
        summary_rows.append(
            summarize_split(
                split=split,
                simulation=simulation,
                available_rows=len(signals.data),
                prediction_threshold=float(signals.pred_threshold),
                base_signal_count=len(selected_path),
            )
        )

    summary = pd.DataFrame(summary_rows)
    tp_values = _parameter_grid(
        args.optimize_tp_start,
        args.optimize_tp_end,
        args.optimize_tp_step,
    )
    sl_values = _parameter_grid(
        args.optimize_sl_start,
        args.optimize_sl_end,
        args.optimize_sl_step,
    )
    optimized_take_profit, optimized_stop_loss, _ = optimize_tp_sl_on_val(
        val_execution_path=simulations["val"],
        trade_cost=float(args.trade_cost),
        same_candle_policy=args.same_candle_policy,
        tp_values=tp_values,
        sl_values=sl_values,
    )
    logger.info(
        "Val-optimal TP/SL | TP=%.4f%% SL=%.4f%% | grid=%dx%d",
        optimized_take_profit * 100.0,
        optimized_stop_loss * 100.0,
        len(tp_values),
        len(sl_values),
    )
    optimized_simulations: dict[str, pd.DataFrame] = {}
    optimized_rows: list[dict[str, Any]] = []
    for split, signals in (("val", bundle.val), ("test", bundle.test)):
        optimized_simulation = simulate_fixed_tp_sl(
            selected_path=simulations[split],
            take_profit=optimized_take_profit,
            stop_loss=optimized_stop_loss,
            trade_cost=float(args.trade_cost),
            same_candle_policy=args.same_candle_policy,
        )
        optimized_simulations[split] = optimized_simulation
        optimized_rows.append(
            summarize_split(
                split=split,
                simulation=optimized_simulation,
                available_rows=len(signals.data),
                prediction_threshold=float(signals.pred_threshold),
                base_signal_count=len(signals.selected_index),
            )
        )
    optimized_summary = pd.DataFrame(optimized_rows)
    optimized_summary.attrs["take_profit"] = optimized_take_profit
    optimized_summary.attrs["stop_loss"] = optimized_stop_loss
    oracle_suffix = (
        f"_next1m_filter{filter_take_profit * 100.0:.3f}pct"
        if args.next_1m_tp_filter
        else ""
    )
    run_name = (
        f"{archive_path.stem}_r{spec.rank:02d}_top"
        f"{spec.top_fraction * 100.0:.0f}_fixed_tp"
        f"{float(args.take_profit) * 100.0:.3f}pct_sl"
        f"{float(args.stop_loss) * 100.0:.3f}pct_"
        f"{args.same_candle_policy}"
        f"{oracle_suffix}"
    ).replace(".", "p")
    output_path = Path(args.out_dir) / f"{run_name}.png"
    title = (
        f"5m {direction.title()} MFE H3 | fixed TP/SL through H1-H3 | "
        f"rank={spec.rank}, top={spec.top_fraction:.0%}, "
        f"TP=+{float(args.take_profit):.2%}, "
        f"SL=-{float(args.stop_loss):.2%}, "
        f"same-candle={args.same_candle_policy}, "
        f"cost={float(args.trade_cost):.3%}, "
        f"next-1m trigger entry={'ON' if args.next_1m_tp_filter else 'OFF'}"
        + (
            f", filter TP={filter_take_profit:.2%}"
            if args.next_1m_tp_filter
            else ""
        )
    )
    draw_report(
        summary=summary,
        simulations=simulations,
        optimized_summary=optimized_summary,
        optimized_simulations=optimized_simulations,
        optimized_take_profit=optimized_take_profit,
        optimized_stop_loss=optimized_stop_loss,
        output_path=output_path,
        title=title,
    )
    logger.info("Saved report: %s", output_path)
    return summary, optimized_summary, output_path


def parse_args(
    default_archive: Path = DEFAULT_ARCHIVE,
    default_direction: str = "long",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Backtest the 5-minute {default_direction.title()} MFE H3 "
            "strategy with a fixed execution table and Val-optimized TP/SL."
        )
    )
    parser.add_argument("--archive", default=str(default_archive))
    parser.add_argument(
        "--strategy-direction",
        choices=("long", "short"),
        default=config.canonical_label_direction(default_direction),
        help="Direction required from the archive and simulated by the strategy.",
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT)
    parser.add_argument(
        "--filter-take-profit",
        type=float,
        default=DEFAULT_FILTER_TAKE_PROFIT,
        help=(
            "Entry trigger distance from open H1, required during H1 minute "
            "1 when --next-1m-tp-filter is enabled."
        ),
    )
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS)
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
    parser.add_argument(
        "--optimize-tp-start",
        type=float,
        default=DEFAULT_OPT_TP_START,
    )
    parser.add_argument(
        "--optimize-tp-end",
        type=float,
        default=DEFAULT_OPT_TP_END,
    )
    parser.add_argument(
        "--optimize-tp-step",
        type=float,
        default=DEFAULT_OPT_TP_STEP,
    )
    parser.add_argument(
        "--optimize-sl-start",
        type=float,
        default=DEFAULT_OPT_SL_START,
    )
    parser.add_argument(
        "--optimize-sl-end",
        type=float,
        default=DEFAULT_OPT_SL_END,
    )
    parser.add_argument(
        "--optimize-sl-step",
        type=float,
        default=DEFAULT_OPT_SL_STEP,
    )
    parser.add_argument(
        "--same-candle-policy",
        choices=("stop_first", "tp_first"),
        default="stop_first",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--data-1m", default=str(DEFAULT_DATA_1M))
    parser.add_argument(
        "--next-1m-tp-filter",
        action="store_true",
        help=(
            "Require H1 minute 1 to touch the trigger, enter exactly at that "
            "trigger, then evaluate TP/SL from minute 2 through H3."
        ),
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
        help="Directory for persistent trained signal bundles.",
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Do not read or write the persistent trained-model cache.",
    )
    parser.add_argument(
        "--rebuild-model-cache",
        action="store_true",
        help="Retrain and overwrite the matching persistent cache entry.",
    )
    return parser.parse_args()


def main(
    default_archive: Path = DEFAULT_ARCHIVE,
    default_direction: str = "long",
) -> None:
    summary, optimized_summary, output_path = run(
        parse_args(
            default_archive=default_archive,
            default_direction=default_direction,
        )
    )
    print("\n=== Fixed TP/SL strategy ===")
    print(main_table(summary).to_string(index=False))
    optimized_tp = float(
        optimized_summary.attrs.get("take_profit", np.nan)
    )
    optimized_sl = float(
        optimized_summary.attrs.get("stop_loss", np.nan)
    )
    print("\n=== Val-optimized TP/SL strategy ===")
    print(
        optimized_table(
            optimized_summary,
            optimized_tp,
            optimized_sl,
        ).to_string(index=False)
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
