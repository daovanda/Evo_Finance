"""Production monitor for four Q20-MFE and three binary-MFE models.

The ``build`` command retrains archive Rank 1 models, calibrates 5%-wide score
bands on Validation, measures each band on Test, and saves a reusable bundle.
The ``run`` command refreshes closed 5m candles and reports the seven model
outputs for the next candle. It never places an order.

Examples (PowerShell)::

    python -m crypto.prod.long_mfe_7seed_monitor build
    python -m crypto.prod.long_mfe_7seed_monitor run --force-ipv4
    python -m crypto.prod.long_mfe_7seed_monitor run --force-ipv4 --loop
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from craw_btc import fetch_klines
from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.data import add_binary_labels, load_ohlcv, split_labeled_by_dates
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.prod.live_backend import (
    DEFAULT_BASE_URL,
    _feature_tail,
    _completed_candles_only,
    _prediction_score_band_index,
    _seconds_until_next_open,
    _timestamp_to_ms,
    fetch_current_open,
    force_ipv4_networking,
    refresh_local_ohlcv,
)
from crypto.prod.train_model import (
    _score_band_cutoffs,
    _train_booster as train_binary_booster,
)
from crypto.quantile_fitness import QuantileFitnessEvaluator


LOGGER = logging.getLogger("crypto.prod.long_mfe_7seed_monitor")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_DATA_1M = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_MODEL_DIR = Path("crypto/prod/model/long_mfe_q20_4seed_mfe_3seed")
DEFAULT_OUTPUT = Path("crypto/prod/live/latest_long_mfe_7seed.json")
HORIZON = 3
Q20_ALPHA = 0.20
BINARY_THRESHOLD = 0.001
DEFAULT_Q20_ARCHIVES = (
    Path("crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed1_3h.json"),
    Path("crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed2_3h.json"),
    Path("crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed3_3h.json"),
    Path("crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_w2_6_seed4_3h.json"),
)
DEFAULT_BINARY_ARCHIVES = (
    Path("crypto/results/crypto_btc_5m_long_mfe_h3_thr010_top60_seed1_3h.json"),
    Path("crypto/results/crypto_btc_5m_long_mfe_h3_thr010_top60_seed2_3h.json"),
    Path("crypto/results/crypto_btc_5m_long_mfe_h3_thr010_top60_seed3_3h.json"),
)


def build_bundle(
    data_path: Path,
    model_dir: Path,
    q20_archives: list[Path],
    binary_archives: list[Path],
    val_start: str,
    test_start: str,
    test_end: str | None,
) -> Path:
    """Train and persist all seven models plus out-of-sample band metrics."""
    if len(q20_archives) != 4 or len(binary_archives) != 3:
        raise ValueError("Exactly four Q20 archives and three binary MFE archives are required.")
    archives = q20_archives + binary_archives
    for path in archives:
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {path}")

    entries = [_load_rank_one(path) for path in archives]
    windows = _required_windows_for_entries(entries)
    raw = load_ohlcv(data_path)
    purge = config.purge_bars_for_horizons([HORIZON])

    quantile_labeled = add_binary_labels(
        raw,
        horizons=[HORIZON],
        label_mode="quantile_trade",
        label_direction="Long",
    )
    binary_labeled = add_binary_labels(
        raw,
        horizons=[HORIZON],
        threshold=BINARY_THRESHOLD,
        label_mode="mfe",
        label_direction="Long",
    )
    q_train, q_val, q_test = split_labeled_by_dates(
        quantile_labeled, val_start, test_start, test_end, purge
    )
    b_train, b_val, b_test = split_labeled_by_dates(
        binary_labeled, val_start, test_start, test_end, purge
    )
    LOGGER.info(
        "Split train=%d val=%d test=%d | feature windows=%s",
        len(q_train), len(q_val), len(q_test), windows,
    )
    features = build_feature_frame(raw, windows=windows, quality_index=q_train.index)
    space = CryptoFeatureSpace(features, selectable_features(features))
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "pipeline": "long_mfe_7seed_monitor",
        "manifest_version": 1,
        "data": str(data_path),
        "horizon": HORIZON,
        "feature_windows": windows,
        "split_policy": {
            "val_start": val_start,
            "test_start": test_start,
            "test_end": test_end,
            "purge_bars": purge,
        },
        "precision_definitions": {
            "binary_mfe": "actual MFE H3 > 0.10%",
        },
        "q20_models": [],
        "binary_models": [],
    }

    q_target = f"quantile_up_mfe_h{HORIZON}"
    evaluator = QuantileFitnessEvaluator(
        horizons=[HORIZON], target="mfe", quantile=Q20_ALPHA
    )
    for seed, (archive, entry) in enumerate(zip(q20_archives, entries[:4]), start=1):
        formulas = _clean_features(entry)
        train = q_train.dropna(subset=[q_target])
        booster = evaluator._train_booster(
            space.matrix(formulas, train.index),
            train[q_target].astype(float),
            horizon=HORIZON,
        )
        model_name = f"q20_seed{seed}_rank1.txt"
        booster.save_model(str(model_dir / model_name))
        manifest["q20_models"].append({
            "seed": seed,
            "rank": 1,
            "archive": str(archive),
            "archive_score": entry.get("score"),
            "features": formulas,
            "model_path": model_name,
        })
        LOGGER.info("Saved Q20 seed%d: %s", seed, model_name)

    label_col = f"label_h{HORIZON}"
    return_col = f"future_return_h{HORIZON}"
    for seed, (archive, entry) in enumerate(zip(binary_archives, entries[4:]), start=1):
        formulas = _clean_features(entry)
        train = b_train.dropna(subset=[label_col, return_col])
        val = b_val.dropna(subset=[label_col, return_col])
        test = b_test.dropna(subset=[label_col, return_col])
        X_train = space.matrix(formulas, train.index)
        X_val = space.matrix(formulas, val.index)
        booster = train_binary_booster(
            X_train, train[label_col].astype(int), X_val, val[label_col].astype(int)
        )
        val_pred = pd.Series(booster.predict(X_val), index=val.index)
        test_pred = pd.Series(booster.predict(space.matrix(formulas, test.index)), index=test.index)
        cutoffs = _score_cutoffs(val_pred)
        metrics = _band_metrics(
            test_pred,
            success=test[label_col].astype(bool),
            cutoffs=cutoffs,
        )
        model_name = f"mfe_thr010_seed{seed}_rank1.txt"
        booster.save_model(str(model_dir / model_name))
        manifest["binary_models"].append(
            _model_record(seed, archive, entry, formulas, model_name, cutoffs, metrics)
        )
        LOGGER.info("Saved binary MFE seed%d: %s", seed, model_name)

    path = model_dir / "manifest.json"
    _write_json(path, manifest)
    LOGGER.info("Saved production bundle: %s", path)
    return path


def run_once(
    data_path: Path,
    data_1m_path: Path,
    model_dir: Path,
    output_path: Path,
    symbol: str,
    interval: str,
    base_url: str,
    crawl_lookback_days: float,
    feature_lookback_bars: int,
    refresh: bool,
) -> dict[str, Any]:
    if str(interval).strip().lower() != "5m":
        raise ValueError("This model bundle requires --interval 5m.")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    update_info = None
    if refresh:
        _, update_info = refresh_local_ohlcv(
            data_1m_path, symbol, "1m", base_url, crawl_lookback_days
        )
    raw_1m = load_ohlcv(data_1m_path)
    current_open = fetch_current_open(symbol, "1m", base_url) if refresh else None
    if current_open is None:
        inference_time = pd.Timestamp(raw_1m.index[-1]) + pd.Timedelta(minutes=1)
        backfill_rows = 0
    else:
        inference_time = pd.Timestamp(current_open.candle_time)
        raw_1m, backfill_rows = _complete_minute_snapshot(
            raw_1m,
            inference_time=inference_time,
            symbol=symbol,
            base_url=base_url,
        )
        if isinstance(update_info, dict):
            update_info["in_memory_backfill_rows"] = int(backfill_rows)
    phase = int(inference_time.minute % 5)
    minute_tail_rows = max(int(feature_lookback_bars) * 5 + 10, 100)
    minute_tail = raw_1m.tail(minute_tail_rows)
    raw = aggregate_rolling_5m(minute_tail, phase=phase)
    if raw.empty or pd.Timestamp(raw.index[-1]) != inference_time:
        last_rolling = pd.Timestamp(raw.index[-1]) if len(raw) else None
        raise ValueError(
            "Latest completed 1m candles do not form a complete rolling 5m bar: "
            f"expected={inference_time}, actual={last_rolling}."
        )
    signal_time = inference_time
    records = list(manifest["q20_models"]) + list(manifest["binary_models"])
    all_features = list(dict.fromkeys(f for record in records for f in record["features"]))
    tail = _feature_tail(raw, feature_lookback_bars)
    feature_frame = build_feature_frame(
        tail,
        windows=manifest["feature_windows"],
        quality_filter=False,
    )
    space = CryptoFeatureSpace(feature_frame, selectable_features(feature_frame))

    q20_results = [_live_q20_prediction(record, model_dir, space, signal_time) for record in manifest["q20_models"]]
    binary_results = [_live_prediction(record, model_dir, space, signal_time) for record in manifest["binary_models"]]
    q_values = [row["prediction"] for row in q20_results]
    b_values = [row["prediction"] for row in binary_results]
    payload = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "monitor_only": True,
        "execution_enabled": False,
        "symbol": symbol.upper(),
        "interval": interval,
        "source_interval": "1m",
        "training_data": str(data_path),
        "rolling_data": str(data_1m_path),
        "rolling_phase": phase,
        "rolling_phase_kind": "aligned" if phase == 0 else "off_boundary",
        "signal_time": signal_time.isoformat(),
        "rolling_bar_start": (signal_time - pd.Timedelta(minutes=5)).isoformat(),
        "entry_candle_time": current_open.candle_time.isoformat() if current_open else None,
        "entry_open": current_open.open if current_open else None,
        "data_update": update_info,
        "feature_build": {
            "rolling_5m_rows_used": len(tail),
            "source_1m_rows_used": len(minute_tail),
            "features_used": len(all_features),
        },
        "q20_models": q20_results,
        "q20_ensemble": {
            "mean_prediction": float(np.mean(q_values)),
            "min_prediction": float(np.min(q_values)),
            "mean_prediction_pct": float(np.mean(q_values) * 100.0),
            "min_prediction_pct": float(np.min(q_values) * 100.0),
        },
        "binary_models": binary_results,
        "binary_ensemble": {
            "mean_probability": float(np.mean(b_values)),
            "mean_probability_pct": float(np.mean(b_values) * 100.0),
        },
    }
    _write_json(output_path, payload)
    _print_report(payload)
    return payload


def _complete_minute_snapshot(
    raw_1m: pd.DataFrame,
    inference_time: pd.Timestamp,
    symbol: str,
    base_url: str,
) -> tuple[pd.DataFrame, int]:
    """Backfill minute candles closed before a fixed live entry timestamp."""
    expected_last = pd.Timestamp(inference_time) - pd.Timedelta(minutes=1)
    last_local = pd.Timestamp(raw_1m.index[-1])
    if last_local < expected_last:
        fetched = fetch_klines(
            symbol=symbol,
            interval="1m",
            start_ms=_timestamp_to_ms(last_local + pd.Timedelta(minutes=1)),
            end_ms=_timestamp_to_ms(pd.Timestamp(inference_time) + pd.Timedelta(minutes=1)),
            base_url=base_url,
        )
        fetched = _completed_candles_only(fetched, interval="1m")
        fetched = fetched[pd.to_datetime(fetched["date"]) <= expected_last]
        fetched_rows = int(len(fetched))
        if fetched_rows:
            old = raw_1m.reset_index()
            combined = pd.concat([old, fetched], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            raw_1m = (
                combined.drop_duplicates("date", keep="last")
                .sort_values("date")
                .set_index("date")
            )
    else:
        fetched_rows = 0

    raw_1m = raw_1m.loc[raw_1m.index <= expected_last].copy()
    actual_last = pd.Timestamp(raw_1m.index[-1]) if len(raw_1m) else None
    if actual_last != expected_last:
        raise ValueError(
            "Cannot construct a causal 1m snapshot through the minute before entry: "
            f"expected_last={expected_last}, actual_last={actual_last}."
        )
    return raw_1m, fetched_rows


def aggregate_rolling_5m(raw_1m: pd.DataFrame, phase: int) -> pd.DataFrame:
    """Build complete causal 5m candles for one minute phase.

    Output timestamps are availability times. For example, a source block
    ``[10:02, 10:07)`` is indexed at ``10:07`` because it can only be used
    after the 10:06 source candle has closed.
    """
    phase = int(phase)
    if phase not in range(5):
        raise ValueError("rolling 5m phase must be one of 0, 1, 2, 3, 4.")
    data = raw_1m.sort_index()
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trade_count": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
    }
    if "is_trading_day" in data.columns:
        aggregation["is_trading_day"] = "max"
    missing = sorted(set(aggregation) - set(data.columns))
    if missing:
        raise ValueError(f"1m data missing rolling aggregation columns: {missing}")

    grouped = data.resample(
        "5min",
        origin="epoch",
        offset=f"{phase}min",
        label="left",
        closed="left",
    )
    bars = grouped.agg(aggregation)
    counts = grouped["close"].count()
    bars = bars.loc[counts.eq(5)].dropna(subset=["open", "high", "low", "close"])
    bars.index = bars.index + pd.Timedelta(minutes=5)
    bars.index.name = "date"
    return bars


def _live_prediction(
    record: dict[str, Any],
    model_dir: Path,
    space: CryptoFeatureSpace,
    signal_time: pd.Timestamp,
) -> dict[str, Any]:
    x = space.matrix(record["features"], pd.Index([signal_time]))
    booster = lgb.Booster(model_file=str(model_dir / record["model_path"]))
    pred = float(booster.predict(x)[0])
    band = _prediction_score_band_index(pred, record["val_score_band_cutoffs"])
    metrics = record["test_band_metrics"].get(f"q{band}", {}) if band else {}
    return {
        "seed": int(record["seed"]),
        "prediction": pred,
        "band": f"Q{band}" if band else None,
        "band_range": _band_range(band),
        "test_band_precision": metrics.get("precision"),
        "test_band_precision_pct": _pct(metrics.get("precision")),
        "test_band_samples": metrics.get("samples"),
        "test_cumulative_precision": metrics.get("cumulative_precision"),
        "test_cumulative_precision_pct": _pct(metrics.get("cumulative_precision")),
        "archive": record["archive"],
    }


def _live_q20_prediction(
    record: dict[str, Any],
    model_dir: Path,
    space: CryptoFeatureSpace,
    signal_time: pd.Timestamp,
) -> dict[str, Any]:
    x = space.matrix(record["features"], pd.Index([signal_time]))
    booster = lgb.Booster(model_file=str(model_dir / record["model_path"]))
    pred = float(booster.predict(x)[0])
    return {
        "seed": int(record["seed"]),
        "prediction": pred,
        "prediction_pct": pred * 100.0,
        "archive": record["archive"],
    }


def _score_cutoffs(pred: pd.Series) -> dict[str, float]:
    raw = _score_band_cutoffs(pred)
    if any(value is None for value in raw.values()):
        raise ValueError("Cannot calibrate score bands from empty Validation predictions.")
    return {name: float(value) for name, value in raw.items() if value is not None}


def _band_metrics(
    pred: pd.Series,
    success: pd.Series,
    cutoffs: dict[str, float],
) -> dict[str, dict[str, Any]]:
    band_ids = pred.map(lambda value: _prediction_score_band_index(float(value), cutoffs))
    result: dict[str, dict[str, Any]] = {}
    for band in range(1, 21):
        mask = band_ids.eq(band)
        cumulative = band_ids.le(band)
        row: dict[str, Any] = {
            "range": _band_range(band),
            "samples": int(mask.sum()),
            "precision": _mean_or_none(success[mask]),
            "cumulative_samples": int(cumulative.sum()),
            "cumulative_precision": _mean_or_none(success[cumulative]),
        }
        result[f"q{band}"] = row
    return result


def _model_record(
    seed: int,
    archive: Path,
    entry: dict[str, Any],
    features: list[str],
    model_name: str,
    cutoffs: dict[str, float],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "rank": 1,
        "archive": str(archive),
        "archive_score": entry.get("score"),
        "features": features,
        "model_path": model_name,
        "val_score_band_cutoffs": cutoffs,
        "test_band_metrics": metrics,
    }


def _load_rank_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    match = next((dict(row) for row in entries if int(row.get("rank", 0)) == 1), None)
    if match is None:
        raise ValueError(f"Rank 1 not found in {path}")
    return match


def _clean_features(entry: dict[str, Any]) -> list[str]:
    features = [str(value).strip() for value in entry.get("features", [])]
    features = list(dict.fromkeys(value for value in features if value))
    if not features:
        raise ValueError("Rank 1 has no usable features.")
    return features


def _band_range(band: int | None) -> str | None:
    if band is None:
        return None
    return f"{(band - 1) * 5}%-{band * 5}%"


def _mean_or_none(values: pd.Series) -> float | None:
    return float(values.astype(float).mean()) if len(values) else None


def _pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _print_report(payload: dict[str, Any]) -> None:
    print(f"\nSignal candle: {payload['signal_time']}")
    print(
        f"Rolling 5m  : phase={payload['rolling_phase']} "
        f"({payload['rolling_phase_kind']}) | bar={payload['rolling_bar_start']} -> "
        f"{payload['signal_time']}"
    )
    if payload.get("entry_candle_time"):
        print(f"Entry candle : {payload['entry_candle_time']} | open={payload['entry_open']:.2f}")
    print("\nQ20 MFE models (prediction is dynamic TP):")
    print("seed  prediction")
    for row in payload["q20_models"]:
        print(f"{row['seed']:>4}  {row['prediction'] * 100:>9.4f}%")
    q = payload["q20_ensemble"]
    print(f"Q20 ensemble: mean={q['mean_prediction_pct']:.4f}% | min={q['min_prediction_pct']:.4f}%")

    print("\nBinary MFE +0.10% models (prediction is probability):")
    print("seed  probability  band       Test precision   Test n   cumulative")
    for row in payload["binary_models"]:
        print(
            f"{row['seed']:>4}  {row['prediction'] * 100:>9.2f}%  "
            f"{row['band_range']:>9}  {_fmt_pct(row['test_band_precision_pct']):>14}  "
            f"{row['test_band_samples']:>7}  {_fmt_pct(row['test_cumulative_precision_pct']):>10}"
        )
    print(f"Binary ensemble mean probability: {payload['binary_ensemble']['mean_probability_pct']:.2f}%")


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}%"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Train/cache seven Rank 1 models and Test band precision.")
    build.add_argument("--data", type=Path, default=DEFAULT_DATA)
    build.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    build.add_argument("--q20-archives", nargs=4, type=Path, default=list(DEFAULT_Q20_ARCHIVES))
    build.add_argument("--binary-archives", nargs=3, type=Path, default=list(DEFAULT_BINARY_ARCHIVES))
    build.add_argument("--val-start", default=config.VAL_START)
    build.add_argument("--test-start", default=config.TEST_START)
    build.add_argument("--test-end", default=config.TEST_END)

    run = sub.add_parser("run", help="Show current seven-model monitor output; no order execution.")
    run.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run.add_argument("--data-1m", type=Path, default=DEFAULT_DATA_1M)
    run.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    run.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--symbol", default="BTCUSDT")
    run.add_argument("--interval", default="5m")
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--crawl-lookback-days", type=float, default=1.0)
    run.add_argument("--feature-lookback-bars", type=int, default=5000)
    run.add_argument("--no-refresh", action="store_true")
    run.add_argument("--force-ipv4", action="store_true")
    run.add_argument("--loop", action="store_true")
    run.add_argument("--sleep-after-open", type=float, default=5.0)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parser().parse_args()
    if args.command == "build":
        build_bundle(
            args.data,
            args.model_dir,
            list(args.q20_archives),
            list(args.binary_archives),
            args.val_start,
            args.test_start,
            args.test_end,
        )
        return
    if args.force_ipv4:
        force_ipv4_networking()
    while True:
        run_once(
            args.data,
            args.data_1m,
            args.model_dir,
            args.out,
            args.symbol,
            args.interval,
            args.base_url,
            args.crawl_lookback_days,
            args.feature_lookback_bars,
            not args.no_refresh,
        )
        if not args.loop:
            return
        sleep_seconds = _seconds_until_next_open("1m") + max(args.sleep_after_open, 0.0)
        LOGGER.info("Sleeping %.1fs until next closed-candle update.", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
