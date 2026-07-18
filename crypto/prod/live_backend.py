"""Live crypto backend: refresh data, run trained models, and write predictions.

Examples:
    python -m crypto.prod.live_backend --model-dir crypto/prod/model/crypto_btc_seed1_12h
    python -m crypto.prod.live_backend --model-dir crypto/prod/model/crypto_btc_seed1_12h --loop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from craw_btc import (
    DEFAULT_BASE_URL,
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    INTERVAL_TO_MS,
    OUTPUT_TIMEZONE,
    fetch_klines,
)
from crypto import config
from crypto.data import load_ohlcv
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.prod import trade_config
from crypto.prod.telegram_notify import send_telegram_message


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.prod.live_backend")


DEFAULT_OUTPUT_PATH = Path("crypto/prod/live/latest_prediction.json")
DEFAULT_CRAWL_LOOKBACK_DAYS = 1.0
DEFAULT_LIVE_FEATURE_LOOKBACK_BARS = 5000
_WINDOW_ARG_RE = re.compile(r",\s*(\d+)(?=\))")
_WINDOW_SUFFIX_RE = re.compile(r"_(\d+)\b")


@dataclass(frozen=True)
class CurrentOpen:
    candle_time: pd.Timestamp
    open: float
    fetched_at: str


def run_once(
    model_dir: str | Path,
    data_path: str | Path = config.DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    base_url: str = DEFAULT_BASE_URL,
    crawl_lookback_days: float = DEFAULT_CRAWL_LOOKBACK_DAYS,
    feature_lookback_bars: int = DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    manifest = _load_manifest(model_dir)

    updated_df, update_info = refresh_local_ohlcv(
        data_path=data_path,
        symbol=symbol,
        interval=interval,
        base_url=base_url,
        crawl_lookback_days=crawl_lookback_days,
    )
    if updated_df.empty:
        raise ValueError("Cannot predict with empty OHLCV data.")

    raw_df = load_ohlcv(data_path)
    signal_time = pd.Timestamp(raw_df.index[-1])
    logger.info("Signal candle: %s", signal_time)

    model_features = _manifest_features(manifest)
    required_windows = _required_windows(model_features)
    feature_raw_df = _feature_tail(raw_df, feature_lookback_bars)
    logger.info(
        "Building live feature tail: rows=%d/%d | windows=%s",
        len(feature_raw_df),
        len(raw_df),
        required_windows,
    )
    feature_df = build_feature_frame(
        feature_raw_df,
        windows=required_windows,
        quality_filter=False,
    )
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)

    current_open = fetch_current_open(
        symbol=symbol,
        interval=interval,
        base_url=base_url,
    )
    if pd.Timestamp(current_open.candle_time) <= signal_time:
        message = (
            "Invalid live entry candle: entry_candle_time must be greater than "
            f"signal_time, got entry={current_open.candle_time}, signal={signal_time}."
        )
        logger.error(message)
        payload = _error_payload(
            message=message,
            symbol=symbol,
            interval=interval,
            data_path=data_path,
            model_dir=model_dir,
            signal_time=signal_time,
            current_open=current_open,
            update_info=update_info,
        )
        _write_payload(output_path, payload)
        _notify_prediction_once(payload)
        return payload

    entries = []
    for entry in manifest.get("entries", []):
        entries.append(
            _predict_entry(
                entry=entry,
                model_dir=model_dir,
                feature_space=feature_space,
                signal_time=signal_time,
            )
        )
    final_ensemble = _predict_final_ensemble(entries, manifest)

    payload = {
        "created_at": _utc_now_iso(),
        "symbol": symbol.upper(),
        "interval": interval,
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "signal_time": _ts_str(signal_time),
        "entry_candle_time": _ts_str(current_open.candle_time),
        "entry_open": current_open.open,
        "entry_open_fetched_at": current_open.fetched_at,
        "can_trade": True,
        "status": "OK",
        "error": None,
        "feature_build": {
            "rows_used": int(len(feature_raw_df)),
            "total_rows": int(len(raw_df)),
            "windows": required_windows,
            "lookback_bars": int(feature_lookback_bars),
        },
        "data_update": update_info,
        "entries": entries,
        "final_ensemble": final_ensemble,
    }
    _write_payload(output_path, payload)
    _notify_prediction_once(payload)
    return payload


def _error_payload(
    message: str,
    symbol: str,
    interval: str,
    data_path: str | Path,
    model_dir: str | Path,
    signal_time: pd.Timestamp,
    current_open: CurrentOpen | None,
    update_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "created_at": _utc_now_iso(),
        "symbol": symbol.upper(),
        "interval": interval,
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "signal_time": _ts_str(signal_time),
        "entry_candle_time": _ts_str(current_open.candle_time) if current_open else None,
        "entry_open": current_open.open if current_open else None,
        "entry_open_fetched_at": current_open.fetched_at if current_open else None,
        "can_trade": False,
        "status": "ERROR",
        "error": message,
        "data_update": update_info or {},
        "entries": [],
    }


def _write_payload(output_path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, _json_safe(payload))
    logger.info("Saved live prediction: %s", output_path)


def _notify_prediction_once(payload: dict[str, Any]) -> None:
    key = _prediction_notify_key(payload)
    if not key:
        return

    state_path = trade_config.LIVE_NOTIFY_STATE_PATH
    state = _load_notify_state(state_path)
    if state.get("last_key") == key:
        return

    try:
        result = send_telegram_message(_prediction_telegram_message(payload))
        if result.get("ok"):
            _save_notify_state(
                state_path,
                {
                    "last_key": key,
                    "last_sent_at": _utc_now_iso(),
                    "last_signal_time": payload.get("signal_time"),
                    "last_status": _decision_status(payload),
                },
            )
        elif not result.get("skipped"):
            _save_notify_state(
                state_path,
                {
                    "last_key": state.get("last_key"),
                    "last_error_at": _utc_now_iso(),
                    "last_error": str(result),
                },
            )
    except Exception as exc:  # noqa: BLE001 - Telegram failure must not break prediction loop
        logger.warning("Telegram prediction notification failed: %s", exc)
        _save_notify_state(
            state_path,
            {
                "last_key": state.get("last_key"),
                "last_error_at": _utc_now_iso(),
                "last_error": str(exc),
            },
        )


def _prediction_notify_key(payload: dict[str, Any]) -> str | None:
    signal_time = str(payload.get("signal_time") or "")
    if not signal_time:
        return None
    return "|".join(
        [
            str(payload.get("symbol") or ""),
            str(payload.get("interval") or ""),
            signal_time,
            str(payload.get("entry_candle_time") or ""),
            str(payload.get("status") or ""),
            _decision_status(payload),
        ]
    )


def _prediction_telegram_message(payload: dict[str, Any]) -> str:
    status = _decision_status(payload)
    lines = [
        f"Evo Crypto Prediction | {status}",
        f"Dien giai: {_prediction_status_description(status)}",
        f"Symbol: {payload.get('symbol')} {payload.get('interval')}",
        f"Signal time: {payload.get('signal_time')}",
        f"Entry candle: {payload.get('entry_candle_time')}",
        f"Entry open: {_fmt_value(payload.get('entry_open'), 2)}",
        f"Status: {status}",
    ]
    if payload.get("error"):
        lines.append(f"Error: {payload.get('error')}")

    for entry in payload.get("entries", []):
        lines.append(
            f"Rank {entry.get('rank')} | direction={entry.get('label_direction')} "
            f"| ensemble={entry.get('ensemble_signal')} "
            f"| pred_mean={_fmt_value(entry.get('pred_mean'), 6)}"
        )
        lines.append("Horizon | Prediction | Threshold | Signal | Model")
        for pred in entry.get("predictions", []):
            lines.append(
                " | ".join(
                    [
                        f"h{pred.get('horizon')}",
                        _fmt_value(pred.get("pred"), 6),
                        _fmt_value(pred.get("threshold"), 6),
                        str(pred.get("is_signal")).lower(),
                        str(pred.get("model_path") or ""),
                    ]
                )
            )
    final_ensemble = payload.get("final_ensemble")
    if isinstance(final_ensemble, dict):
        lines.append(
            f"Final ensemble | signal={final_ensemble.get('ensemble_signal')} "
            f"| direction={final_ensemble.get('label_direction')} "
            f"| members={final_ensemble.get('member_count')} "
            f"| pred_mean={_fmt_value(final_ensemble.get('pred_mean'), 6)}"
        )
    return "\n".join(lines)


def _decision_status(payload: dict[str, Any]) -> str:
    if payload.get("status") == "ERROR" or payload.get("error"):
        return "ERROR"
    final_ensemble = payload.get("final_ensemble")
    if isinstance(final_ensemble, dict):
        signal = final_ensemble.get("ensemble_signal")
        if signal is True:
            return "TRADE"
        if signal is False:
            return "NO TRADE"
    has_trade = any(entry.get("ensemble_signal") is True for entry in payload.get("entries", []))
    return "TRADE" if has_trade else "NO TRADE"


def _prediction_status_description(status: str) -> str:
    descriptions = {
        "TRADE": "Co tin hieu ensemble dong thuan, trader se cho dung entry candle de xu ly.",
        "NO TRADE": "Chua co tin hieu ensemble dong thuan, trader se khong vao lenh.",
        "ERROR": "Backend gap loi khi tao prediction, khong nen vao lenh.",
    }
    return descriptions.get(str(status).upper(), "Prediction moi da duoc cap nhat.")


def _load_notify_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_notify_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, _json_safe(state))


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    last_exc: PermissionError | None = None
    for attempt in range(20):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_exc if last_exc is not None else PermissionError(f"Could not replace {path}")


def _fmt_value(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.{int(digits)}f}"


def refresh_local_ohlcv(
    data_path: str | Path,
    symbol: str,
    interval: str,
    base_url: str = DEFAULT_BASE_URL,
    crawl_lookback_days: float = DEFAULT_CRAWL_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path = Path(data_path)
    old_df = _read_existing_ohlcv(data_path)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if old_df.empty:
        start_ms = _timestamp_to_ms(pd.Timestamp("2018-01-01 00:00:00", tz=OUTPUT_TIMEZONE))
        old_last = None
    else:
        old_last_ts = pd.Timestamp(old_df["date"].max())
        old_last = _ts_str(old_last_ts)
        refetch_start = old_last_ts - pd.Timedelta(days=float(crawl_lookback_days))
        start_ms = max(_timestamp_to_ms(refetch_start), 0)

    logger.info(
        "Refreshing %s %s from %s",
        symbol.upper(),
        interval,
        pd.to_datetime(start_ms, unit="ms", utc=True),
    )
    new_df = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=now_ms,
        base_url=base_url,
    )
    new_df = _completed_candles_only(new_df, interval=interval)
    merged = _merge_ohlcv(old_df, new_df)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(data_path, index=False)

    info = {
        "old_rows": int(len(old_df)),
        "old_last": old_last,
        "crawl_lookback_days": float(crawl_lookback_days),
        "crawl_start": str(pd.to_datetime(start_ms, unit="ms", utc=True)),
        "fetched_rows": int(len(new_df)),
        "rows": int(len(merged)),
        "first": _ts_str(merged["date"].min()) if not merged.empty else None,
        "last_closed": _ts_str(merged["date"].max()) if not merged.empty else None,
    }
    logger.info("Data refreshed: %s", info)
    return merged, info


def fetch_current_open(
    symbol: str,
    interval: str,
    base_url: str = DEFAULT_BASE_URL,
) -> CurrentOpen:
    step_ms = _interval_ms(interval)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_open_ms = (now_ms // step_ms) * step_ms
    df = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_ms=current_open_ms,
        end_ms=None,
        base_url=base_url,
        limit=1,
    )
    if df.empty:
        raise RuntimeError("Could not fetch current candle open.")
    row = df.iloc[0]
    return CurrentOpen(
        candle_time=pd.Timestamp(row["date"]),
        open=float(row["open"]),
        fetched_at=_utc_now_iso(),
    )


def loop(
    model_dir: str | Path,
    data_path: str | Path = config.DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    base_url: str = DEFAULT_BASE_URL,
    sleep_after_open_sec: float = 5.0,
    crawl_lookback_days: float = DEFAULT_CRAWL_LOOKBACK_DAYS,
    feature_lookback_bars: int = DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
) -> None:
    logger.info("Starting live loop for %s %s", symbol.upper(), interval)
    while True:
        sleep_sec = _seconds_until_next_open(interval) + max(float(sleep_after_open_sec), 0.0)
        logger.info("Sleeping %.1fs until next live update.", sleep_sec)
        time.sleep(sleep_sec)
        try:
            run_once(
                model_dir=model_dir,
                data_path=data_path,
                output_path=output_path,
                symbol=symbol,
                interval=interval,
                base_url=base_url,
                crawl_lookback_days=crawl_lookback_days,
                feature_lookback_bars=feature_lookback_bars,
            )
        except Exception:
            logger.exception("Live update failed.")


def _predict_entry(
    entry: dict[str, Any],
    model_dir: Path,
    feature_space: CryptoFeatureSpace,
    signal_time: pd.Timestamp,
) -> dict[str, Any]:
    features = list(entry.get("features", []))
    if not features:
        raise ValueError(f"Manifest entry rank {entry.get('rank')} has no features.")
    x = feature_space.matrix(features, pd.Index([signal_time]))

    predictions = []
    for model_record in entry.get("models", []):
        model_path = _resolve_model_path(model_record.get("model_path"), model_dir)
        booster = lgb.Booster(model_file=str(model_path))
        pred = float(booster.predict(x)[0])
        threshold = model_record.get("val_trade_threshold")
        threshold_value = float(threshold) if threshold is not None else None
        is_signal = pred >= threshold_value if threshold_value is not None else None
        predictions.append(
            {
                "horizon": int(model_record.get("horizon")),
                "pred": pred,
                "threshold": threshold_value,
                "is_signal": is_signal,
                "model_path": str(model_path),
            }
        )

    known_signals = [item["is_signal"] for item in predictions if item["is_signal"] is not None]
    ensemble_signal = bool(known_signals) and all(known_signals)
    if len(known_signals) != len(predictions):
        ensemble_signal = None

    label_direction = config.canonical_label_direction(
        entry.get("label_direction") or "long"
    )
    return {
        "entry_id": entry.get("entry_id"),
        "rank": int(entry.get("rank", 0) or 0),
        "archive": entry.get("archive"),
        "label_mode": entry.get("label_mode"),
        "label_direction": label_direction,
        "label_threshold": entry.get("label_threshold"),
        "score": entry.get("score"),
        "n_features": len(features),
        "ensemble_signal": ensemble_signal,
        "pred_mean": float(np.mean([item["pred"] for item in predictions])) if predictions else None,
        "predictions": predictions,
    }


def _predict_final_ensemble(
    entries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if len(entries) < 2:
        return None
    ensemble_config = manifest.get("ensemble") if isinstance(manifest.get("ensemble"), dict) else None
    if not ensemble_config:
        return None
    members = [str(member) for member in ensemble_config.get("members", []) if str(member)]
    if not members:
        return None
    if members:
        by_id = {str(entry.get("entry_id")): entry for entry in entries if entry.get("entry_id")}
        selected_entries = [by_id[member] for member in members if member in by_id]
        missing = [member for member in members if member not in by_id]
    if not selected_entries:
        return None

    signals = [entry.get("ensemble_signal") for entry in selected_entries]
    known = [signal for signal in signals if signal is not None]
    if len(known) != len(signals) or missing:
        ensemble_signal: bool | None = None
    else:
        ensemble_signal = bool(known) and all(signal is True for signal in known)

    pred_values = [
        float(entry["pred_mean"])
        for entry in selected_entries
        if entry.get("pred_mean") is not None and np.isfinite(float(entry["pred_mean"]))
    ]
    pred_mean = float(np.mean(pred_values)) if pred_values else None
    horizon = _max_entry_horizon(selected_entries)
    directions = {
        config.canonical_label_direction(entry.get("label_direction") or "long")
        for entry in selected_entries
    }
    direction_conflict = len(directions) != 1
    label_direction = next(iter(directions)) if not direction_conflict else None
    if direction_conflict:
        ensemble_signal = None
    return {
        "entry_id": "final_ensemble",
        "rank": "ensemble",
        "horizon": horizon,
        "label_direction": label_direction,
        "direction_conflict": direction_conflict,
        "ensemble_signal": ensemble_signal,
        "pred_mean": pred_mean,
        "member_count": int(len(selected_entries)),
        "required_members": members,
        "missing_members": missing,
        "members": [
            {
                "entry_id": entry.get("entry_id"),
                "rank": entry.get("rank"),
                "archive": entry.get("archive"),
                "label_mode": entry.get("label_mode"),
                "label_direction": entry.get("label_direction"),
                "label_threshold": entry.get("label_threshold"),
                "ensemble_signal": entry.get("ensemble_signal"),
                "pred_mean": entry.get("pred_mean"),
            }
            for entry in selected_entries
        ],
        "predictions": [
            {
                "horizon": horizon,
                "pred": pred_mean,
                "threshold": None,
                "is_signal": ensemble_signal,
                "model_path": "final_ensemble",
            }
        ],
    }


def _max_entry_horizon(entries: list[dict[str, Any]]) -> int | None:
    horizons: list[int] = []
    for entry in entries:
        if entry.get("horizon") is not None:
            horizons.append(int(entry["horizon"]))
        for item in entry.get("predictions", []):
            if item.get("horizon") is not None:
                horizons.append(int(item["horizon"]))
    return max(horizons) if horizons else None


def _resolve_model_path(model_path_value: Any, model_dir: Path) -> Path:
    if not model_path_value:
        raise ValueError("Manifest model record has empty model_path.")
    raw = str(model_path_value)
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path

    normalized = raw.replace("\\", "/")
    normalized_path = Path(normalized)
    candidates = []
    if normalized_path.is_absolute():
        candidates.append(normalized_path)
    else:
        candidates.append(normalized_path)
        candidates.append(model_dir / normalized_path.name)
        candidates.append(model_dir / PureWindowsPath(raw).name)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return model_dir / PureWindowsPath(raw).name


def _load_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_features(manifest: dict[str, Any]) -> list[str]:
    features: list[str] = []
    for entry in manifest.get("entries", []):
        for feature in entry.get("features", []):
            feature = str(feature).strip()
            if feature and feature not in features:
                features.append(feature)
    if not features:
        raise ValueError("Manifest has no features.")
    return features


def _required_windows(features: list[str]) -> list[int]:
    windows: set[int] = set()
    for feature in features:
        for match in _WINDOW_SUFFIX_RE.finditer(feature):
            windows.add(int(match.group(1)))
        for match in _WINDOW_ARG_RE.finditer(feature):
            windows.add(int(match.group(1)))
    if not windows:
        windows = {int(w) for w in config.WINDOWS}
    return sorted(window for window in windows if window > 1)


def _feature_tail(raw_df: pd.DataFrame, lookback_bars: int) -> pd.DataFrame:
    lookback = max(int(lookback_bars), _minimum_feature_lookback_bars())
    if len(raw_df) <= lookback:
        return raw_df.copy()
    return raw_df.tail(lookback).copy()


def _minimum_feature_lookback_bars() -> int:
    max_window = max(int(w) for w in config.WINDOWS)
    return int(max_window * (config.EXPR_MAX_DEPTH + 2))


def _read_existing_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


def _merge_ohlcv(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [old_df, new_df] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    return merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _completed_candles_only(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    now_local = pd.Timestamp.now(tz=OUTPUT_TIMEZONE).tz_localize(None)
    interval_delta = pd.to_timedelta(_interval_ms(interval), unit="ms")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] + interval_delta <= now_local].copy()


def _timestamp_to_ms(ts: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(OUTPUT_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(OUTPUT_TIMEZONE)
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


def _interval_ms(interval: str) -> int:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return int(INTERVAL_TO_MS[interval])


def _seconds_until_next_open(interval: str) -> float:
    step_ms = _interval_ms(interval)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    next_open_ms = ((now_ms // step_ms) + 1) * step_ms
    return max((next_open_ms - now_ms) / 1000.0, 0.0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return _ts_str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory containing manifest.json.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Local crypto CSV path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--loop", action="store_true", help="Run forever at each new candle.")
    parser.add_argument(
        "--crawl-lookback-days",
        type=float,
        default=DEFAULT_CRAWL_LOOKBACK_DAYS,
        help="Refetch from last local candle minus this many days before merging.",
    )
    parser.add_argument(
        "--feature-lookback-bars",
        type=int,
        default=DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
        help="Recent bars used to compute live features.",
    )
    parser.add_argument(
        "--sleep-after-open",
        type=float,
        default=5.0,
        help="Seconds to wait after a new candle opens before refresh/predict.",
    )
    args = parser.parse_args()

    if args.loop:
        loop(
            model_dir=args.model_dir,
            data_path=args.data,
            output_path=args.out,
            symbol=args.symbol,
            interval=args.interval,
            base_url=args.base_url,
            sleep_after_open_sec=args.sleep_after_open,
            crawl_lookback_days=args.crawl_lookback_days,
            feature_lookback_bars=args.feature_lookback_bars,
        )
    else:
        run_once(
            model_dir=args.model_dir,
            data_path=args.data,
            output_path=args.out,
            symbol=args.symbol,
            interval=args.interval,
            base_url=args.base_url,
            crawl_lookback_days=args.crawl_lookback_days,
            feature_lookback_bars=args.feature_lookback_bars,
        )


if __name__ == "__main__":
    main()
