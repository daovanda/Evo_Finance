"""Monitor 5-minute Long/Short MFE models and notify Exness manual trades.

This backend never places an order. It refreshes public Binance 5m candles,
predicts one Long and one Short production model, writes a JSON snapshot, and
sends one Telegram status update after every closed 5-minute candle.

Train production models first:

    python -m crypto.prod.train_model `
      --archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --data data/crypto/BTCUSDT_5m.csv `
      --rank 1 `
      --label-mode mfe `
      --label-direction Long `
      --label-threshold 0.001 `
      --trade-top-fraction 0.40 `
      --run-name exness_5m_long_mfe_r1

    python -m crypto.prod.train_model `
      --archive crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json `
      --data data/crypto/BTCUSDT_5m.csv `
      --rank 1 `
      --label-mode mfe `
      --label-direction Short `
      --label-threshold 0.001 `
      --trade-top-fraction 0.40 `
      --run-name exness_5m_short_mfe_r1

Run once locally:

    python -m crypto.prod.backend_exness_5m --force-ipv4

Run continuously:

    python -m crypto.prod.backend_exness_5m --force-ipv4 --loop
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from craw_btc import DEFAULT_BASE_URL, DEFAULT_SYMBOL
from crypto.data import load_ohlcv
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.prod.live_backend import (
    DEFAULT_CRAWL_LOOKBACK_DAYS,
    DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
    _feature_tail,
    _load_manifest,
    _manifest_features,
    _predict_entry,
    _rank_one_manifest_entry,
    _required_windows,
    _seconds_until_next_open,
    _write_payload,
    fetch_current_open,
    force_ipv4_networking,
    refresh_local_ohlcv,
)
from crypto.prod.telegram_notify import (
    send_telegram_message,
    telegram_configured,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.prod.backend_exness_5m")


DEFAULT_LONG_MODEL_DIR = Path("crypto/prod/model/exness_5m_long_mfe_r1")
DEFAULT_SHORT_MODEL_DIR = Path("crypto/prod/model/exness_5m_short_mfe_r1")
DEFAULT_DATA_PATH = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT_PATH = Path("crypto/prod/live/latest_exness_5m_signal.json")
DEFAULT_NOTIFY_STATE_PATH = Path(
    "crypto/prod/live/exness_5m_notify_state.json"
)
DEFAULT_INTERVAL = "5m"
DEFAULT_FILTER_TRIGGER = 0.00025
DEFAULT_TAKE_PROFIT = 0.01
DEFAULT_STOP_LOSS = 0.0
DEFAULT_EXNESS_PRICE_OFFSET = 80.0
TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN_EXNESS_5M"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID_EXNESS_5M"


def run_once(
    long_model_dir: str | Path = DEFAULT_LONG_MODEL_DIR,
    short_model_dir: str | Path = DEFAULT_SHORT_MODEL_DIR,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    notify_state_path: str | Path = DEFAULT_NOTIFY_STATE_PATH,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    base_url: str = DEFAULT_BASE_URL,
    crawl_lookback_days: float = DEFAULT_CRAWL_LOOKBACK_DAYS,
    feature_lookback_bars: int = DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
    filter_trigger: float = DEFAULT_FILTER_TRIGGER,
    take_profit: float = DEFAULT_TAKE_PROFIT,
    stop_loss: float = DEFAULT_STOP_LOSS,
    exness_price_offset: float = DEFAULT_EXNESS_PRICE_OFFSET,
    notify_telegram: bool = True,
) -> dict[str, Any]:
    if str(interval) != DEFAULT_INTERVAL:
        raise ValueError(
            f"Exness backend requires interval={DEFAULT_INTERVAL}, got {interval}."
        )
    if float(filter_trigger) <= 0.0:
        raise ValueError("filter_trigger must be positive.")
    if float(take_profit) <= 0.0 or float(stop_loss) < 0.0:
        raise ValueError("take_profit must be positive and stop_loss non-negative.")
    if float(exness_price_offset) < 0.0:
        raise ValueError("exness_price_offset must be non-negative.")

    model_dirs = {
        "long": Path(long_model_dir),
        "short": Path(short_model_dir),
    }
    manifests = {
        direction: _load_manifest(path)
        for direction, path in model_dirs.items()
    }
    entries = {
        direction: _validated_entry(
            manifest=manifests[direction],
            expected_direction=direction,
        )
        for direction in ("long", "short")
    }

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
    all_features: list[str] = []
    for manifest in manifests.values():
        for feature in _manifest_features(manifest):
            if feature not in all_features:
                all_features.append(feature)
    required_windows = _required_windows(all_features)
    feature_raw_df = _feature_tail(raw_df, feature_lookback_bars)
    logger.info(
        "Building Exness 5m feature tail: rows=%d/%d | windows=%s",
        len(feature_raw_df),
        len(raw_df),
        required_windows,
    )
    feature_df = build_feature_frame(
        feature_raw_df,
        windows=required_windows,
        quality_filter=False,
    )
    feature_space = CryptoFeatureSpace(
        feature_df,
        selectable_features(feature_df),
    )

    current_open = fetch_current_open(
        symbol=symbol,
        interval=interval,
        base_url=base_url,
    )
    if pd.Timestamp(current_open.candle_time) <= signal_time:
        raise ValueError(
            "Entry candle must be newer than the closed signal candle: "
            f"entry={current_open.candle_time}, signal={signal_time}."
        )

    states: dict[str, dict[str, Any]] = {}
    for direction in ("long", "short"):
        state = _predict_entry(
            entry=entries[direction],
            model_dir=model_dirs[direction],
            feature_space=feature_space,
            signal_time=signal_time,
        )
        state["trade_top_fraction"] = _entry_top_fraction(
            entries[direction]
        )
        state["decision_thresholds"] = [
            {
                "horizon": prediction.get("horizon"),
                "threshold": prediction.get("threshold"),
            }
            for prediction in state.get("predictions", [])
        ]
        states[direction] = state

    long_signal = states["long"].get("ensemble_signal") is True
    short_signal = states["short"].get("ensemble_signal") is True
    signal = _combined_signal(long_signal, short_signal)
    entry_open = float(current_open.open)
    long_trigger_price = entry_open * (1.0 + float(filter_trigger))
    short_trigger_price = entry_open * (1.0 - float(filter_trigger))
    price_offset = float(exness_price_offset)
    payload: dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "status": "OK",
        "error": None,
        "monitor_only": True,
        "execution_enabled": False,
        "symbol": symbol.upper(),
        "interval": interval,
        "signal_time": str(signal_time),
        "entry_candle_time": str(current_open.candle_time),
        "entry_open_binance": entry_open,
        "entry_open_exness": entry_open - price_offset,
        "entry_open_fetched_at": current_open.fetched_at,
        "signal": signal,
        "has_trade_signal": bool(long_signal or short_signal),
        "long_signal": long_signal,
        "short_signal": short_signal,
        "strategy": {
            "filter_first_1m": float(filter_trigger),
            "long_trigger_price": long_trigger_price,
            "long_trigger_price_exness": long_trigger_price - price_offset,
            "short_trigger_price": short_trigger_price,
            "short_trigger_price_exness": short_trigger_price - price_offset,
            "exness_price_offset": price_offset,
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "manual_execution": True,
        },
        "models": states,
        "model_dirs": {
            direction: str(path)
            for direction, path in model_dirs.items()
        },
        "feature_build": {
            "rows_used": int(len(feature_raw_df)),
            "total_rows": int(len(raw_df)),
            "windows": required_windows,
            "lookback_bars": int(feature_lookback_bars),
        },
        "data_update": update_info,
    }
    _write_payload(output_path, payload)

    notification = _notify_signal_once(
        payload=payload,
        state_path=Path(notify_state_path),
        enabled=bool(notify_telegram),
    )
    payload["telegram"] = notification
    _write_payload(output_path, payload)
    return payload


def _validated_entry(
    manifest: dict[str, Any],
    expected_direction: str,
) -> dict[str, Any]:
    entry = _rank_one_manifest_entry(manifest, expected_direction)
    mode = str(entry.get("label_mode") or "").strip().lower()
    direction = str(entry.get("label_direction") or "").strip().lower()
    horizons = sorted(
        int(model.get("horizon"))
        for model in entry.get("models", [])
        if model.get("horizon") is not None
    )
    if mode != "mfe" or direction != expected_direction or horizons != [3]:
        raise ValueError(
            f"{expected_direction} model must be mode=mfe, direction="
            f"{expected_direction}, horizons=[3]; got mode={mode}, "
            f"direction={direction}, horizons={horizons}."
        )
    fraction = _entry_top_fraction(entry)
    if fraction is None:
        raise ValueError(
            f"{expected_direction} manifest is missing trade_top_fraction."
        )
    return entry


def _entry_top_fraction(entry: dict[str, Any]) -> float | None:
    values = {
        float(model["trade_top_fraction"])
        for model in entry.get("models", [])
        if model.get("trade_top_fraction") is not None
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(
            f"Model horizons use different top fractions: {sorted(values)}"
        )
    return next(iter(values))


def _combined_signal(long_signal: bool, short_signal: bool) -> str:
    if long_signal and short_signal:
        return "BOTH"
    if long_signal:
        return "LONG"
    if short_signal:
        return "SHORT"
    return "NO_SIGNAL"


def _notify_signal_once(
    payload: dict[str, Any],
    state_path: Path,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"ok": False, "skipped": True, "reason": "disabled_by_cli"}

    key = "|".join(
        [
            str(payload.get("symbol") or ""),
            str(payload.get("interval") or ""),
            str(payload.get("signal_time") or ""),
            str(payload.get("signal") or ""),
        ]
    )
    state = _load_json(state_path)
    if state.get("last_key") == key:
        return {"ok": False, "skipped": True, "reason": "duplicate"}

    try:
        result = send_telegram_message(
            _telegram_message(payload),
            token_env=TELEGRAM_TOKEN_ENV,
            chat_id_env=TELEGRAM_CHAT_ID_ENV,
        )
    except Exception as exc:  # Telegram must not stop market monitoring.
        logger.warning("Exness 5m Telegram notification failed: %s", exc)
        return {
            "ok": False,
            "skipped": False,
            "reason": "telegram_error",
            "error": str(exc),
        }
    if result.get("ok"):
        _save_json(
            state_path,
            {
                "last_key": key,
                "last_sent_at": _utc_now_iso(),
                "signal_time": payload.get("signal_time"),
                "signal": payload.get("signal"),
            },
        )
    return result


def _telegram_message(payload: dict[str, Any]) -> str:
    strategy = payload.get("strategy", {})
    lines = [
        f"5M SIGNAL | {payload.get('signal')}",
        f"Signal candle: {payload.get('signal_time')}",
        f"Entry candle: {payload.get('entry_candle_time')}",
        (
            f"Binance open: {_fmt_price(payload.get('entry_open_binance'))} "
            f"| {_fmt_price(payload.get('entry_open_exness'))}"
        ),
        (
            "First 1m trigger: "
            f"{float(strategy.get('filter_first_1m', 0.0)):.3%}"
        ),
        (
            "Long trigger >= "
            f"{_fmt_price(strategy.get('long_trigger_price'))} "
            f"| {_fmt_price(strategy.get('long_trigger_price_exness'))}"
        ),
        (
            "Short trigger <= "
            f"{_fmt_price(strategy.get('short_trigger_price'))} "
            f"| {_fmt_price(strategy.get('short_trigger_price_exness'))}"
        ),
    ]
    models = payload.get("models", {})
    for direction, label in (("long", "LONG"), ("short", "SHORT")):
        state = models.get(direction, {}) if isinstance(models, dict) else {}
        predictions = state.get("predictions", [])
        threshold = predictions[0].get("threshold") if predictions else None
        lines.append(
            f"{label}: signal={_yes_no(state.get('ensemble_signal'))} "
            f"| score={_fmt_number(state.get('pred_mean'), 6)} "
            f"| thr={_fmt_number(threshold, 6)} |"
        )
    return "\n".join(lines)


def _yes_no(value: Any) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return "N/A"


def _fmt_number(value: Any, digits: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if np.isfinite(number) else "N/A"


def _fmt_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.2%}" if np.isfinite(number) else "N/A"


def _fmt_price(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:,.2f}" if np.isfinite(number) else "N/A"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def loop(
    *,
    sleep_after_open: float = 5.0,
    **run_kwargs: Any,
) -> None:
    logger.info(
        "Starting Exness 5m monitor loop; Binance public data only, "
        "order execution disabled."
    )
    while True:
        sleep_seconds = _seconds_until_next_open(DEFAULT_INTERVAL) + max(
            float(sleep_after_open),
            0.0,
        )
        logger.info("Sleeping %.1fs until next 5m prediction.", sleep_seconds)
        time.sleep(sleep_seconds)
        try:
            run_once(**run_kwargs)
        except Exception:
            logger.exception("Exness 5m monitor update failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long-model-dir",
        default=str(DEFAULT_LONG_MODEL_DIR),
    )
    parser.add_argument(
        "--short-model-dir",
        default=str(DEFAULT_SHORT_MODEL_DIR),
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--notify-state",
        default=str(DEFAULT_NOTIFY_STATE_PATH),
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--crawl-lookback-days",
        type=float,
        default=DEFAULT_CRAWL_LOOKBACK_DAYS,
    )
    parser.add_argument(
        "--feature-lookback-bars",
        type=int,
        default=DEFAULT_LIVE_FEATURE_LOOKBACK_BARS,
    )
    parser.add_argument(
        "--filter-trigger",
        type=float,
        default=DEFAULT_FILTER_TRIGGER,
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULT_TAKE_PROFIT,
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_STOP_LOSS,
    )
    parser.add_argument(
        "--exness-price-offset",
        type=float,
        default=DEFAULT_EXNESS_PRICE_OFFSET,
        help="Subtract this amount from Binance prices shown as Exness prices.",
    )
    parser.add_argument("--force-ipv4", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--sleep-after-open", type=float, default=5.0)
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Update JSON but never send Telegram.",
    )
    parser.add_argument(
        "--telegram-test",
        action="store_true",
        help="Send one test through EXNESS_5M Telegram credentials and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force_ipv4:
        force_ipv4_networking()
    if args.telegram_test:
        if not telegram_configured(
            token_env=TELEGRAM_TOKEN_ENV,
            chat_id_env=TELEGRAM_CHAT_ID_ENV,
        ):
            raise RuntimeError(
                "Missing TELEGRAM_BOT_TOKEN_EXNESS_5M or "
                "TELEGRAM_CHAT_ID_EXNESS_5M."
            )
        result = send_telegram_message(
            "Evo Exness 5m Telegram test: OK",
            token_env=TELEGRAM_TOKEN_ENV,
            chat_id_env=TELEGRAM_CHAT_ID_ENV,
        )
        print(json.dumps(result, indent=2))
        return

    kwargs = {
        "long_model_dir": args.long_model_dir,
        "short_model_dir": args.short_model_dir,
        "data_path": args.data,
        "output_path": args.out,
        "notify_state_path": args.notify_state,
        "symbol": args.symbol,
        "interval": args.interval,
        "base_url": args.base_url,
        "crawl_lookback_days": args.crawl_lookback_days,
        "feature_lookback_bars": args.feature_lookback_bars,
        "filter_trigger": args.filter_trigger,
        "take_profit": args.take_profit,
        "stop_loss": args.stop_loss,
        "exness_price_offset": args.exness_price_offset,
        "notify_telegram": not args.no_telegram,
    }
    if args.loop:
        loop(
            **kwargs,
            sleep_after_open=args.sleep_after_open,
        )
    else:
        payload = run_once(**kwargs)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
