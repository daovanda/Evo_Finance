"""Plot 5m H1 candles accepted by MFE Q20 and an after-1m meta learner.

PowerShell example:
    python -m temp.plot_5m_meta_after_1m_signals `
      --meta-archive crypto/results/crypto_btc_5m_meta_after_1m_mfe_q20_h3_seed1_1h.json `
      --rank 1 `
      --data data/crypto/BTCUSDT_5m.csv `
      --data-1m data/crypto/BTCUSDT_1m.csv `
      --bars 500 `
      --panels 4 `
      --seed 1 `
      --out temp/output/meta_after_1m_accepted_h1.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from crypto import config
from crypto.data import (
    add_binary_labels,
    load_ohlcv,
    make_walk_forward_folds,
    split_labeled_by_dates,
)
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.fitness import CryptoFitnessEvaluator
from crypto.meta_targets import (
    align_meta_feature_frame,
    build_meta_feature_alignment,
    build_meta_learner_data,
    load_meta_base,
    required_feature_windows,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.plot_5m_meta_after_1m_signals")


def _load_rank(path: Path, rank: int) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    entry = next(
        (item for item in payload.get("entries", []) if int(item["rank"]) == rank),
        None,
    )
    if entry is None:
        raise ValueError(f"Rank {rank} is absent from {path}.")
    return metadata, dict(entry)


def _timestamp(value: object, fallback: str | None = None) -> str | None:
    if value in (None, ""):
        return fallback
    return str(value)


def _valid_meta_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return frame.dropna(
        subset=[f"label_h{horizon}", f"future_return_h{horizon}"]
    )


def _top_fraction_threshold(prediction: pd.Series, fraction: float) -> float:
    clean = prediction.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("Cannot derive a selection threshold from empty predictions.")
    count = min(len(clean), max(1, int(np.ceil(len(clean) * float(fraction)))))
    return float(clean.nlargest(count).iloc[-1])


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _build_direct_meta_features(
    minute: pd.DataFrame,
    feature_names: list[str],
    output_index: pd.Index,
) -> pd.DataFrame | None:
    """Fast exact path for the direct rolling features used by current Rank 1."""
    original_supported = {
        "bb_width_2",
        "bb_pos_2",
        "min_ret_2",
        "taker_delta_quote_sum_ratio_3",
        "imbalance_x_trend_3",
    }
    offset_rank1_supported = {
        "imbalance_x_trend_7",
        "body_pct",
        "rogers_satchell_vol_2",
        "max_ret_2",
        "signed_log(shift(q25(bb_width_7, 7), 7))",
        "where(lt(garman_klass_vol_5, const(0.00319219)), const(-1), const(0))",
    }
    requested = set(feature_names)
    if not (
        requested.issubset(original_supported)
        or requested.issubset(offset_rank1_supported)
    ):
        return None

    open_ = pd.to_numeric(minute["open"], errors="coerce").astype(float)
    high = pd.to_numeric(minute["high"], errors="coerce").astype(float)
    low = pd.to_numeric(minute["low"], errors="coerce").astype(float)
    close = pd.to_numeric(minute["close"], errors="coerce").astype(float)
    close_ret = close.pct_change()

    if requested.issubset(offset_rank1_supported):
        volume = pd.to_numeric(minute["volume"], errors="coerce").astype(float)
        taker_base = pd.to_numeric(
            minute["taker_buy_base_volume"], errors="coerce"
        ).astype(float)
        taker_delta_base_ratio = _safe_div(2.0 * taker_base - volume, volume)
        delta_mean7 = taker_delta_base_ratio.rolling(7, min_periods=7).mean()
        delta_std7 = taker_delta_base_ratio.rolling(7, min_periods=7).std()
        taker_delta_z7 = _safe_div(
            taker_delta_base_ratio - delta_mean7,
            delta_std7,
        )
        ma7 = close.rolling(7, min_periods=7).mean()
        std7 = close.rolling(7, min_periods=7).std()
        bb_width7 = _safe_div(4.0 * std7, ma7)

        log_hc = np.log(_safe_div(high, close))
        log_ho = np.log(_safe_div(high, open_))
        log_lc = np.log(_safe_div(low, close))
        log_lo = np.log(_safe_div(low, open_))
        rs_component = (log_hc * log_ho) + (log_lc * log_lo)
        rs_vol2 = np.sqrt(
            rs_component.rolling(2, min_periods=2).mean().clip(lower=0.0)
        )

        log_hl = np.log(_safe_div(high, low))
        log_co = np.log(_safe_div(close, open_))
        gk_component = (
            0.5 * log_hl.pow(2)
            - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
        )
        gk_vol5 = np.sqrt(
            gk_component.rolling(5, min_periods=5).mean().clip(lower=0.0)
        )
        shifted_q25_width7 = (
            bb_width7.rolling(7, min_periods=7).quantile(0.25).shift(7)
        )
        values = {
            "imbalance_x_trend_7": taker_delta_z7
            * (_safe_div(close, ma7) - 1.0),
            "body_pct": _safe_div(close - open_, open_),
            "rogers_satchell_vol_2": rs_vol2,
            "max_ret_2": close_ret.rolling(2, min_periods=2).max(),
            "signed_log(shift(q25(bb_width_7, 7), 7))": np.sign(
                shifted_q25_width7
            )
            * np.log1p(shifted_q25_width7.abs()),
            "where(lt(garman_klass_vol_5, const(0.00319219)), const(-1), const(0))": pd.Series(
                np.where(gk_vol5 < 0.00319219, -1.0, 0.0),
                index=minute.index,
            ),
        }
        return pd.DataFrame(
            {name: values[name].reindex(output_index) for name in feature_names},
            index=output_index,
        )

    volume = pd.to_numeric(minute["volume"], errors="coerce").astype(float)
    taker_base = pd.to_numeric(
        minute["taker_buy_base_volume"], errors="coerce"
    ).astype(float)
    taker_quote = pd.to_numeric(
        minute["taker_buy_quote_volume"], errors="coerce"
    ).astype(float)
    ma2 = close.rolling(2, min_periods=2).mean()
    std2 = close.rolling(2, min_periods=2).std()
    quote_volume = close * volume
    taker_delta_quote = 2.0 * taker_quote - quote_volume
    taker_delta_base_ratio = _safe_div(2.0 * taker_base - volume, volume)
    delta_mean3 = taker_delta_base_ratio.rolling(3, min_periods=3).mean()
    delta_std3 = taker_delta_base_ratio.rolling(3, min_periods=3).std()
    taker_delta_z3 = _safe_div(
        taker_delta_base_ratio - delta_mean3,
        delta_std3,
    )
    ma3 = close.rolling(3, min_periods=3).mean()
    values = {
        "bb_width_2": _safe_div(4.0 * std2, ma2),
        "bb_pos_2": _safe_div(close - ma2, 2.0 * std2),
        "min_ret_2": close_ret.rolling(2, min_periods=2).min(),
        "taker_delta_quote_sum_ratio_3": _safe_div(
            taker_delta_quote.rolling(3, min_periods=3).sum(),
            quote_volume.rolling(3, min_periods=3).sum(),
        ),
        "imbalance_x_trend_3": taker_delta_z3 * (_safe_div(close, ma3) - 1.0),
    }
    return pd.DataFrame(
        {name: values[name].reindex(output_index) for name in feature_names},
        index=output_index,
    )


def _minute_two_open_summary(
    signal_prediction: pd.Series,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
) -> dict[str, float | int]:
    signal_index = pd.DatetimeIndex(signal_prediction.index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(
        signal_index + target_interval
    )
    open_minute_2 = pd.to_numeric(minute["open"], errors="coerce").reindex(
        signal_index + target_interval + pd.Timedelta(minutes=1)
    )
    movement = pd.Series(
        open_minute_2.to_numpy(float) / open_h1.to_numpy(float) - 1.0,
        index=signal_index,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(len(movement)),
        "mean": float(movement.mean()),
        "q10": float(movement.quantile(0.10)),
        "q25": float(movement.quantile(0.25)),
        "median": float(movement.median()),
        "q75": float(movement.quantile(0.75)),
        "q80": float(movement.quantile(0.80)),
        "q90": float(movement.quantile(0.90)),
        "positive_rate": float(movement.gt(0.0).mean()),
        "negative_rate": float(movement.lt(0.0).mean()),
    }


def _minute_two_low_summary(
    signal_prediction: pd.Series,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
) -> dict[str, float | int]:
    signal_index = pd.DatetimeIndex(signal_prediction.index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(
        signal_index + target_interval
    )
    low_minute_2 = pd.to_numeric(minute["low"], errors="coerce").reindex(
        signal_index + target_interval + pd.Timedelta(minutes=1)
    )
    movement = pd.Series(
        low_minute_2.to_numpy(float) / open_h1.to_numpy(float) - 1.0,
        index=signal_index,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(len(movement)),
        "mean": float(movement.mean()),
        "minimum": float(movement.min()),
        "q01": float(movement.quantile(0.01)),
        "q05": float(movement.quantile(0.05)),
        "q10": float(movement.quantile(0.10)),
        "q25": float(movement.quantile(0.25)),
        "median": float(movement.median()),
        "q75": float(movement.quantile(0.75)),
        "q80": float(movement.quantile(0.80)),
        "q90": float(movement.quantile(0.90)),
        "below_005": float(movement.lt(-0.0005).mean()),
        "below_010": float(movement.lt(-0.0010).mean()),
        "below_015": float(movement.lt(-0.0015).mean()),
    }


def _close_horizon_summary(
    signal_prediction: pd.Series,
    raw: pd.DataFrame,
    horizon: int,
) -> dict[str, float | int]:
    signal_index = pd.DatetimeIndex(signal_prediction.index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").shift(-1).reindex(
        signal_index
    )
    close_h = pd.to_numeric(raw["close"], errors="coerce").shift(
        -int(horizon)
    ).reindex(signal_index)
    returns = close_h.div(open_h1).sub(1.0).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return {
        "n": int(len(returns)),
        "mean": float(returns.mean()),
        "q10": float(returns.quantile(0.10)),
        "q25": float(returns.quantile(0.25)),
        "median": float(returns.median()),
        "q75": float(returns.quantile(0.75)),
        "q90": float(returns.quantile(0.90)),
        "positive_rate": float(returns.gt(0.0).mean()),
        "negative_rate": float(returns.lt(0.0).mean()),
    }


def _h1_dynamic_tp_summary(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    horizon: int,
) -> dict[str, float | int]:
    index = pd.DatetimeIndex(accepted_prediction.index)
    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").shift(-1).reindex(index)
    high_h1 = pd.to_numeric(raw["high"], errors="coerce").shift(-1).reindex(index)
    high_return = high_h1.div(open_h1).sub(1.0)
    frame = pd.DataFrame({"tp": tp, "high_return": high_return}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    hit = frame["high_return"].gt(frame["tp"])
    hit_tp = frame.loc[hit, "tp"]
    return {
        "n": int(len(frame)),
        "hit_n": int(hit.sum()),
        "hit_rate": float(hit.mean()),
        "tp_mean_all": float(frame["tp"].mean()),
        "tp_min_all": float(frame["tp"].min()),
        "tp_q01_all": float(frame["tp"].quantile(0.01)),
        "tp_q05_all": float(frame["tp"].quantile(0.05)),
        "tp_q10_all": float(frame["tp"].quantile(0.10)),
        "tp_q20_all": float(frame["tp"].quantile(0.20)),
        "tp_q25_all": float(frame["tp"].quantile(0.25)),
        "tp_median_all": float(frame["tp"].median()),
        "tp_q75_all": float(frame["tp"].quantile(0.75)),
        "tp_q80_all": float(frame["tp"].quantile(0.80)),
        "tp_q90_all": float(frame["tp"].quantile(0.90)),
        "tp_q95_all": float(frame["tp"].quantile(0.95)),
        "tp_q99_all": float(frame["tp"].quantile(0.99)),
        "tp_max_all": float(frame["tp"].max()),
        "tp_mean_hit": float(hit_tp.mean()),
        "tp_median_hit": float(hit_tp.median()),
    }


def _first_minute_dynamic_tp_summary(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
    horizon: int,
) -> dict[str, float | int]:
    index = pd.DatetimeIndex(accepted_prediction.index)
    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(
        index + target_interval
    )
    high_minute_1 = pd.to_numeric(minute["high"], errors="coerce").reindex(
        index + target_interval
    )
    high_return = pd.Series(
        high_minute_1.to_numpy(float) / open_h1.to_numpy(float) - 1.0,
        index=index,
    )
    frame = pd.DataFrame({"tp": tp, "high_return": high_return}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    hit = frame["high_return"].gt(frame["tp"])
    return {
        "n": int(len(frame)),
        "hit_n": int(hit.sum()),
        "hit_rate": float(hit.mean()),
        "tp_mean_hit": float(frame.loc[hit, "tp"].mean()),
    }


def _minute_2_to_15_high_summary(
    accepted_prediction: pd.Series,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
) -> dict[str, float | int]:
    index = pd.DatetimeIndex(accepted_prediction.index)
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(
        index + target_interval
    )
    minute_high = pd.to_numeric(minute["high"], errors="coerce")
    high_columns = [
        minute_high.reindex(
            index + target_interval + pd.Timedelta(minutes=offset)
        ).to_numpy(float)
        for offset in range(1, 15)
    ]
    max_high = np.nanmax(np.column_stack(high_columns), axis=1)
    movement = pd.Series(
        max_high / open_h1.to_numpy(float) - 1.0,
        index=index,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(len(movement)),
        "mean": float(movement.mean()),
        "minimum": float(movement.min()),
        "q01": float(movement.quantile(0.01)),
        "q05": float(movement.quantile(0.05)),
        "q10": float(movement.quantile(0.10)),
        "q20": float(movement.quantile(0.20)),
        "q25": float(movement.quantile(0.25)),
        "median": float(movement.median()),
        "q75": float(movement.quantile(0.75)),
        "q80": float(movement.quantile(0.80)),
        "q90": float(movement.quantile(0.90)),
        "q95": float(movement.quantile(0.95)),
        "q99": float(movement.quantile(0.99)),
        "maximum": float(movement.max()),
    }


def _distribution_summary(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "minimum": float(clean.min()),
        "q01": float(clean.quantile(0.01)),
        "q05": float(clean.quantile(0.05)),
        "q10": float(clean.quantile(0.10)),
        "q20": float(clean.quantile(0.20)),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "q80": float(clean.quantile(0.80)),
        "q90": float(clean.quantile(0.90)),
        "q95": float(clean.quantile(0.95)),
        "q99": float(clean.quantile(0.99)),
        "maximum": float(clean.max()),
    }


def _requested_trade_distributions(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
    horizon: int,
) -> dict[str, dict[str, float | int]]:
    index = pd.DatetimeIndex(accepted_prediction.index)
    entry_index = index + target_interval
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(entry_index)
    denominator = open_h1.to_numpy(float)

    def minute_return(column: str, offset: int) -> pd.Series:
        price = pd.to_numeric(minute[column], errors="coerce").reindex(
            entry_index + pd.Timedelta(minutes=offset)
        )
        return pd.Series(price.to_numpy(float) / denominator - 1.0, index=index)

    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    close_h = pd.to_numeric(raw["close"], errors="coerce").shift(
        -int(horizon)
    ).reindex(index)
    low_h2 = pd.to_numeric(raw["low"], errors="coerce").shift(-2).reindex(index)
    close_return = pd.Series(
        close_h.to_numpy(float) / denominator - 1.0,
        index=index,
    )
    low_h2_return = pd.Series(
        low_h2.to_numpy(float) / denominator - 1.0,
        index=index,
    )
    return {
        "low_minute_1": _distribution_summary(minute_return("low", 0)),
        "high_minute_1": _distribution_summary(minute_return("high", 0)),
        "dynamic_tp": _distribution_summary(tp),
        "open_minute_2": _distribution_summary(minute_return("open", 1)),
        "low_h2": _distribution_summary(low_h2_return),
        "close_h": _distribution_summary(close_return),
    }


def _ma3_slope_filter_summary(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    horizon: int,
    trade_cost: float,
) -> dict[str, float | int]:
    index = pd.DatetimeIndex(accepted_prediction.index)
    close = pd.to_numeric(raw["close"], errors="coerce")
    ma3 = close.rolling(3, min_periods=3).mean()
    ma3_slope = ma3 - ma3.shift(2)
    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    label = pd.to_numeric(
        meta_frame[f"label_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    gross_return = pd.to_numeric(
        meta_frame[f"future_return_h{int(horizon)}"], errors="coerce"
    ).reindex(index)
    frame = pd.DataFrame(
        {
            "ma3_slope": ma3_slope.reindex(index),
            "tp": tp,
            "label": label,
            "gross_return": gross_return,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    filtered = frame[frame["ma3_slope"].gt(0.0)]

    def metrics(data: pd.DataFrame, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_n": int(len(data)),
            f"{prefix}_hit_rate": float(data["label"].mean()),
            f"{prefix}_gross_mean": float(data["gross_return"].mean()),
            f"{prefix}_net_mean": float(
                data["gross_return"].mean() - float(trade_cost)
            ),
            f"{prefix}_tp_mean": float(data["tp"].mean()),
        }

    return {
        **metrics(frame, "before"),
        **metrics(filtered, "after"),
        "kept_rate": float(len(filtered) / len(frame)) if len(frame) else 0.0,
    }


def _filter_positive_ma3_slope(
    prediction: pd.Series,
    raw: pd.DataFrame,
) -> pd.Series:
    close = pd.to_numeric(raw["close"], errors="coerce")
    ma3 = close.rolling(3, min_periods=3).mean()
    ma3_slope = ma3 - ma3.shift(2)
    keep = ma3_slope.reindex(prediction.index).gt(0.0)
    return prediction[keep.fillna(False)]


def _dynamic_tp_sl_strategy_summary(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    horizon: int,
    stop_loss: float,
    trade_cost: float,
) -> dict[str, float | int]:
    signal_index = pd.DatetimeIndex(accepted_prediction.index)
    entry_open = pd.to_numeric(raw["open"], errors="coerce").shift(-1).reindex(
        signal_index
    )
    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(signal_index)
    close_h = pd.to_numeric(raw["close"], errors="coerce").shift(
        -int(horizon)
    ).reindex(signal_index)

    frame = pd.DataFrame(
        {"entry": entry_open, "tp": tp, "close_h": close_h},
        index=signal_index,
    )
    for step in range(1, int(horizon) + 1):
        frame[f"high_{step}"] = pd.to_numeric(
            raw["high"], errors="coerce"
        ).shift(-step).reindex(signal_index)
        frame[f"low_{step}"] = pd.to_numeric(
            raw["low"], errors="coerce"
        ).shift(-step).reindex(signal_index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    gross = frame["close_h"].to_numpy(float) / frame["entry"].to_numpy(float) - 1.0
    exit_code = np.full(len(frame), "close", dtype=object)
    active = np.ones(len(frame), dtype=bool)
    entry = frame["entry"].to_numpy(float)
    tp_values = frame["tp"].to_numpy(float)
    tp_price = entry * (1.0 + tp_values)
    sl_price = entry * (1.0 - float(stop_loss))

    for step in range(1, int(horizon) + 1):
        high = frame[f"high_{step}"].to_numpy(float)
        low = frame[f"low_{step}"].to_numpy(float)
        tp_hit = active & (high >= tp_price)
        sl_hit = active & (low <= sl_price)
        both = tp_hit & sl_hit
        sl_exit = sl_hit
        tp_exit = tp_hit & ~both
        gross[sl_exit] = -float(stop_loss)
        gross[tp_exit] = tp_values[tp_exit]
        exit_code[sl_exit] = f"sl_h{step}"
        exit_code[tp_exit] = f"tp_h{step}"
        active[sl_exit | tp_exit] = False

    exit_text = exit_code.astype(str)
    net = gross - float(trade_cost)
    return {
        "n": int(len(frame)),
        "tp_n": int(np.count_nonzero(np.char.startswith(exit_text, "tp_"))),
        "sl_n": int(np.count_nonzero(np.char.startswith(exit_text, "sl_"))),
        "close_n": int(np.count_nonzero(exit_code == "close")),
        "gross_mean": float(np.mean(gross)) if len(gross) else float("nan"),
        "net_mean": float(np.mean(net)) if len(net) else float("nan"),
        "win_rate": float(np.mean(net > 0.0)) if len(net) else float("nan"),
    }


def _open_h1_limit_after_first_minute_summary(
    accepted_prediction: pd.Series,
    meta_frame: pd.DataFrame,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
    horizon: int,
    trade_cost: float,
    stop_loss: float | None,
) -> tuple[dict[str, float | int], pd.DatetimeIndex]:
    signal_index = pd.DatetimeIndex(accepted_prediction.index)
    entry_index = signal_index + target_interval
    entry_open = pd.to_numeric(raw["open"], errors="coerce").reindex(entry_index)
    entry_open.index = signal_index
    tp = pd.to_numeric(
        meta_frame[f"meta_dynamic_tp_h{int(horizon)}"], errors="coerce"
    ).reindex(signal_index)

    minutes_per_target_bar = int(target_interval / pd.Timedelta(minutes=1))
    if minutes_per_target_bar != 5 or int(horizon) != 3:
        raise ValueError(
            "The open-H1 limit strategy currently requires 5m data and H=3."
        )
    total_minutes = minutes_per_target_bar * int(horizon)
    close_h3 = pd.to_numeric(raw["close"], errors="coerce").shift(-3).reindex(
        signal_index
    )
    frame = pd.DataFrame(
        {
            "entry": entry_open,
            "tp": tp,
            "close_h3": close_h3,
        },
        index=signal_index,
    )
    for minute_offset in range(1, total_minutes):
        timestamp = entry_index + pd.Timedelta(minutes=minute_offset)
        frame[f"low_m{minute_offset + 1}"] = pd.to_numeric(
            minute["low"], errors="coerce"
        ).reindex(timestamp).to_numpy(float)
        frame[f"high_m{minute_offset + 1}"] = pd.to_numeric(
            minute["high"], errors="coerce"
        ).reindex(timestamp).to_numpy(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    entry = frame["entry"].to_numpy(float)
    fill_offset = np.full(len(frame), -1, dtype=int)
    for minute_offset in range(1, minutes_per_target_bar):
        low = frame[f"low_m{minute_offset + 1}"].to_numpy(float)
        newly_filled = (fill_offset < 0) & (low <= entry)
        fill_offset[newly_filled] = minute_offset
    filled = fill_offset >= 0
    unfilled_frame = frame.iloc[np.flatnonzero(~filled)]
    unfilled_close_return = (
        unfilled_frame["close_h3"] / unfilled_frame["entry"] - 1.0
    )
    trades = frame.iloc[np.flatnonzero(filled)].copy()
    trade_fill_offset = fill_offset[filled]
    trade_entry = trades["entry"].to_numpy(float)
    trade_tp = trades["tp"].to_numpy(float)
    tp_price = trade_entry * (1.0 + trade_tp)
    sl_price = (
        trade_entry * (1.0 - float(stop_loss))
        if stop_loss is not None
        else np.full(len(trades), -np.inf)
    )
    gross = trades["close_h3"].to_numpy(float) / trade_entry - 1.0
    exit_code = np.full(len(trades), "close_h3", dtype=object)
    active = np.ones(len(trades), dtype=bool)
    for minute_offset in range(1, total_minutes):
        after_fill = active & (minute_offset >= trade_fill_offset)
        low = trades[f"low_m{minute_offset + 1}"].to_numpy(float)
        high = trades[f"high_m{minute_offset + 1}"].to_numpy(float)
        sl_hit = after_fill & (low <= sl_price)
        tp_allowed = minute_offset >= minutes_per_target_bar
        tp_hit = after_fill & tp_allowed & (high >= tp_price)
        both = sl_hit & tp_hit
        sl_exit = sl_hit
        tp_exit = tp_hit & ~both
        gross[sl_exit] = -float(stop_loss) if stop_loss is not None else gross[sl_exit]
        gross[tp_exit] = trade_tp[tp_exit]
        exit_code[sl_exit] = "sl"
        if minute_offset < 2 * minutes_per_target_bar:
            exit_code[tp_exit] = "tp_h2"
        else:
            exit_code[tp_exit] = "tp_h3"
        active[sl_exit | tp_exit] = False

    tp_h2 = exit_code == "tp_h2"
    tp_h3 = exit_code == "tp_h3"
    sl = exit_code == "sl"
    close_exit = exit_code == "close_h3"
    hit = tp_h2 | tp_h3
    net = gross - float(trade_cost)
    n_accepted = int(len(frame))
    n_filled = int(len(trades))
    summary = {
        "accepted_n": n_accepted,
        "filled_n": n_filled,
        "fill_rate": float(n_filled / n_accepted) if n_accepted else 0.0,
        "unfilled_n": int(len(unfilled_frame)),
        "unfilled_close_h3_mean": float(unfilled_close_return.mean()),
        "unfilled_close_h3_median": float(unfilled_close_return.median()),
        "unfilled_close_h3_q10": float(unfilled_close_return.quantile(0.10)),
        "unfilled_close_h3_q25": float(unfilled_close_return.quantile(0.25)),
        "unfilled_close_h3_q75": float(unfilled_close_return.quantile(0.75)),
        "unfilled_close_h3_q90": float(unfilled_close_return.quantile(0.90)),
        "unfilled_close_h3_positive_rate": float(unfilled_close_return.gt(0.0).mean()),
        "tp_h2_n": int(tp_h2.sum()),
        "tp_h3_n": int(tp_h3.sum()),
        "tp_n": int(hit.sum()),
        "sl_n": int(sl.sum()),
        "close_h3_n": int(close_exit.sum()),
        "gross_mean": float(np.mean(gross)) if n_filled else float("nan"),
        "net_mean": float(np.mean(net)) if n_filled else float("nan"),
        "win_rate": float(np.mean(net > 0.0)) if n_filled else float("nan"),
        "tp_mean": float(trades["tp"].mean()) if n_filled else float("nan"),
        "stop_loss": float(stop_loss) if stop_loss is not None else None,
    }
    return summary, pd.DatetimeIndex(trades.index) + target_interval


def _open_h2_sl_close_h3_summary(
    accepted_prediction: pd.Series,
    raw: pd.DataFrame,
    stop_loss: float,
    take_profit: float | None,
    trade_cost: float,
) -> dict[str, float | int]:
    signal_index = pd.DatetimeIndex(accepted_prediction.index)
    frame = pd.DataFrame(
        {
            "entry": pd.to_numeric(raw["open"], errors="coerce")
            .shift(-2)
            .reindex(signal_index),
            "low_h2": pd.to_numeric(raw["low"], errors="coerce")
            .shift(-2)
            .reindex(signal_index),
            "high_h2": pd.to_numeric(raw["high"], errors="coerce")
            .shift(-2)
            .reindex(signal_index),
            "low_h3": pd.to_numeric(raw["low"], errors="coerce")
            .shift(-3)
            .reindex(signal_index),
            "high_h3": pd.to_numeric(raw["high"], errors="coerce")
            .shift(-3)
            .reindex(signal_index),
            "close_h3": pd.to_numeric(raw["close"], errors="coerce")
            .shift(-3)
            .reindex(signal_index),
        },
        index=signal_index,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    stop_price = frame["entry"] * (1.0 - float(stop_loss))
    tp_price = (
        frame["entry"] * (1.0 + float(take_profit))
        if take_profit is not None
        else pd.Series(np.inf, index=frame.index)
    )
    sl_h2 = frame["low_h2"].le(stop_price)
    tp_h2 = ~sl_h2 & frame["high_h2"].ge(tp_price)
    active_h3 = ~(sl_h2 | tp_h2)
    sl_h3 = active_h3 & frame["low_h3"].le(stop_price)
    tp_h3 = active_h3 & ~sl_h3 & frame["high_h3"].ge(tp_price)
    sl_hit = sl_h2 | sl_h3
    tp_hit = tp_h2 | tp_h3
    gross = frame["close_h3"] / frame["entry"] - 1.0
    gross.loc[sl_hit] = -float(stop_loss)
    if take_profit is not None:
        gross.loc[tp_hit] = float(take_profit)
    net = gross - float(trade_cost)
    close_mask = ~(sl_hit | tp_hit)
    close_only = gross[close_mask]
    return {
        "n": int(len(frame)),
        "tp_h2_n": int(tp_h2.sum()),
        "tp_h3_n": int(tp_h3.sum()),
        "tp_n": int(tp_hit.sum()),
        "sl_h2_n": int(sl_h2.sum()),
        "sl_h3_n": int(sl_h3.sum()),
        "sl_n": int(sl_hit.sum()),
        "close_h3_n": int(close_mask.sum()),
        "close_h3_mean": float(close_only.mean()),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "win_rate": float(net.gt(0.0).mean()),
        "take_profit": float(take_profit) if take_profit is not None else None,
    }


def _accepted_signal_indices(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    metadata, entry = _load_rank(args.meta_archive, args.rank)
    if config.canonical_label_mode(metadata.get("label_mode")) != "meta_learner":
        raise ValueError("--meta-archive must use label_mode=meta_learner.")
    horizons = [int(value) for value in metadata.get("horizons", [])]
    if len(horizons) != 1:
        raise ValueError("The plot currently requires exactly one horizon.")
    horizon = horizons[0]
    split = dict(metadata.get("split_policy", {}))
    val_start = _timestamp(split.get("val_start"), config.VAL_START)
    test_start = _timestamp(split.get("test_start"), config.TEST_START)
    test_end = _timestamp(split.get("test_end"), config.TEST_END)
    wf_end = _timestamp(split.get("wf_end"), val_start)
    purge = config.purge_bars_for_horizons(horizons)

    raw = load_ohlcv(args.data)
    labeled = add_binary_labels(
        raw,
        horizons=horizons,
        label_mode="quantile_trade",
        label_direction="long",
    )
    train, val, test = split_labeled_by_dates(
        labeled,
        val_start=str(val_start),
        test_start=str(test_start),
        test_end=test_end,
        purge_bars=purge,
    )
    folds = make_walk_forward_folds(
        labeled[labeled.index < pd.Timestamp(wf_end)],
        wf_end=str(wf_end),
        min_train_months=int(split.get("wf_min_train_months", config.WF_MIN_TRAIN_MONTHS)),
        val_months=int(split.get("wf_val_months", config.WF_VAL_MONTHS)),
        step_months=int(split.get("wf_step_months", config.WF_STEP_MONTHS)),
        purge_bars=purge,
    )
    if not folds:
        raise ValueError("Archive split policy produced no walk-forward folds.")

    base_archive = Path(str(metadata["meta_base_archive"]))
    base = load_meta_base(
        base_archive,
        int(metadata.get("meta_base_rank", 1)),
        horizons,
    )
    base_windows = required_feature_windows(base.individual)
    logger.info("Building 5m base features | windows=%s", base_windows)
    base_features = build_feature_frame(
        raw,
        windows=base_windows,
        quality_filter=False,
    )
    base_space = CryptoFeatureSpace(
        base_features,
        selectable_features(base_features),
    )

    minute = load_ohlcv(args.data_1m)
    lookahead = int(metadata.get("meta_feature_lookahead_bars", 0))
    include_h1 = bool(metadata.get("meta_feature_include_h1", False))
    alignment = build_meta_feature_alignment(
        labeled.index,
        minute.index,
        include_h1=include_h1,
        lookahead_bars=lookahead,
    )
    meta_individual = CryptoIndividual(
        features=[str(value) for value in entry["features"]],
        generation=int(entry.get("generation", 0) or 0),
        score=float(entry.get("score", np.nan)),
    )
    meta_windows = required_feature_windows(meta_individual)
    logger.info("Building aligned 1m meta features | windows=%s", meta_windows)
    native_features = _build_direct_meta_features(
        minute,
        meta_individual.features,
        alignment.source_index,
    )
    if native_features is None:
        native_quality_index = alignment.source_index_for_targets(
            folds[0].train_df.index
        )
        native_features = build_feature_frame(
            minute,
            windows=meta_windows,
            quality_index=native_quality_index,
            output_index=alignment.source_index,
        )
    else:
        logger.info("Using exact selective builder for %d Rank features.", len(meta_individual.features))
    meta_features = align_meta_feature_frame(native_features, alignment)
    meta_space = CryptoFeatureSpace(
        meta_features,
        selectable_features(meta_features),
    )

    # This recreates the archive's OOF base predictions and final Val/Test targets.
    meta_data = build_meta_learner_data(
        base_labeled_df=labeled,
        original_folds=folds,
        final_train_df=train,
        final_val_df=val,
        final_test_df=test,
        feature_space=base_space,
        base=base,
        min_prediction=float(metadata.get("meta_min_prediction", 0.0002)),
        tp_offset=float(metadata.get("meta_tp_offset", 0.0)),
        meta_val_fraction=float(metadata.get("meta_val_fraction", 0.20)),
        target_start_step=int(metadata.get("meta_target_start_step", 1)),
        purge_bars=purge,
        test_start=str(test_start),
    )

    train_meta = _valid_meta_frame(meta_data.train_df, horizon)
    val_meta = _valid_meta_frame(meta_data.val_df, horizon)
    test_meta = _valid_meta_frame(meta_data.test_df, horizon)
    x_train = meta_space.matrix(meta_individual.features, train_meta.index)
    x_val = meta_space.matrix(meta_individual.features, val_meta.index)
    x_test = meta_space.matrix(meta_individual.features, test_meta.index)
    evaluator = CryptoFitnessEvaluator(horizons=horizons)
    booster = evaluator._train_booster(
        x_train,
        train_meta[f"label_h{horizon}"].astype(int),
        x_val,
        val_meta[f"label_h{horizon}"].astype(int),
    )
    val_prediction = pd.Series(booster.predict(x_val), index=val_meta.index)
    test_prediction = pd.Series(booster.predict(x_test), index=test_meta.index)
    booster.free_dataset()

    top_fraction = float(metadata.get("trade_top_fraction", 0.20))
    trade_cost = float(metadata.get("trade_cost", config.TRADE_COST))
    threshold = _top_fraction_threshold(val_prediction, top_fraction)
    accepted_val_before_ma3 = val_prediction[val_prediction >= threshold]
    accepted_test_before_ma3 = test_prediction[test_prediction >= threshold]
    if args.ma3_filter:
        accepted_val = _filter_positive_ma3_slope(accepted_val_before_ma3, raw)
        accepted_test = _filter_positive_ma3_slope(accepted_test_before_ma3, raw)
    else:
        accepted_val = accepted_val_before_ma3
        accepted_test = accepted_test_before_ma3
    target_interval = alignment.target_interval
    accepted_all = pd.concat([accepted_val, accepted_test])
    accepted_meta_all = pd.concat([val_meta, test_meta])
    accepted_entries = (
        pd.DatetimeIndex(accepted_val.index.append(accepted_test.index))
        + target_interval
    )
    limit_val, limit_val_entries = _open_h1_limit_after_first_minute_summary(
        accepted_val,
        val_meta,
        raw,
        minute,
        target_interval,
        horizon,
        trade_cost,
        args.limit_entry_stop_loss,
    )
    limit_test, limit_test_entries = _open_h1_limit_after_first_minute_summary(
        accepted_test,
        test_meta,
        raw,
        minute,
        target_interval,
        horizon,
        trade_cost,
        args.limit_entry_stop_loss,
    )
    limit_all, limit_all_entries = _open_h1_limit_after_first_minute_summary(
        accepted_all,
        accepted_meta_all,
        raw,
        minute,
        target_interval,
        horizon,
        trade_cost,
        args.limit_entry_stop_loss,
    )
    open_h2_val = _open_h2_sl_close_h3_summary(
        accepted_val, raw, args.stop_loss, args.open_h2_take_profit, trade_cost
    )
    open_h2_test = _open_h2_sl_close_h3_summary(
        accepted_test, raw, args.stop_loss, args.open_h2_take_profit, trade_cost
    )
    open_h2_all = _open_h2_sl_close_h3_summary(
        accepted_all, raw, args.stop_loss, args.open_h2_take_profit, trade_cost
    )
    if args.limit_open_h1_entry:
        accepted_entries = limit_all_entries
    elif args.enter_open_h2_close_h3:
        accepted_entries = (
            pd.DatetimeIndex(accepted_val.index.append(accepted_test.index))
            + 2 * target_interval
        )
    info = {
        "raw": raw,
        "horizon": horizon,
        "top_fraction": top_fraction,
        "threshold": threshold,
        "base_val": len(val_meta),
        "base_test": len(test_meta),
        "accepted_val_before_ma3": len(accepted_val_before_ma3),
        "accepted_test_before_ma3": len(accepted_test_before_ma3),
        "accepted_val": len(accepted_val),
        "accepted_test": len(accepted_test),
        "lookahead": lookahead,
        "minute2_val": _minute_two_open_summary(
            accepted_val, raw, minute, target_interval
        ),
        "minute2_test": _minute_two_open_summary(
            accepted_test, raw, minute, target_interval
        ),
        "minute2_all": _minute_two_open_summary(
            accepted_all,
            raw,
            minute,
            target_interval,
        ),
        "minute2_low_val": _minute_two_low_summary(
            accepted_val, raw, minute, target_interval
        ),
        "minute2_low_test": _minute_two_low_summary(
            accepted_test, raw, minute, target_interval
        ),
        "minute2_low_all": _minute_two_low_summary(
            accepted_all,
            raw,
            minute,
            target_interval,
        ),
        "h1_tp_val": _h1_dynamic_tp_summary(
            accepted_val, val_meta, raw, horizon
        ),
        "h1_tp_test": _h1_dynamic_tp_summary(
            accepted_test, test_meta, raw, horizon
        ),
        "h1_tp_all": _h1_dynamic_tp_summary(
            accepted_all,
            accepted_meta_all,
            raw,
            horizon,
        ),
        "minute1_tp_val": _first_minute_dynamic_tp_summary(
            accepted_val, val_meta, raw, minute, target_interval, horizon
        ),
        "minute1_tp_test": _first_minute_dynamic_tp_summary(
            accepted_test, test_meta, raw, minute, target_interval, horizon
        ),
        "minute1_tp_all": _first_minute_dynamic_tp_summary(
            accepted_all,
            accepted_meta_all,
            raw,
            minute,
            target_interval,
            horizon,
        ),
        "minute2_15_high_val": _minute_2_to_15_high_summary(
            accepted_val, raw, minute, target_interval
        ),
        "minute2_15_high_test": _minute_2_to_15_high_summary(
            accepted_test, raw, minute, target_interval
        ),
        "minute2_15_high_all": _minute_2_to_15_high_summary(
            accepted_all,
            raw,
            minute,
            target_interval,
        ),
        "close_h_val": _close_horizon_summary(
            accepted_val, raw, horizon
        ),
        "close_h_test": _close_horizon_summary(
            accepted_test, raw, horizon
        ),
        "close_h_all": _close_horizon_summary(
            accepted_all, raw, horizon
        ),
        "requested_distributions_val": _requested_trade_distributions(
            accepted_val,
            val_meta,
            raw,
            minute,
            target_interval,
            horizon,
        ),
        "requested_distributions_test": _requested_trade_distributions(
            accepted_test,
            test_meta,
            raw,
            minute,
            target_interval,
            horizon,
        ),
        "requested_distributions_all": _requested_trade_distributions(
            accepted_all,
            accepted_meta_all,
            raw,
            minute,
            target_interval,
            horizon,
        ),
        "ma3_filter_val": _ma3_slope_filter_summary(
            accepted_val_before_ma3, val_meta, raw, horizon, trade_cost
        ),
        "ma3_filter_test": _ma3_slope_filter_summary(
            accepted_test_before_ma3, test_meta, raw, horizon, trade_cost
        ),
        "ma3_filter_all": _ma3_slope_filter_summary(
            pd.concat([accepted_val_before_ma3, accepted_test_before_ma3]),
            accepted_meta_all,
            raw,
            horizon,
            trade_cost,
        ),
        "tp_sl_strategy_val": _dynamic_tp_sl_strategy_summary(
            accepted_val, val_meta, raw, horizon, args.stop_loss, trade_cost
        ),
        "tp_sl_strategy_test": _dynamic_tp_sl_strategy_summary(
            accepted_test, test_meta, raw, horizon, args.stop_loss, trade_cost
        ),
        "tp_sl_strategy_all": _dynamic_tp_sl_strategy_summary(
            accepted_all,
            accepted_meta_all,
            raw,
            horizon,
            args.stop_loss,
            trade_cost,
        ),
        "stop_loss": float(args.stop_loss),
        "ma3_filter_enabled": bool(args.ma3_filter),
        "limit_open_h1_entry_enabled": bool(args.limit_open_h1_entry),
        "limit_open_h1_val": limit_val,
        "limit_open_h1_test": limit_test,
        "limit_open_h1_all": limit_all,
        "enter_open_h2_enabled": bool(args.enter_open_h2_close_h3),
        "open_h2_strategy_val": open_h2_val,
        "open_h2_strategy_test": open_h2_test,
        "open_h2_strategy_all": open_h2_all,
    }
    return pd.DataFrame(index=accepted_entries), info


def _draw_candles(
    ax: plt.Axes,
    frame: pd.DataFrame,
    accepted: pd.DatetimeIndex,
) -> int:
    accepted_set = set(accepted)
    price_span = float(frame["high"].max() - frame["low"].min())
    min_body = max(price_span * 0.0008, 1e-9)
    highlighted = 0
    for x, (timestamp, row) in enumerate(frame.iterrows()):
        selected = timestamp in accepted_set
        color = "#f6c945" if selected else "#f2f3f5"
        if selected:
            highlighted += 1
        ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.65)
        lower = min(float(row["open"]), float(row["close"]))
        height = max(abs(float(row["close"]) - float(row["open"])), min_body)
        ax.add_patch(
            Rectangle(
                (x - 0.32, lower),
                0.64,
                height,
                facecolor=color if selected else "none",
                edgecolor=color,
                linewidth=0.7,
            )
        )
    tick_count = min(7, len(frame))
    ticks = np.linspace(0, len(frame) - 1, tick_count, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [frame.index[pos].strftime("%Y-%m-%d\n%H:%M") for pos in ticks],
        fontsize=8,
    )
    ax.grid(color="#39414d", linewidth=0.45, alpha=0.65)
    ax.tick_params(colors="#d9dde5", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#596273")
    return highlighted


def _plot(
    accepted_frame: pd.DataFrame,
    info: dict,
    args: argparse.Namespace,
) -> Path:
    raw = info["raw"]
    accepted = pd.DatetimeIndex(accepted_frame.index).intersection(raw.index)
    if accepted.empty:
        raise ValueError("No accepted H1 timestamps overlap the 5m data.")
    rng = np.random.default_rng(args.seed)
    positions = raw.index.get_indexer(accepted)
    positions = positions[positions >= 0]
    starts: list[int] = []
    order = rng.permutation(positions)
    for center in order:
        start = int(np.clip(center - args.bars // 2, 0, max(len(raw) - args.bars, 0)))
        if all(abs(start - old) >= args.bars // 2 for old in starts):
            starts.append(start)
        if len(starts) == args.panels:
            break
    while len(starts) < args.panels:
        center = int(rng.choice(positions))
        starts.append(
            int(np.clip(center - args.bars // 2, 0, max(len(raw) - args.bars, 0)))
        )

    fig, axes = plt.subplots(
        args.panels,
        1,
        figsize=(18, 3.35 * args.panels),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#11151b")
    entry_label = "H2 entry" if info["enter_open_h2_enabled"] else "H1 entry"
    total_visible = 0
    for panel, (ax, start) in enumerate(zip(axes, starts), start=1):
        ax.set_facecolor("#11151b")
        sample = raw.iloc[start : start + args.bars]
        visible = _draw_candles(ax, sample, accepted)
        total_visible += visible
        ax.set_title(
            f"Panel {panel} | {sample.index[0]} to {sample.index[-1]} | "
            f"accepted {entry_label}={visible}",
            color="#f2f3f5",
            fontsize=10,
            loc="left",
        )
        ax.set_ylabel("BTCUSDT", color="#d9dde5")
    filters = ["MFE Q20", "after-1m Meta"]
    if info["ma3_filter_enabled"]:
        filters.append("slopeMA3 > 0")
    if info["limit_open_h1_entry_enabled"]:
        filters.append("filled BUY LIMIT at open H1 in minutes 2-5")
    elif info["enter_open_h2_enabled"]:
        filters.append("Long at open H2, TP/SL, fallback close H3")
    fig.suptitle(
        " + ".join(filters) + " entries\n"
        f"Meta Rank {args.rank} | top {100 * info['top_fraction']:.0f}% "
        f"(Val threshold={info['threshold']:.6f}) | "
        f"Val {info['accepted_val']:,}/{info['base_val']:,} | "
        f"Test {info['accepted_test']:,}/{info['base_test']:,} | "
        f"yellow = 5m {entry_label} candle | visible={total_visible}",
        color="#f2f3f5",
        fontsize=13,
    )
    axes[0].legend(
        handles=[
            Patch(
                facecolor="#f6c945",
                edgecolor="#f6c945",
                label=f"Accepted {entry_label}",
            )
        ],
        loc="upper right",
        frameon=False,
        labelcolor="#f2f3f5",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return args.out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-archive",
        type=Path,
        default=Path(
            "crypto/results/crypto_btc_5m_meta_after_1m_mfe_q20_h3_seed1_1h.json"
        ),
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--data", type=Path, default=Path("data/crypto/BTCUSDT_5m.csv"))
    parser.add_argument(
        "--data-1m", type=Path, default=Path("data/crypto/BTCUSDT_1m.csv")
    )
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.002,
        help="Stop loss measured from open H1 (default: 0.002 = 0.20%%).",
    )
    parser.add_argument(
        "--ma3-filter",
        action="store_true",
        help="Require observable 5m slopeMA3 > 0 at the signal candle.",
    )
    parser.add_argument(
        "--limit-open-h1-entry",
        action="store_true",
        help=(
            "After observing minute 1, require a BUY LIMIT at open H1 to fill "
            "during minutes 2-5; evaluate TP only in H2-H3."
        ),
    )
    parser.add_argument(
        "--limit-entry-stop-loss",
        type=float,
        default=None,
        help=(
            "Optional SL measured from open H1, active from the exact 1m fill "
            "onward (example: 0.001 = 0.10%%)."
        ),
    )
    parser.add_argument(
        "--enter-open-h2-close-h3",
        action="store_true",
        help=(
            "Enter every Meta-accepted signal at open H2, apply --stop-loss "
            "through H2-H3, and otherwise exit at close H3."
        ),
    )
    parser.add_argument(
        "--open-h2-take-profit",
        type=float,
        default=None,
        help=(
            "Optional fixed TP measured from open H2 for the open-H2 strategy "
            "(example: 0.0015 = 0.15%%)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("temp/output/meta_after_1m_accepted_h1.png"),
    )
    args = parser.parse_args()
    if args.rank < 1 or args.bars < 20 or args.panels < 1:
        parser.error("rank/panels must be positive and bars must be at least 20.")
    if not 0.0 < args.stop_loss < 1.0:
        parser.error("--stop-loss must be between 0 and 1.")
    if args.limit_entry_stop_loss is not None and not (
        0.0 < args.limit_entry_stop_loss < 1.0
    ):
        parser.error("--limit-entry-stop-loss must be between 0 and 1.")
    if args.limit_open_h1_entry and args.enter_open_h2_close_h3:
        parser.error(
            "--limit-open-h1-entry and --enter-open-h2-close-h3 are mutually exclusive."
        )
    if args.open_h2_take_profit is not None and not (
        0.0 < args.open_h2_take_profit < 1.0
    ):
        parser.error("--open-h2-take-profit must be between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()
    accepted, info = _accepted_signal_indices(args)
    output = _plot(accepted, info, args)
    filter_name = " after slopeMA3 > 0" if info["ma3_filter_enabled"] else ""
    print(
        f"Accepted H1{filter_name} | "
        f"Val={info['accepted_val']:,}/{info['accepted_val_before_ma3']:,} | "
        f"Test={info['accepted_test']:,}/{info['accepted_test_before_ma3']:,} | "
        f"threshold={info['threshold']:.8f}"
    )
    print("\nOpen minute 2 / open H1 - 1:")
    for split, key in (("val", "minute2_val"), ("test", "minute2_test"), ("all", "minute2_all")):
        row = info[key]
        print(
            f"{split:>4} n={row['n']:,} | mean={100 * row['mean']:+.4f}% | "
            f"Q10={100 * row['q10']:+.4f}% Q25={100 * row['q25']:+.4f}% "
            f"median={100 * row['median']:+.4f}% Q75={100 * row['q75']:+.4f}% "
            f"Q90={100 * row['q90']:+.4f}% | "
            f"P(>0)={100 * row['positive_rate']:.2f}% P(<0)={100 * row['negative_rate']:.2f}%"
        )
    print("\nLow minute 2 / open H1 - 1:")
    for split, key in (
        ("val", "minute2_low_val"),
        ("test", "minute2_low_test"),
        ("all", "minute2_low_all"),
    ):
        row = info[key]
        print(
            f"{split:>4} n={row['n']:,} | mean={100 * row['mean']:+.4f}% | "
            f"min={100 * row['minimum']:+.4f}% Q1={100 * row['q01']:+.4f}% "
            f"Q5={100 * row['q05']:+.4f}% Q10={100 * row['q10']:+.4f}% "
            f"Q25={100 * row['q25']:+.4f}% median={100 * row['median']:+.4f}% "
            f"Q75={100 * row['q75']:+.4f}% Q80={100 * row['q80']:+.4f}% "
            f"Q90={100 * row['q90']:+.4f}% | "
            f"P(<-0.05%)={100 * row['below_005']:.2f}% "
            f"P(<-0.10%)={100 * row['below_010']:.2f}% "
            f"P(<-0.15%)={100 * row['below_015']:.2f}%"
        )
    print("\nHigh H1 > dynamic TP:")
    for split, key in (
        ("val", "h1_tp_val"),
        ("test", "h1_tp_test"),
        ("all", "h1_tp_all"),
    ):
        row = info[key]
        print(
            f"{split:>4} hit={row['hit_n']:,}/{row['n']:,} "
            f"({100 * row['hit_rate']:.2f}%) | "
            f"TP mean all={100 * row['tp_mean_all']:.4f}% "
            f"median all={100 * row['tp_median_all']:.4f}% | "
            f"TP mean hit={100 * row['tp_mean_hit']:.4f}% "
            f"median hit={100 * row['tp_median_hit']:.4f}%"
        )
    print("\nDynamic TP distribution for accepted trades:")
    for split, key in (
        ("val", "h1_tp_val"),
        ("test", "h1_tp_test"),
        ("all", "h1_tp_all"),
    ):
        row = info[key]
        print(
            f"{split:>4} n={row['n']:,} | mean={100 * row['tp_mean_all']:.4f}% "
            f"min={100 * row['tp_min_all']:.4f}% | "
            f"Q1={100 * row['tp_q01_all']:.4f}% Q5={100 * row['tp_q05_all']:.4f}% "
            f"Q10={100 * row['tp_q10_all']:.4f}% Q20={100 * row['tp_q20_all']:.4f}% "
            f"Q25={100 * row['tp_q25_all']:.4f}% median={100 * row['tp_median_all']:.4f}% "
            f"Q75={100 * row['tp_q75_all']:.4f}% Q80={100 * row['tp_q80_all']:.4f}% "
            f"Q90={100 * row['tp_q90_all']:.4f}% Q95={100 * row['tp_q95_all']:.4f}% "
            f"Q99={100 * row['tp_q99_all']:.4f}% max={100 * row['tp_max_all']:.4f}%"
        )
    print("\nHigh minute 1 > dynamic TP:")
    for split, key in (
        ("val", "minute1_tp_val"),
        ("test", "minute1_tp_test"),
        ("all", "minute1_tp_all"),
    ):
        row = info[key]
        print(
            f"{split:>4} hit={row['hit_n']:,}/{row['n']:,} "
            f"({100 * row['hit_rate']:.2f}%) | "
            f"TP mean hit={100 * row['tp_mean_hit']:.4f}%"
        )
    print("\nMax high minute 2..15 / open H1 - 1:")
    for split, key in (
        ("val", "minute2_15_high_val"),
        ("test", "minute2_15_high_test"),
        ("all", "minute2_15_high_all"),
    ):
        row = info[key]
        print(
            f"{split:>4} n={row['n']:,} | mean={100 * row['mean']:+.4f}% "
            f"min={100 * row['minimum']:+.4f}% | "
            f"Q1={100 * row['q01']:+.4f}% Q5={100 * row['q05']:+.4f}% "
            f"Q10={100 * row['q10']:+.4f}% Q20={100 * row['q20']:+.4f}% "
            f"Q25={100 * row['q25']:+.4f}% median={100 * row['median']:+.4f}% "
            f"Q75={100 * row['q75']:+.4f}% Q80={100 * row['q80']:+.4f}% "
            f"Q90={100 * row['q90']:+.4f}% Q95={100 * row['q95']:+.4f}% "
            f"Q99={100 * row['q99']:+.4f}% max={100 * row['maximum']:+.4f}%"
        )
    print(f"\nClose H{info['horizon']} / open H1 - 1:")
    for split, key in (("val", "close_h_val"), ("test", "close_h_test"), ("all", "close_h_all")):
        row = info[key]
        print(
            f"{split:>4} n={row['n']:,} | mean={100 * row['mean']:+.4f}% | "
            f"Q10={100 * row['q10']:+.4f}% Q25={100 * row['q25']:+.4f}% "
            f"median={100 * row['median']:+.4f}% Q75={100 * row['q75']:+.4f}% "
            f"Q90={100 * row['q90']:+.4f}% | "
            f"P(>0)={100 * row['positive_rate']:.2f}% P(<0)={100 * row['negative_rate']:.2f}%"
        )
    print("\nRequested accepted-trade distributions (all returns vs open H1):")
    labels = (
        ("low minute 1", "low_minute_1"),
        ("high minute 1", "high_minute_1"),
        ("dynamic TP", "dynamic_tp"),
        ("open minute 2", "open_minute_2"),
        ("low H2", "low_h2"),
        (f"close H{info['horizon']}", "close_h"),
    )
    for split, info_key in (
        ("val", "requested_distributions_val"),
        ("test", "requested_distributions_test"),
        ("all", "requested_distributions_all"),
    ):
        print(f"  [{split}]")
        for label, value_key in labels:
            row = info[info_key][value_key]
            print(
                f"    {label:<14} n={row['n']:,} mean={100 * row['mean']:+.4f}% "
                f"min={100 * row['minimum']:+.4f}% | "
                f"Q1={100 * row['q01']:+.4f}% Q5={100 * row['q05']:+.4f}% "
                f"Q10={100 * row['q10']:+.4f}% Q20={100 * row['q20']:+.4f}% "
                f"Q25={100 * row['q25']:+.4f}% Q50={100 * row['median']:+.4f}% "
                f"Q75={100 * row['q75']:+.4f}% Q80={100 * row['q80']:+.4f}% "
                f"Q90={100 * row['q90']:+.4f}% Q95={100 * row['q95']:+.4f}% "
                f"Q99={100 * row['q99']:+.4f}% max={100 * row['maximum']:+.4f}%"
            )
    print("\nObservable 5m MA3 slope > 0 filter:")
    for split, info_key in (
        ("val", "ma3_filter_val"),
        ("test", "ma3_filter_test"),
        ("all", "ma3_filter_all"),
    ):
        row = info[info_key]
        print(
            f"{split:>4} kept={row['after_n']:,}/{row['before_n']:,} "
            f"({100 * row['kept_rate']:.2f}%) | "
            f"hit {100 * row['before_hit_rate']:.2f}% -> "
            f"{100 * row['after_hit_rate']:.2f}% | "
            f"gross {100 * row['before_gross_mean']:+.4f}% -> "
            f"{100 * row['after_gross_mean']:+.4f}% | "
            f"E[net] {100 * row['before_net_mean']:+.4f}% -> "
            f"{100 * row['after_net_mean']:+.4f}% | "
            f"TP mean {100 * row['before_tp_mean']:.4f}% -> "
            f"{100 * row['after_tp_mean']:.4f}%"
        )
    if info["limit_open_h1_entry_enabled"]:
        limit_sl = info["limit_open_h1_all"]["stop_loss"]
        sl_text = (
            f"; SL={100 * limit_sl:.3f}% from open H1, active after fill"
            if limit_sl is not None
            else "; no SL"
        )
        print(
            "\nBUY LIMIT at open H1 during minutes 2-5; "
            f"dynamic TP evaluated only in H2-H3{sl_text}:"
        )
        for split, info_key in (
            ("val", "limit_open_h1_val"),
            ("test", "limit_open_h1_test"),
            ("all", "limit_open_h1_all"),
        ):
            row = info[info_key]
            print(
                f"{split:>4} accepted={row['accepted_n']:,} | "
                f"filled={row['filled_n']:,} ({100 * row['fill_rate']:.2f}%) | "
                f"TP H2={row['tp_h2_n']:,} "
                f"({100 * row['tp_h2_n'] / max(1, row['filled_n']):.2f}%) | "
                f"TP H3={row['tp_h3_n']:,} "
                f"({100 * row['tp_h3_n'] / max(1, row['filled_n']):.2f}%) | "
                f"all TP={row['tp_n']:,} "
                f"({100 * row['tp_n'] / max(1, row['filled_n']):.2f}%) | "
                f"SL={row['sl_n']:,} "
                f"({100 * row['sl_n'] / max(1, row['filled_n']):.2f}%) | "
                f"close H3={row['close_h3_n']:,} "
                f"({100 * row['close_h3_n'] / max(1, row['filled_n']):.2f}%) | "
                f"TP mean={100 * row['tp_mean']:.4f}% | "
                f"gross={100 * row['gross_mean']:+.4f}% | "
                f"E[net]={100 * row['net_mean']:+.4f}% | "
                f"win={100 * row['win_rate']:.2f}%"
            )
            print(
                f"     unfilled={row['unfilled_n']:,} | hypothetical close H3 "
                f"mean={100 * row['unfilled_close_h3_mean']:+.4f}% "
                f"median={100 * row['unfilled_close_h3_median']:+.4f}% | "
                f"Q10={100 * row['unfilled_close_h3_q10']:+.4f}% "
                f"Q25={100 * row['unfilled_close_h3_q25']:+.4f}% "
                f"Q75={100 * row['unfilled_close_h3_q75']:+.4f}% "
                f"Q90={100 * row['unfilled_close_h3_q90']:+.4f}% | "
                f"P(>0)={100 * row['unfilled_close_h3_positive_rate']:.2f}%"
            )
    elif info["enter_open_h2_enabled"]:
        open_h2_tp = info["open_h2_strategy_all"]["take_profit"]
        tp_text = (
            f"TP={100 * open_h2_tp:.3f}%"
            if open_h2_tp is not None
            else "no TP"
        )
        print(
            f"\nLong at open H2; SL={100 * info['stop_loss']:.3f}% from open H2; "
            f"{tp_text}; otherwise exit at close H3; same candle SL first:"
        )
        for split, info_key in (
            ("val", "open_h2_strategy_val"),
            ("test", "open_h2_strategy_test"),
            ("all", "open_h2_strategy_all"),
        ):
            row = info[info_key]
            print(
                f"{split:>4} n={row['n']:,} | "
                f"TP H2={row['tp_h2_n']:,} "
                f"({100 * row['tp_h2_n'] / max(1, row['n']):.2f}%) | "
                f"TP H3={row['tp_h3_n']:,} "
                f"({100 * row['tp_h3_n'] / max(1, row['n']):.2f}%) | "
                f"all TP={row['tp_n']:,} "
                f"({100 * row['tp_n'] / max(1, row['n']):.2f}%) | "
                f"SL H2={row['sl_h2_n']:,} "
                f"({100 * row['sl_h2_n'] / max(1, row['n']):.2f}%) | "
                f"SL H3={row['sl_h3_n']:,} "
                f"({100 * row['sl_h3_n'] / max(1, row['n']):.2f}%) | "
                f"all SL={row['sl_n']:,} "
                f"({100 * row['sl_n'] / max(1, row['n']):.2f}%) | "
                f"close H3={row['close_h3_n']:,} "
                f"({100 * row['close_h3_n'] / max(1, row['n']):.2f}%) "
                f"mean={100 * row['close_h3_mean']:+.4f}% | "
                f"gross={100 * row['gross_mean']:+.4f}% | "
                f"E[net]={100 * row['net_mean']:+.4f}% | "
                f"win={100 * row['win_rate']:.2f}%"
            )
    else:
        ma3_text = " after slopeMA3 > 0" if info["ma3_filter_enabled"] else ""
        print(
            f"\nDynamic TP + SL {100 * info['stop_loss']:.3f}% from open H1"
            f"{ma3_text} (same 5m candle: SL first):"
        )
        for split, info_key in (
            ("val", "tp_sl_strategy_val"),
            ("test", "tp_sl_strategy_test"),
            ("all", "tp_sl_strategy_all"),
        ):
            row = info[info_key]
            print(
                f"{split:>4} n={row['n']:,} | "
                f"TP={row['tp_n']:,} ({100 * row['tp_n'] / max(1, row['n']):.2f}%) | "
                f"SL={row['sl_n']:,} ({100 * row['sl_n'] / max(1, row['n']):.2f}%) | "
                f"close H{info['horizon']}={row['close_n']:,} "
                f"({100 * row['close_n'] / max(1, row['n']):.2f}%) | "
                f"gross={100 * row['gross_mean']:+.4f}% | "
                f"E[net]={100 * row['net_mean']:+.4f}% | "
                f"win={100 * row['win_rate']:.2f}%"
            )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
