"""Safe crypto feature construction.

The feature pool uses all available raw columns, but avoids selectable raw
price/volume scale. Every selectable feature is a return, ratio, rolling
normalization, bounded oscillator, or interaction of normalized quantities.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from crypto import config


logger = logging.getLogger(__name__)


RAW_SCALE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "trade_size_base",
    "trade_size_quote",
}


def build_feature_frame(
    df: pd.DataFrame,
    windows: list[int] | tuple[int, ...] = tuple(config.WINDOWS),
    min_valid_ratio: float = config.FEATURE_MIN_VALID_RATIO,
    max_dominant_value_ratio: float = config.FEATURE_MAX_DOMINANT_VALUE_RATIO,
    quality_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Return a safe feature matrix aligned to df.index."""
    start_time = time.time()
    data = df.sort_index()
    windows = sorted({int(w) for w in windows if int(w) > 1})
    logger.info(
        "Crypto feature build: rows=%d | windows=%s | quality_rows=%s",
        len(data),
        windows,
        len(quality_index) if quality_index is not None else "all",
    )
    open_ = _num(data["open"])
    high = _num(data["high"])
    low = _num(data["low"])
    close = _num(data["close"])
    volume = _num(data["volume"])
    trade_count = _num(data["trade_count"])
    taker_base = _num(data["taker_buy_base_volume"])
    taker_quote = _num(data["taker_buy_quote_volume"])

    quote_volume_proxy = close * volume
    trade_count_safe = trade_count.where(trade_count > 0)
    true_range = _true_range(high, low, close)
    close_ret_1 = close.pct_change()
    logret_1 = np.log(close).diff()
    log_hl = np.log(_safe_div(high, low))
    log_co = np.log(_safe_div(close, open_))
    log_hc = np.log(_safe_div(high, close))
    log_ho = np.log(_safe_div(high, open_))
    log_lc = np.log(_safe_div(low, close))
    log_lo = np.log(_safe_div(low, open_))
    upside_logret = logret_1.clip(lower=0.0)
    downside_logret = logret_1.clip(upper=0.0)
    taker_sell_base = volume - taker_base
    taker_sell_quote = quote_volume_proxy - taker_quote
    taker_delta_base = taker_base - taker_sell_base
    taker_delta_quote = taker_quote - taker_sell_quote
    signed_volume = volume * np.sign(close_ret_1)

    features: dict[str, pd.Series] = {
        "ret_close_1": close_ret_1,
        "logret_close_1": logret_1,
        "body_pct": _safe_div(close - open_, open_),
        "range_pct": _safe_div(high - low, open_),
        "gap_pct": _safe_div(open_ - close.shift(1), close.shift(1)),
        "upper_wick_pct": _safe_div(high - np.maximum(open_, close), open_),
        "lower_wick_pct": _safe_div(np.minimum(open_, close) - low, open_),
        "close_pos_in_range": _safe_div(close - low, high - low),
        "taker_buy_base_ratio": _safe_div(taker_base, volume),
        "taker_buy_quote_ratio": _safe_div(taker_quote, quote_volume_proxy),
        "taker_delta_base_ratio": _safe_div(taker_delta_base, volume),
        "taker_delta_quote_ratio": _safe_div(taker_delta_quote, quote_volume_proxy),
        "trade_size_base": _safe_div(volume, trade_count_safe),
        "trade_size_quote": _safe_div(quote_volume_proxy, trade_count_safe),
    }
    features["buy_pressure_base"] = (2.0 * features["taker_buy_base_ratio"]) - 1.0
    features["buy_pressure_quote"] = (2.0 * features["taker_buy_quote_ratio"]) - 1.0
    features["buy_pressure_diff"] = (
        features["buy_pressure_base"] - features["buy_pressure_quote"]
    )

    flow_sources = {
        "volume": volume,
        "trade_count": trade_count,
        "taker_buy_base_volume": taker_base,
        "taker_buy_quote_volume": taker_quote,
        "quote_volume_proxy": quote_volume_proxy,
        "trade_size_base": features["trade_size_base"],
        "trade_size_quote": features["trade_size_quote"],
    }

    logger.info("Crypto feature build: base features=%d", len(features))

    for idx, w in enumerate(windows, start=1):
        window_start = time.time()
        before_count = len(features)
        logger.info(
            "Crypto feature build: window %d/%d w=%d started.",
            idx,
            len(windows),
            w,
        )
        ma = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std()
        ret_w = close.pct_change(w)
        logret_w = np.log(close).diff(w)
        vol_w = logret_1.rolling(w, min_periods=w).std()
        tr_ma = true_range.rolling(w, min_periods=w).mean()

        features[f"ret_close_{w}"] = ret_w
        features[f"logret_close_{w}"] = logret_w
        features[f"rolling_ret_mean_{w}"] = close_ret_1.rolling(w, min_periods=w).mean()
        features[f"rolling_logret_mean_{w}"] = logret_1.rolling(w, min_periods=w).mean()
        features[f"volatility_{w}"] = vol_w
        realized_var_mean = logret_1.pow(2).rolling(w, min_periods=w).mean()
        realized_var_sum = logret_1.pow(2).rolling(w, min_periods=w).sum()
        upside_var = upside_logret.pow(2).rolling(w, min_periods=w).mean()
        downside_var = downside_logret.pow(2).rolling(w, min_periods=w).mean()
        parkinson_var = (log_hl.pow(2) / (4.0 * np.log(2.0))).rolling(
            w, min_periods=w
        ).mean()
        gk_var = (
            0.5 * log_hl.pow(2)
            - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
        ).rolling(w, min_periods=w).mean()
        rs_var = (
            (log_hc * log_ho) + (log_lc * log_lo)
        ).rolling(w, min_periods=w).mean()
        realized_vol = _sqrt_nonnegative(realized_var_mean)
        downside_vol = _sqrt_nonnegative(downside_var)
        upside_vol = _sqrt_nonnegative(upside_var)
        features[f"realized_vol_{w}"] = realized_vol
        features[f"realized_vol_sum_{w}"] = _sqrt_nonnegative(realized_var_sum)
        features[f"downside_realized_vol_{w}"] = downside_vol
        features[f"upside_realized_vol_{w}"] = upside_vol
        features[f"up_down_realized_vol_ratio_{w}"] = _safe_div(upside_vol, downside_vol) - 1.0
        features[f"vol_of_vol_{w}"] = vol_w.rolling(w, min_periods=w).std()
        features[f"realized_vol_z_{w}"] = _rolling_zscore(realized_vol, w)
        features[f"parkinson_vol_{w}"] = _sqrt_nonnegative(parkinson_var)
        features[f"garman_klass_vol_{w}"] = _sqrt_nonnegative(gk_var)
        features[f"rogers_satchell_vol_{w}"] = _sqrt_nonnegative(rs_var)
        features[f"gap_vol_{w}"] = features["gap_pct"].rolling(w, min_periods=w).std()
        features[f"ma_ratio_close_{w}"] = _safe_div(close, ma) - 1.0
        features[f"ts_zscore_close_{w}"] = _safe_div(close - ma, std)
        features[f"drawdown_{w}"] = _safe_div(close, close.rolling(w, min_periods=w).max()) - 1.0
        features[f"runup_{w}"] = _safe_div(close, close.rolling(w, min_periods=w).min()) - 1.0
        features[f"bb_pos_{w}"] = _safe_div(close - ma, 2.0 * std)
        features[f"bb_width_{w}"] = _safe_div(4.0 * std, ma)
        features[f"atr_pct_{w}"] = _safe_div(tr_ma, close)
        features[f"range_pct_mean_{w}"] = features["range_pct"].rolling(w, min_periods=w).mean()
        features[f"body_pct_mean_{w}"] = features["body_pct"].rolling(w, min_periods=w).mean()
        features[f"rsi_{w}"] = _rsi(close, w)
        features[f"efficiency_ratio_{w}"] = _efficiency_ratio(close, w)
        features[f"close_ts_rank_{w}"] = _rolling_rank_pct(close, w)
        features[f"ret_ts_rank_{w}"] = _rolling_rank_pct(close_ret_1, w)
        features[f"max_ret_{w}"] = close_ret_1.rolling(w, min_periods=w).max()
        features[f"min_ret_{w}"] = close_ret_1.rolling(w, min_periods=w).min()
        features[f"skew_ret_{w}"] = close_ret_1.rolling(w, min_periods=w).skew()
        features[f"kurt_ret_{w}"] = close_ret_1.rolling(w, min_periods=w).kurt()

        buy_pressure = features["buy_pressure_base"]
        features[f"buy_pressure_mean_{w}"] = buy_pressure.rolling(w, min_periods=w).mean()
        features[f"buy_pressure_z_{w}"] = _rolling_zscore(buy_pressure, w)
        features[f"buy_pressure_positive_ratio_{w}"] = (
            (buy_pressure > 0).astype(float).rolling(w, min_periods=w).mean()
        )
        features[f"buy_pressure_negative_ratio_{w}"] = (
            (buy_pressure < 0).astype(float).rolling(w, min_periods=w).mean()
        )
        features[f"buy_pressure_persistence_{w}"] = _persistence_ratio(buy_pressure, w)
        features[f"taker_ratio_mean_{w}"] = features["taker_buy_base_ratio"].rolling(w, min_periods=w).mean()
        features[f"taker_ratio_z_{w}"] = _rolling_zscore(features["taker_buy_base_ratio"], w)
        features[f"taker_delta_sum_ratio_{w}"] = _safe_div(
            taker_delta_base.rolling(w, min_periods=w).sum(),
            volume.rolling(w, min_periods=w).sum(),
        )
        features[f"taker_delta_quote_sum_ratio_{w}"] = _safe_div(
            taker_delta_quote.rolling(w, min_periods=w).sum(),
            quote_volume_proxy.rolling(w, min_periods=w).sum(),
        )
        features[f"taker_delta_z_{w}"] = _rolling_zscore(
            features["taker_delta_base_ratio"], w
        )
        features[f"taker_delta_quote_z_{w}"] = _rolling_zscore(
            features["taker_delta_quote_ratio"], w
        )
        features[f"taker_delta_accel_{w}"] = features[f"taker_delta_sum_ratio_{w}"].diff(w)
        features[f"signed_volume_sum_ratio_{w}"] = _safe_div(
            signed_volume.rolling(w, min_periods=w).sum(),
            volume.rolling(w, min_periods=w).sum(),
        )

        for name, source in flow_sources.items():
            log_source = np.log1p(source.clip(lower=0))
            rolling_mean = source.rolling(w, min_periods=w).mean()
            features[f"{name}_ratio_{w}"] = _safe_div(source, rolling_mean) - 1.0
            features[f"{name}_log_z_{w}"] = _rolling_zscore(log_source, w)
            features[f"{name}_log_delta_{w}"] = log_source.diff(w)

        features[f"ret_x_volume_z_{w}"] = close_ret_1 * features[f"volume_log_z_{w}"]
        features[f"ret_x_trade_count_z_{w}"] = close_ret_1 * features[f"trade_count_log_z_{w}"]
        features[f"ret_x_buy_pressure_{w}"] = close_ret_1 * features[f"buy_pressure_z_{w}"]
        features[f"range_x_volume_z_{w}"] = features["range_pct"] * features[f"volume_log_z_{w}"]
        features[f"volatility_x_buy_pressure_{w}"] = features[f"volatility_{w}"] * features[f"buy_pressure_z_{w}"]
        features[f"vol_regime_z_{w}"] = _rolling_zscore(vol_w, w)
        features[f"imbalance_x_high_volume_{w}"] = (
            features[f"taker_delta_z_{w}"] * features[f"volume_log_z_{w}"].clip(lower=0.0)
        )
        features[f"imbalance_x_high_volatility_{w}"] = (
            features[f"taker_delta_z_{w}"] * features[f"vol_regime_z_{w}"].clip(lower=0.0)
        )
        features[f"imbalance_x_trend_{w}"] = (
            features[f"taker_delta_z_{w}"] * features[f"ma_ratio_close_{w}"]
        )
        features[f"imbalance_return_corr_{w}"] = (
            features["taker_delta_base_ratio"].rolling(w, min_periods=w).corr(close_ret_1)
        )
        features[f"imbalance_volume_corr_{w}"] = (
            features["taker_delta_base_ratio"]
            .rolling(w, min_periods=w)
            .corr(features[f"volume_log_z_{w}"])
        )
        features[f"imbalance_divergence_{w}"] = (
            features[f"taker_delta_z_{w}"] - features[f"ret_ts_rank_{w}"]
        )
        logger.info(
            "Crypto feature build: window %d/%d w=%d done | added=%d | total=%d | %.1fs",
            idx,
            len(windows),
            w,
            len(features) - before_count,
            len(features),
            time.time() - window_start,
        )

    logger.info("Crypto feature build: assembling DataFrame with %d columns.", len(features))
    feature_df = pd.DataFrame(features, index=data.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    filtered = _quality_filter(
        feature_df,
        min_valid_ratio=float(min_valid_ratio),
        max_dominant_value_ratio=float(max_dominant_value_ratio),
        quality_index=quality_index,
    )
    logger.info(
        "Crypto feature build: kept %d/%d features after quality filter | total %.1fs",
        len(filtered.columns),
        len(feature_df.columns),
        time.time() - start_time,
    )
    return filtered


def selectable_features(feature_df: pd.DataFrame) -> list[str]:
    return [
        col for col in feature_df.columns
        if col not in RAW_SCALE_COLUMNS and not col.startswith("date")
    ]


def _quality_filter(
    feature_df: pd.DataFrame,
    min_valid_ratio: float,
    max_dominant_value_ratio: float,
    quality_index: pd.Index | None = None,
) -> pd.DataFrame:
    start_time = time.time()
    kept = []
    quality_df = feature_df if quality_index is None else feature_df.loc[quality_index]
    n_rows = max(len(quality_df), 1)
    logger.info(
        "Crypto feature quality filter: checking %d columns on %d rows.",
        len(feature_df.columns),
        len(quality_df),
    )
    for col in feature_df.columns:
        series = pd.to_numeric(quality_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_ratio = float(series.notna().mean())
        if valid_ratio < min_valid_ratio:
            continue
        valid = series.dropna()
        if valid.nunique(dropna=True) < 2:
            continue
        dominant = float(valid.value_counts(normalize=True, dropna=True).iloc[0])
        if dominant > max_dominant_value_ratio:
            continue
        if len(valid) < max(20, int(0.01 * n_rows)):
            continue
        kept.append(col)
    logger.info(
        "Crypto feature quality filter: kept=%d dropped=%d | %.1fs",
        len(kept),
        len(feature_df.columns) - len(kept),
        time.time() - start_time,
    )
    return feature_df[kept].copy()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_div(num, den) -> pd.Series:
    num_s = pd.Series(num)
    den_s = pd.Series(den).replace(0, np.nan)
    return (num_s / den_s).replace([np.inf, -np.inf], np.nan)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    parts = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return _safe_div(series - mean, std)


def _sqrt_nonnegative(series: pd.Series) -> pd.Series:
    return np.sqrt(pd.to_numeric(series, errors="coerce").clip(lower=0.0))


def _persistence_ratio(series: pd.Series, window: int) -> pd.Series:
    signs = np.sign(series)
    pos_ratio = (signs > 0).astype(float).rolling(window, min_periods=window).mean()
    neg_ratio = (signs < 0).astype(float).rolling(window, min_periods=window).mean()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result = result.mask(signs > 0, pos_ratio)
    result = result.mask(signs < 0, neg_ratio)
    result = result.mask(signs == 0, 0.0)
    return result


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = _safe_div(gain, loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    direction = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window, min_periods=window).sum()
    return _safe_div(direction, volatility)


def _rolling_rank_pct(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        last = values[-1]
        if np.isnan(last):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        less = np.sum(valid < last)
        equal = np.sum(valid == last)
        average_rank = less + ((equal + 1.0) / 2.0)
        return float(average_rank / len(valid))

    return series.rolling(window, min_periods=window).apply(rank_last, raw=True)
