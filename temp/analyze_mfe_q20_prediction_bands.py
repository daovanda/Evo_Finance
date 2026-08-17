"""Analyze the original MFE Q20 dynamic-TP strategy by prediction band."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.data import load_ohlcv


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT = Path("temp/output/mfe_q20_dynamic_tp_prediction_bands.csv")


def _trade_frame(frame: pd.DataFrame, raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    signal_index = pd.DatetimeIndex(frame.index)
    entry_index = signal_index + pd.Timedelta(minutes=5)
    entry = pd.to_numeric(raw["open"], errors="coerce").reindex(entry_index)

    trades = pd.DataFrame(index=signal_index)
    trades["prediction"] = pd.to_numeric(
        frame["meta_dynamic_tp_h3"], errors="coerce"
    ).to_numpy(float)
    trades["entry"] = entry.to_numpy(float)
    high_returns = []
    for step in range(1, horizon + 1):
        high = pd.to_numeric(raw["high"], errors="coerce").reindex(
            signal_index + pd.Timedelta(minutes=5 * step)
        )
        high_returns.append(high.to_numpy(float) / trades["entry"].to_numpy(float) - 1.0)
    close_h = pd.to_numeric(raw["close"], errors="coerce").reindex(
        signal_index + pd.Timedelta(minutes=5 * horizon)
    )
    trades["actual_mfe"] = np.nanmax(np.column_stack(high_returns), axis=1)
    trades["close_h3_return"] = (
        close_h.to_numpy(float) / trades["entry"].to_numpy(float) - 1.0
    )
    trades = trades.replace([np.inf, -np.inf], np.nan).dropna()
    trades["tp_hit"] = trades["actual_mfe"].ge(trades["prediction"])
    trades["gross_return"] = np.where(
        trades["tp_hit"], trades["prediction"], trades["close_h3_return"]
    )
    return trades


def _summarize(split: str, trades: pd.DataFrame, edges: np.ndarray) -> list[dict]:
    predictions = trades["prediction"].to_numpy(float)
    band_ids = np.searchsorted(edges, predictions, side="right")
    rows = []
    band_count = len(edges) + 1
    for ascending_band in range(band_count - 1, -1, -1):
        rank_band = band_count - 1 - ascending_band
        selected = trades.iloc[np.flatnonzero(band_ids == ascending_band)]
        tp = selected.loc[selected["tp_hit"]]
        close = selected.loc[~selected["tp_hit"]]
        gross = selected["gross_return"]
        rows.append(
            {
                "split": split,
                "prediction_band": f"top {10 * rank_band}-{10 * (rank_band + 1)}%",
                "n": len(selected),
                "share": len(selected) / max(len(trades), 1),
                "prediction_mean": selected["prediction"].mean(),
                "prediction_min": selected["prediction"].min(),
                "prediction_max": selected["prediction"].max(),
                "tp_hits": int(selected["tp_hit"].sum()),
                "tp_hit_rate": selected["tp_hit"].mean(),
                "hit_tp_mean": tp["gross_return"].mean(),
                "close_h3_count": len(close),
                "close_h3_mean": close["close_h3_return"].mean(),
                "gross_win_rate": gross.gt(0.0).mean(),
                "gross_loss_rate": gross.lt(0.0).mean(),
                "gross_mean": gross.mean(),
                "gross_median": gross.median(),
                "gross_sum": gross.sum(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    val = _trade_frame(cached.val_df.dropna(subset=["meta_dynamic_tp_h3"]), raw, args.horizon)
    test = _trade_frame(cached.test_df.dropna(subset=["meta_dynamic_tp_h3"]), raw, args.horizon)

    # Fit prediction-magnitude boundaries on Val only, then freeze for Test.
    edges = np.quantile(val["prediction"].to_numpy(float), np.arange(0.1, 1.0, 0.1))
    rows = _summarize("val", val, edges) + _summarize("test", test, edges)
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)

    display = output.copy()
    pct_columns = [
        "share", "prediction_mean", "prediction_min", "prediction_max",
        "tp_hit_rate", "hit_tp_mean", "close_h3_mean", "gross_win_rate",
        "gross_loss_rate", "gross_mean", "gross_median", "gross_sum",
    ]
    for column in pct_columns:
        display[column] = display[column].map(
            lambda value: "nan" if pd.isna(value) else f"{100.0 * value:+.4f}%"
        )
    with pd.option_context("display.max_columns", None, "display.width", 260):
        print(display.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
