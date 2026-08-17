"""Analyze baseline MFE Q20 dynamic-TP trades by fixed Vietnam entry hour."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from crypto.data import load_ohlcv
from temp.analyze_mfe_q20_prediction_bands import _trade_frame


DEFAULT_CACHE = Path("temp/model/meta_mfe_q20_after_1m_2m_3m/common_meta_oof.pkl")
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUTPUT = Path("temp/output/mfe_q20_dynamic_tp_by_entry_hour.csv")
DEFAULT_DAILY_OUTPUT = Path(
    "temp/output/mfe_q20_dynamic_tp_daily_by_entry_hour.csv"
)


def _hourly(split: str, trades: pd.DataFrame, trade_cost: float) -> pd.DataFrame:
    data = trades.copy()
    data["entry_time"] = pd.DatetimeIndex(data.index) + pd.Timedelta(minutes=5)
    # craw_btc.py stores Asia/Ho_Chi_Minh candle times as timezone-naive CSV
    # values, so the hour read from the index is already Vietnam local time.
    data["entry_hour_vn"] = data["entry_time"].dt.hour
    data["net_return"] = data["gross_return"] - float(trade_cost)
    rows = []
    for hour in range(24):
        selected = data.loc[data["entry_hour_vn"].eq(hour)]
        rows.append(
            {
                "split": split,
                "entry_hour_vn": hour,
                "entry_hour_utc": (hour - 7) % 24,
                "n": len(selected),
                "active_days": selected["entry_time"].dt.normalize().nunique(),
                "trades_per_active_day": (
                    len(selected) / selected["entry_time"].dt.normalize().nunique()
                    if len(selected)
                    else 0.0
                ),
                "tp_hit_rate": selected["tp_hit"].mean(),
                "gross_mean": selected["gross_return"].mean(),
                "net_mean": selected["net_return"].mean(),
                "net_win_rate": selected["net_return"].gt(0.0).mean(),
                "net_median": selected["net_return"].median(),
                "net_std": selected["net_return"].std(ddof=1),
                "net_sum": selected["net_return"].sum(),
            }
        )
    return pd.DataFrame(rows)


def _daily_hourly(
    split: str,
    trades: pd.DataFrame,
    full_index: pd.Index,
    trade_cost: float,
) -> pd.DataFrame:
    data = trades.copy()
    data["entry_time"] = pd.DatetimeIndex(data.index) + pd.Timedelta(minutes=5)
    data["entry_date"] = data["entry_time"].dt.normalize()
    data["entry_hour_vn"] = data["entry_time"].dt.hour
    data["net_return"] = data["gross_return"] - float(trade_cost)

    full_entry_dates = (
        pd.DatetimeIndex(full_index) + pd.Timedelta(minutes=5)
    ).normalize().unique()
    calendar = pd.DatetimeIndex(full_entry_dates).sort_values()
    rows = []
    for hour in range(24):
        selected = data.loc[data["entry_hour_vn"].eq(hour)]
        daily = selected.groupby("entry_date")["net_return"].sum().reindex(
            calendar, fill_value=0.0
        )
        active = daily.ne(0.0)
        # A zero sum can theoretically contain offsetting trades, so use the
        # actual dates with at least one selected trade for the active count.
        active_dates = pd.Index(selected["entry_date"].unique())
        active = pd.Series(daily.index.isin(active_dates), index=daily.index)
        positive = daily.gt(0.0)
        negative = daily.lt(0.0)
        rows.append(
            {
                "split": split,
                "entry_hour_vn": hour,
                "entry_hour_utc": (hour - 7) % 24,
                "total_days": len(daily),
                "active_days": int(active.sum()),
                "no_trade_days": int((~active).sum()),
                "positive_days": int(positive.sum()),
                "negative_days": int(negative.sum()),
                "flat_active_days": int((active & daily.eq(0.0)).sum()),
                "positive_day_rate_all": positive.mean(),
                "positive_day_rate_active": (
                    positive.sum() / active.sum() if active.any() else 0.0
                ),
                "mean_daily_net_all": daily.mean(),
                "mean_daily_net_active": daily.loc[active].mean() if active.any() else 0.0,
                "median_daily_net_active": (
                    daily.loc[active].median() if active.any() else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--trade-cost", type=float, default=0.0002)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--daily-out", type=Path, default=DEFAULT_DAILY_OUTPUT)
    args = parser.parse_args()

    cached = pd.read_pickle(args.cache)
    raw = load_ohlcv(args.data)
    required = ["meta_dynamic_tp_h3"]
    val = _trade_frame(cached.val_df.dropna(subset=required), raw, args.horizon)
    test = _trade_frame(cached.test_df.dropna(subset=required), raw, args.horizon)
    result = pd.concat(
        [
            _hourly("val", val, args.trade_cost),
            _hourly("test", test, args.trade_cost),
        ],
        ignore_index=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    daily_result = pd.concat(
        [
            _daily_hourly(
                "val", val, cached.val_df.index, args.trade_cost
            ),
            _daily_hourly(
                "test", test, cached.test_df.index, args.trade_cost
            ),
        ],
        ignore_index=True,
    )
    daily_result.to_csv(args.daily_out, index=False)

    display = result.copy()
    for column in (
        "tp_hit_rate", "gross_mean", "net_mean", "net_win_rate",
        "net_median", "net_std", "net_sum",
    ):
        display[column] = display[column].map(lambda value: f"{100.0 * value:+.4f}%")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(display.to_string(index=False))
    print(f"\nSaved: {args.out}")
    daily_display = daily_result.copy()
    for column in (
        "positive_day_rate_all", "positive_day_rate_active",
        "mean_daily_net_all", "mean_daily_net_active",
        "median_daily_net_active",
    ):
        daily_display[column] = daily_display[column].map(
            lambda value: f"{100.0 * value:+.4f}%"
        )
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\nDaily result by fixed Vietnam entry hour:")
        print(daily_display.to_string(index=False))
    print(f"\nSaved: {args.daily_out}")


if __name__ == "__main__":
    main()
