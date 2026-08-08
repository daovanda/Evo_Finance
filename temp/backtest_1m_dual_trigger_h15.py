"""Backtest trigger-based Long and Short entries on every 1m candle.

For every contiguous 15-candle path:

1. Require causal ``ATR14[t-1] / close[t-1] >= MIN_ATR_PCT``.
2. Minute 1 open is the common reference.
3. Long enters at ``open * (1 + trigger)`` if minute-1 high touches it.
4. Short enters at ``open * (1 - trigger)`` if minute-1 low touches it.
5. Minute 1 is trigger-only. From minutes 2 through 15, each active leg:
   - exits at the minute open if that open has already crossed its SL;
   - otherwise checks TP/SL using that minute's high/low;
   - exits at minute-15 close if neither barrier is reached.
6. Long and Short are independent and may both enter on the same candle.

Transaction cost is charged once per entered leg.

PowerShell:
    python -m temp.backtest_1m_dual_trigger_h15 `
      --data data/crypto/BTCUSDT_1m.csv `
      --trigger 0.00025 `
      --atr-lookback 14 `
      --min-atr-pct 0.0004 `
      --take-profit 0.01 `
      --stop-loss 0.00025 `
      --trade-cost 0.00016 `
      --same-candle-policy tp_first `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config


DEFAULT_DATA = Path("data/crypto/BTCUSDT_1m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_TRIGGER = 0.00025
ATR_LOOKBACK = 14
MIN_ATR_PCT = 0.0004
DEFAULT_TAKE_PROFIT = 0.01
DEFAULT_STOP_LOSS = 0.00025
DEFAULT_TRADE_COST = 0.00016
DEFAULT_HOLD_MINUTES = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("temp.backtest_1m_dual_trigger_h15")


def load_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.loc[
        (frame[["open", "high", "low", "close"]] > 0.0).all(axis=1)
    ]
    return frame.reset_index(drop=True)


def causal_atr_percent(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    """Return ATR% known before each candidate candle opens."""
    period = int(lookback)
    if period < 1:
        raise ValueError("atr_lookback must be at least 1.")
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    return (atr / close).shift(1).to_numpy(dtype=float)


def simulate_dual_trigger(
    frame: pd.DataFrame,
    *,
    trigger: float,
    atr_lookback: int,
    min_atr_pct: float,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    hold_minutes: int,
    same_candle_policy: str,
) -> pd.DataFrame:
    trigger = float(trigger)
    tp = float(take_profit)
    sl = float(stop_loss)
    cost = float(trade_cost)
    horizon = int(hold_minutes)
    policy = str(same_candle_policy).strip().lower()
    if trigger <= 0.0 or tp <= 0.0 or sl <= 0.0:
        raise ValueError("trigger, take_profit, and stop_loss must be positive.")
    if float(min_atr_pct) < 0.0:
        raise ValueError("min_atr_pct must be non-negative.")
    if cost < 0.0:
        raise ValueError("trade_cost must be non-negative.")
    if horizon < 2:
        raise ValueError("hold_minutes must be at least 2.")
    if policy not in {"stop_first", "tp_first"}:
        raise ValueError("same_candle_policy must be stop_first or tp_first.")
    if len(frame) < horizon:
        return pd.DataFrame()

    dates = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]")
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    count = len(frame) - horizon + 1
    base_open = opens[:count]
    atr_pct = causal_atr_percent(frame, int(atr_lookback))[:count]

    expected_span = np.timedelta64(horizon - 1, "m")
    contiguous = (dates[horizon - 1 :] - dates[:count]) == expected_span
    atr_allowed = np.isfinite(atr_pct) & (atr_pct >= float(min_atr_pct))
    long_entry = base_open * (1.0 + trigger)
    short_entry = base_open * (1.0 - trigger)
    long_entered = contiguous & atr_allowed & (highs[:count] >= long_entry)
    short_entered = contiguous & atr_allowed & (lows[:count] <= short_entry)

    long_tp = long_entry * (1.0 + tp)
    long_sl = long_entry * (1.0 - sl)
    short_tp = short_entry * (1.0 - tp)
    short_sl = short_entry * (1.0 + sl)

    long_active = long_entered.copy()
    short_active = short_entered.copy()
    long_return = np.full(count, np.nan, dtype=float)
    short_return = np.full(count, np.nan, dtype=float)
    long_exit_minute = np.full(count, -1, dtype=np.int16)
    short_exit_minute = np.full(count, -1, dtype=np.int16)
    long_outcome = np.full(count, "not_entered", dtype=object)
    short_outcome = np.full(count, "not_entered", dtype=object)
    long_outcome[long_entered] = "close_m15"
    short_outcome[short_entered] = "close_m15"

    for offset in range(1, horizon):
        minute_number = offset + 1
        minute_open = opens[offset : offset + count]
        minute_high = highs[offset : offset + count]
        minute_low = lows[offset : offset + count]

        long_gap = long_active & (minute_open <= long_sl)
        long_return[long_gap] = minute_open[long_gap] / long_entry[long_gap] - 1.0
        long_exit_minute[long_gap] = minute_number
        long_outcome[long_gap] = f"sl_open_m{minute_number}"
        long_active[long_gap] = False

        short_gap = short_active & (minute_open >= short_sl)
        short_return[short_gap] = 1.0 - minute_open[short_gap] / short_entry[short_gap]
        short_exit_minute[short_gap] = minute_number
        short_outcome[short_gap] = f"sl_open_m{minute_number}"
        short_active[short_gap] = False

        long_tp_hit = minute_high >= long_tp
        long_sl_hit = minute_low <= long_sl
        short_tp_hit = minute_low <= short_tp
        short_sl_hit = minute_high >= short_sl
        if policy == "tp_first":
            long_tp_exit = long_active & long_tp_hit
            long_sl_exit = long_active & ~long_tp_hit & long_sl_hit
            short_tp_exit = short_active & short_tp_hit
            short_sl_exit = short_active & ~short_tp_hit & short_sl_hit
        else:
            long_sl_exit = long_active & long_sl_hit
            long_tp_exit = long_active & ~long_sl_hit & long_tp_hit
            short_sl_exit = short_active & short_sl_hit
            short_tp_exit = short_active & ~short_sl_hit & short_tp_hit

        long_return[long_tp_exit] = tp
        long_exit_minute[long_tp_exit] = minute_number
        long_outcome[long_tp_exit] = f"tp_m{minute_number}"
        long_active[long_tp_exit] = False
        long_return[long_sl_exit] = -sl
        long_exit_minute[long_sl_exit] = minute_number
        long_outcome[long_sl_exit] = f"sl_m{minute_number}"
        long_active[long_sl_exit] = False

        short_return[short_tp_exit] = tp
        short_exit_minute[short_tp_exit] = minute_number
        short_outcome[short_tp_exit] = f"tp_m{minute_number}"
        short_active[short_tp_exit] = False
        short_return[short_sl_exit] = -sl
        short_exit_minute[short_sl_exit] = minute_number
        short_outcome[short_sl_exit] = f"sl_m{minute_number}"
        short_active[short_sl_exit] = False

    final_close = closes[horizon - 1 : horizon - 1 + count]
    long_return[long_active] = final_close[long_active] / long_entry[long_active] - 1.0
    long_exit_minute[long_active] = horizon
    short_return[short_active] = 1.0 - final_close[short_active] / short_entry[short_active]
    short_exit_minute[short_active] = horizon

    leg_count = long_entered.astype(np.int8) + short_entered.astype(np.int8)
    any_entry = leg_count > 0
    gross = np.nan_to_num(long_return, nan=0.0) + np.nan_to_num(
        short_return, nan=0.0
    )
    net = gross - cost * leg_count
    result = pd.DataFrame(
        {
            "trigger_time": pd.to_datetime(dates[:count]),
            "open_reference": base_open,
            "atr_pct": atr_pct,
            "long_entered": long_entered,
            "short_entered": short_entered,
            "leg_count": leg_count,
            "long_entry": np.where(long_entered, long_entry, np.nan),
            "short_entry": np.where(short_entered, short_entry, np.nan),
            "long_outcome": long_outcome,
            "short_outcome": short_outcome,
            "long_exit_minute": long_exit_minute,
            "short_exit_minute": short_exit_minute,
            "long_return": long_return,
            "short_return": short_return,
            "gross_return": gross,
            "net_return": net,
        }
    )
    return result.loc[any_entry].reset_index(drop=True)


def summarize(name: str, trades: pd.DataFrame) -> dict[str, object]:
    n = len(trades)
    long_mask = trades["long_entered"].astype(bool) if n else pd.Series(dtype=bool)
    short_mask = trades["short_entered"].astype(bool) if n else pd.Series(dtype=bool)
    long_outcome = trades.loc[long_mask, "long_outcome"].astype(str)
    short_outcome = trades.loc[short_mask, "short_outcome"].astype(str)
    if n:
        elapsed_days = max(
            (trades["trigger_time"].max() - trades["trigger_time"].min()).total_seconds()
            / 86400.0,
            1.0,
        )
    else:
        elapsed_days = 1.0
    return {
        "split": name,
        "signal candles": n,
        "legs": int(trades["leg_count"].sum()) if n else 0,
        "long entries": int(long_mask.sum()) if n else 0,
        "short entries": int(short_mask.sum()) if n else 0,
        "both entries": int((long_mask & short_mask).sum()) if n else 0,
        "long TP": float(long_outcome.str.startswith("tp_").mean()) if len(long_outcome) else np.nan,
        "long SL": float(long_outcome.str.startswith("sl_").mean()) if len(long_outcome) else np.nan,
        "short TP": float(short_outcome.str.startswith("tp_").mean()) if len(short_outcome) else np.nan,
        "short SL": float(short_outcome.str.startswith("sl_").mean()) if len(short_outcome) else np.nan,
        "ATR% mean": float(trades["atr_pct"].mean()) if n else np.nan,
        "gross mean": float(trades["gross_return"].mean()) if n else np.nan,
        "E[net]": float(trades["net_return"].mean()) if n else np.nan,
        "win rate": float((trades["net_return"] > 0.0).mean()) if n else np.nan,
        "legs/day": float(trades["leg_count"].sum() / elapsed_days) if n else 0.0,
    }


def summarize_splits(trades: pd.DataFrame) -> pd.DataFrame:
    time = pd.to_datetime(trades["trigger_time"])
    val_start = pd.Timestamp(config.VAL_START)
    test_start = pd.Timestamp(config.TEST_START)
    test_end = pd.Timestamp(config.TEST_END) if config.TEST_END is not None else None
    masks = {
        "all": np.ones(len(trades), dtype=bool),
        "val": (time >= val_start) & (time < test_start),
        "test": (time >= test_start)
        & ((time <= test_end) if test_end is not None else np.ones(len(trades), dtype=bool)),
    }
    return pd.DataFrame(
        [summarize(name, trades.loc[np.asarray(mask)]) for name, mask in masks.items()]
    )


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for column in ("signal candles", "legs", "long entries", "short entries", "both entries"):
        result[column] = result[column].map(lambda value: f"{int(value):,}")
    for column in (
        "long TP",
        "long SL",
        "short TP",
        "short SL",
        "ATR% mean",
        "win rate",
    ):
        result[column] = result[column].map(lambda value: f"{value:.2%}")
    for column in ("gross mean", "E[net]"):
        result[column] = result[column].map(lambda value: f"{value:+.4%}")
    result["legs/day"] = result["legs/day"].map(lambda value: f"{value:.1f}")
    return result


def draw_report(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    trigger: float,
    atr_lookback: int,
    min_atr_pct: float,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    hold_minutes: int,
    same_candle_policy: str,
    output: Path,
) -> None:
    ordered = trades.sort_values("trigger_time").copy()
    ordered["cumulative_net"] = ordered["net_return"].cumsum()
    formatted = format_summary(summary)
    fig, (table_ax, equity_ax) = plt.subplots(
        2,
        1,
        figsize=(21, 9),
        gridspec_kw={"height_ratios": [1.2, 3.5], "hspace": 0.32},
    )
    fig.patch.set_facecolor("#f4f6f8")
    table_ax.axis("off")
    table = table_ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#9aa3ad")
        cell.set_facecolor("#222831" if row == 0 else "#ffffff")
        cell.set_text_props(color="#ffffff" if row == 0 else "#111111")
    table_ax.set_title(
        "Every-1m dual trigger strategy | "
        f"ATR{atr_lookback}[t-1] >= {min_atr_pct:.3%} | "
        f"trigger=±{trigger:.3%} TP={take_profit:.3%} SL={stop_loss:.3%} | "
        f"close M{hold_minutes} | cost={trade_cost:.3%}/leg | "
        f"same candle={same_candle_policy}",
        fontsize=11,
        pad=14,
    )

    equity_ax.plot(
        pd.to_datetime(ordered["trigger_time"]),
        ordered["cumulative_net"],
        color="#1769aa",
        linewidth=0.9,
    )
    equity_ax.axhline(0.0, color="#333333", linewidth=0.8)
    equity_ax.grid(True, alpha=0.3)
    equity_ax.set_xlabel("Trigger candle time")
    equity_ax.set_ylabel("Cumulative arithmetic net return")
    equity_ax.set_title("Overlapping Long/Short positions allowed")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trigger", type=float, default=DEFAULT_TRIGGER)
    parser.add_argument("--atr-lookback", type=int, default=ATR_LOOKBACK)
    parser.add_argument("--min-atr-pct", type=float, default=MIN_ATR_PCT)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS)
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
    parser.add_argument("--hold-minutes", type=int, default=DEFAULT_HOLD_MINUTES)
    parser.add_argument(
        "--same-candle-policy",
        choices=("stop_first", "tp_first"),
        default="stop_first",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading %s", args.data)
    frame = load_prices(args.data)
    logger.info("Loaded %s one-minute candles.", f"{len(frame):,}")
    trades = simulate_dual_trigger(
        frame,
        trigger=float(args.trigger),
        atr_lookback=int(args.atr_lookback),
        min_atr_pct=float(args.min_atr_pct),
        take_profit=float(args.take_profit),
        stop_loss=float(args.stop_loss),
        trade_cost=float(args.trade_cost),
        hold_minutes=int(args.hold_minutes),
        same_candle_policy=str(args.same_candle_policy),
    )
    summary = summarize_splits(trades)
    output = args.out_dir / (
        "BTCUSDT_1m_every_candle_dual_trigger_"
        f"atr{args.atr_lookback}min{args.min_atr_pct * 100:.3f}pct_"
        f"f{args.trigger * 100:.3f}pct_"
        f"tp{args.take_profit * 100:.3f}pct_"
        f"sl{args.stop_loss * 100:.3f}pct_"
        f"m{args.hold_minutes}_{args.same_candle_policy}.png"
    )
    draw_report(
        trades,
        summary,
        trigger=float(args.trigger),
        atr_lookback=int(args.atr_lookback),
        min_atr_pct=float(args.min_atr_pct),
        take_profit=float(args.take_profit),
        stop_loss=float(args.stop_loss),
        trade_cost=float(args.trade_cost),
        hold_minutes=int(args.hold_minutes),
        same_candle_policy=str(args.same_candle_policy),
        output=output,
    )
    print("\n=== Every-1m dual trigger strategy ===")
    print(format_summary(summary).to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
