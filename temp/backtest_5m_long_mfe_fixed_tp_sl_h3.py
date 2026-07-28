"""Backtest a 5-minute Long MFE signal with fixed TP/SL through H3.

Flow:
1. Train one selected Long MFE archive rank.
2. Learn its top-fraction cutoff on Final Val and apply it unchanged to Test.
3. Enter Long at open H1 for every selected timestamp.
4. Keep one fixed TP and SL active through H1, H2, and H3.
5. If neither barrier is reached, exit at close H3.

When TP and SL are both reached in the same 5-minute candle, OHLC cannot reveal
their order. The default is conservative ``stop_first``; use
``--same-candle-policy tp_first`` for the optimistic sensitivity case.

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

Oracle look-ahead variant (for diagnostic use only):
    python -m temp.backtest_5m_long_mfe_fixed_tp_sl_h3 `
      --archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --rank 1 `
      --top-fraction 0.40 `
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
import logging
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
    ModelSpec,
    _archive_horizons,
    _cached_feature_space,
    _load_rank_entry,
    _quality_train_index,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv
from temp.backtest_5m_long_mfe_h3 import (
    load_archive_metadata,
    load_one_minute_ohlc,
    make_price_path,
)


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
DEFAULT_RANK = 1
DEFAULT_TOP_FRACTION = 0.40
DEFAULT_TAKE_PROFIT = 0.001
DEFAULT_STOP_LOSS = 0.001
DEFAULT_TRADE_COST = 0.00016


def simulate_fixed_tp_sl(
    selected_path: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
) -> pd.DataFrame:
    """Simulate fixed barriers in chronological H1-H3 order."""
    required = [
        "high_h1",
        "low_h1",
        "high_h2",
        "low_h2",
        "high_h3",
        "low_h3",
        "close_h3",
    ]
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

    for step in range(1, 4):
        high = pd.to_numeric(
            result[f"high_h{step}"], errors="coerce"
        ).to_numpy()
        low = pd.to_numeric(
            result[f"low_h{step}"], errors="coerce"
        ).to_numpy()
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
    take_profit: float,
    stop_loss: float,
    trade_cost: float,
    same_candle_policy: str,
) -> pd.DataFrame:
    """Filter on H1 minute 1, then trade from H1 minutes 2-5 onward.

    This deliberately uses future information and is therefore an oracle
    diagnostic, not a causal strategy suitable for live trading.
    """
    tp = float(take_profit)
    sl = float(stop_loss)
    policy = str(same_candle_policy).strip().lower()
    if policy not in {"stop_first", "tp_first"}:
        raise ValueError("same_candle_policy must be stop_first or tp_first.")

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

    high_return = high_values / entry_open[:, None] - 1.0
    low_return = low_values / entry_open[:, None] - 1.0
    keep = high_return[:, 0] >= tp
    result = selected_path.iloc[np.flatnonzero(keep)].copy()
    kept_high = high_return[keep]
    kept_low = low_return[keep]

    # Minute 1 is filter-only. H1 execution starts at minutes 2 through 5.
    result["high_h1"] = np.max(kept_high[:, 1:5], axis=1)
    result["low_h1"] = np.min(kept_low[:, 1:5], axis=1)
    n = len(result)
    active = np.ones(n, dtype=bool)
    gross = np.full(n, np.nan, dtype=float)
    exit_h = np.full(n, 3, dtype=int)
    outcome = np.full(n, "close_h3", dtype=object)

    for minute_offset in range(1, 5):
        tp_hit = kept_high[:, minute_offset] >= tp
        sl_hit = kept_low[:, minute_offset] <= -sl
        if policy == "stop_first":
            sl_exit = active & sl_hit
            tp_exit = active & ~sl_hit & tp_hit
        else:
            tp_exit = active & tp_hit
            sl_exit = active & ~tp_hit & sl_hit

        gross[sl_exit] = -sl
        outcome[sl_exit] = "sl_h1"
        exit_h[sl_exit] = 1
        active[sl_exit] = False

        gross[tp_exit] = tp
        outcome[tp_exit] = "tp_h1"
        exit_h[tp_exit] = 1
        active[tp_exit] = False

    for step in (2, 3):
        high = pd.to_numeric(
            result[f"high_h{step}"], errors="coerce"
        ).to_numpy(dtype=float)
        low = pd.to_numeric(
            result[f"low_h{step}"], errors="coerce"
        ).to_numpy(dtype=float)
        tp_hit = high >= tp
        sl_hit = low <= -sl
        if policy == "stop_first":
            sl_exit = active & sl_hit
            tp_exit = active & ~sl_hit & tp_hit
        else:
            tp_exit = active & tp_hit
            sl_exit = active & ~tp_hit & sl_hit

        gross[sl_exit] = -sl
        outcome[sl_exit] = f"sl_h{step}"
        exit_h[sl_exit] = step
        active[sl_exit] = False

        gross[tp_exit] = tp
        outcome[tp_exit] = f"tp_h{step}"
        exit_h[tp_exit] = step
        active[tp_exit] = False

    close_h3 = pd.to_numeric(
        result["close_h3"], errors="coerce"
    ).to_numpy(dtype=float)
    gross[active] = close_h3[active]
    result["outcome"] = outcome
    result["exit_h"] = exit_h
    result["gross_return"] = gross
    result["net_return"] = gross - float(trade_cost)
    return result


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


def summarize_h1_both(
    split: str,
    simulation: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
) -> dict[str, Any]:
    high_h1 = pd.to_numeric(simulation["high_h1"], errors="coerce")
    low_h1 = pd.to_numeric(simulation["low_h1"], errors="coerce")
    close_h1 = pd.to_numeric(simulation["close_h1"], errors="coerce")
    close_h2 = pd.to_numeric(simulation["close_h2"], errors="coerce")
    close_h3 = pd.to_numeric(simulation["close_h3"], errors="coerce")
    low_h2 = pd.to_numeric(simulation["low_h2"], errors="coerce")
    low_h3 = pd.to_numeric(simulation["low_h3"], errors="coerce")
    high_h2 = pd.to_numeric(simulation["high_h2"], errors="coerce")
    high_h3 = pd.to_numeric(simulation["high_h3"], errors="coerce")
    tp_hit = high_h1.ge(float(take_profit))
    sl_hit = low_h1.le(-float(stop_loss))
    both = tp_hit & sl_hit
    tp_only = tp_hit & ~sl_hit
    sl_only = sl_hit & ~tp_hit
    one_side = tp_only | sl_only
    neither = ~tp_hit & ~sl_hit
    both_count = int(both.sum())
    tp_only_count = int(tp_only.sum())
    sl_only_count = int(sl_only.sum())
    one_side_count = int(one_side.sum())
    neither_count = int(neither.sum())
    tp_only_h2_low = tp_only & low_h2.lt(float(take_profit))
    sl_only_h2_high = sl_only & high_h2.gt(-float(stop_loss))
    tp_only_h2_low_h3_high = (
        tp_only_h2_low & high_h3.gt(float(take_profit))
    )
    sl_only_h2_high_h3_low = (
        sl_only_h2_high & low_h3.lt(-float(stop_loss))
    )
    tp_only_h2_low_count = int(tp_only_h2_low.sum())
    sl_only_h2_high_count = int(sl_only_h2_high.sum())
    tp_only_h2_low_h3_high_count = int(tp_only_h2_low_h3_high.sum())
    sl_only_h2_high_h3_low_count = int(sl_only_h2_high_h3_low.sum())
    close_above_tp = both & close_h1.ge(float(take_profit))
    close_below_sl = both & close_h1.le(-float(stop_loss))
    close_between = both & ~(close_above_tp | close_below_sl)

    def distribution(
        series: pd.Series,
        mask: pd.Series,
        prefix: str,
        quantile_name: str,
        quantile: float,
    ) -> dict[str, float]:
        values = pd.to_numeric(series.loc[mask], errors="coerce").dropna()
        if values.empty:
            return {
                f"{prefix}_mean": np.nan,
                f"{prefix}_min": np.nan,
                f"{prefix}_{quantile_name}": np.nan,
                f"{prefix}_max": np.nan,
            }
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_{quantile_name}": float(values.quantile(quantile)),
            f"{prefix}_max": float(values.max()),
        }

    path_distribution: dict[str, float] = {}
    for group_prefix, group_mask in (
        ("tp_only", tp_only),
        ("sl_only", sl_only),
    ):
        for horizon, low_series, high_series in (
            (2, low_h2, high_h2),
            (3, low_h3, high_h3),
        ):
            path_distribution.update(
                distribution(
                    low_series,
                    group_mask,
                    f"{group_prefix}_low_h{horizon}",
                    "q70",
                    0.70,
                )
            )
            path_distribution.update(
                distribution(
                    high_series,
                    group_mask,
                    f"{group_prefix}_high_h{horizon}",
                    "q30",
                    0.30,
                )
            )
    return {
        "split": split,
        "signals": len(simulation),
        "both_count": both_count,
        "both_rate": (
            float(both_count / len(simulation)) if len(simulation) else 0.0
        ),
        "close_above_tp_count": int(close_above_tp.sum()),
        "close_above_tp_rate": (
            float(close_above_tp.sum() / both_count) if both_count else 0.0
        ),
        "close_between_count": int(close_between.sum()),
        "close_between_rate": (
            float(close_between.sum() / both_count) if both_count else 0.0
        ),
        "close_below_sl_count": int(close_below_sl.sum()),
        "close_below_sl_rate": (
            float(close_below_sl.sum() / both_count) if both_count else 0.0
        ),
        "both_close_h1_mean": (
            float(close_h1.loc[both].mean()) if both_count else np.nan
        ),
        "one_side_count": one_side_count,
        "one_side_rate": (
            float(one_side_count / len(simulation))
            if len(simulation)
            else 0.0
        ),
        "one_side_close_h3_mean": (
            float(close_h3.loc[one_side].mean())
            if one_side_count
            else np.nan
        ),
        "tp_only_count": tp_only_count,
        "tp_only_rate": (
            float(tp_only_count / len(simulation))
            if len(simulation)
            else 0.0
        ),
        "tp_only_h2_low_h3_high_count": tp_only_h2_low_h3_high_count,
        "tp_only_h2_low_count": tp_only_h2_low_count,
        "tp_only_h2_low_h3_high_rate": (
            float(tp_only_h2_low_h3_high_count / tp_only_h2_low_count)
            if tp_only_h2_low_count
            else 0.0
        ),
        "tp_only_close_h3_mean": (
            float(close_h3.loc[tp_only].mean())
            if tp_only_count
            else np.nan
        ),
        "tp_only_close_h1_mean": (
            float(close_h1.loc[tp_only].mean())
            if tp_only_count
            else np.nan
        ),
        "tp_only_close_h2_mean": (
            float(close_h2.loc[tp_only].mean())
            if tp_only_count
            else np.nan
        ),
        "sl_only_count": sl_only_count,
        "sl_only_rate": (
            float(sl_only_count / len(simulation))
            if len(simulation)
            else 0.0
        ),
        "sl_only_h2_high_h3_low_count": sl_only_h2_high_h3_low_count,
        "sl_only_h2_high_count": sl_only_h2_high_count,
        "sl_only_h2_high_h3_low_rate": (
            float(sl_only_h2_high_h3_low_count / sl_only_h2_high_count)
            if sl_only_h2_high_count
            else 0.0
        ),
        "sl_only_close_h3_mean": (
            float(close_h3.loc[sl_only].mean())
            if sl_only_count
            else np.nan
        ),
        "sl_only_close_h1_mean": (
            float(close_h1.loc[sl_only].mean())
            if sl_only_count
            else np.nan
        ),
        "sl_only_close_h2_mean": (
            float(close_h2.loc[sl_only].mean())
            if sl_only_count
            else np.nan
        ),
        **path_distribution,
        "neither_count": neither_count,
        "neither_rate": (
            float(neither_count / len(simulation))
            if len(simulation)
            else 0.0
        ),
        "neither_close_h3_mean": (
            float(close_h3.loc[neither].mean())
            if neither_count
            else np.nan
        ),
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


def both_table(h1_both: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in h1_both.iterrows():
        rows.append(
            {
                "split": row["split"],
                "H1 TP+SL / signals": (
                    f"{int(row['both_count']):,}/{int(row['signals']):,} "
                    f"({float(row['both_rate']):.2%})"
                ),
                "close H1 >= TP | both": (
                    f"{int(row['close_above_tp_count']):,} "
                    f"({float(row['close_above_tp_rate']):.2%})"
                ),
                "SL < close H1 < TP | both": (
                    f"{int(row['close_between_count']):,} "
                    f"({float(row['close_between_rate']):.2%})"
                ),
                "close H1 <= SL | both": (
                    f"{int(row['close_below_sl_count']):,} "
                    f"({float(row['close_below_sl_rate']):.2%})"
                ),
                "both mean close H1": (
                    f"{float(row['both_close_h1_mean']):+.3%}"
                ),
                "H1 only TP or SL -> close H3": (
                    f"{int(row['one_side_count']):,} "
                    f"({float(row['one_side_rate']):.2%}) "
                    f"mean={float(row['one_side_close_h3_mean']):+.3%}"
                ),
                "H1 neither TP nor SL -> close H3": (
                    f"{int(row['neither_count']):,} "
                    f"({float(row['neither_rate']):.2%}) "
                    f"mean={float(row['neither_close_h3_mean']):+.3%}"
                ),
            }
        )
    return pd.DataFrame(rows)


def one_side_detail_table(h1_both: pd.DataFrame) -> pd.DataFrame:
    """Show close-path means separately for H1 TP-only and SL-only groups."""
    def distribution_text(
        row: pd.Series,
        metric_prefix: str,
        quantile_name: str,
    ) -> str:
        keys = ("mean", "min", quantile_name, "max")
        values = [row[f"{metric_prefix}_{key}"] for key in keys]
        if any(pd.isna(value) for value in values):
            return "n/a"
        return "|".join(f"{float(value):+.3%}" for value in values)

    rows: list[dict[str, str]] = []
    for _, row in h1_both.iterrows():
        for group, prefix in (("TP-only", "tp_only"), ("SL-only", "sl_only")):
            rows.append(
                {
                    "split / H1 group": f"{row['split']} {group}",
                    "n / signals": (
                        f"{int(row[f'{prefix}_count']):,}/"
                        f"{int(row['signals']):,} "
                        f"({float(row[f'{prefix}_rate']):.2%})"
                    ),
                    "mean close H1": (
                        f"{float(row[f'{prefix}_close_h1_mean']):+.3%}"
                    ),
                    "mean close H2": (
                        f"{float(row[f'{prefix}_close_h2_mean']):+.3%}"
                    ),
                    "mean close H3": (
                        f"{float(row[f'{prefix}_close_h3_mean']):+.3%}"
                    ),
                    "low H2 mean|min|Q70|max": (
                        distribution_text(
                            row,
                            f"{prefix}_low_h2",
                            "q70",
                        )
                    ),
                    "low H3 mean|min|Q70|max": (
                        distribution_text(
                            row,
                            f"{prefix}_low_h3",
                            "q70",
                        )
                    ),
                    "high H2 mean|min|Q30|max": (
                        distribution_text(
                            row,
                            f"{prefix}_high_h2",
                            "q30",
                        )
                    ),
                    "high H3 mean|min|Q30|max": (
                        distribution_text(
                            row,
                            f"{prefix}_high_h3",
                            "q30",
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def one_side_follow_through_table(h1_both: pd.DataFrame) -> pd.DataFrame:
    """Measure H2-H3 follow-through within each H1 one-sided cohort."""
    rows: list[dict[str, str]] = []
    for _, row in h1_both.iterrows():
        rows.append(
            {
                "split": row["split"],
                "H1 TP-only AND low H2 < TP n": (
                    f"{int(row['tp_only_h2_low_count']):,}"
                ),
                (
                    "P(low H2 < TP AND high H3 > TP | "
                    "H1 TP-only AND low H2 < TP)"
                ): (
                    f"{int(row['tp_only_h2_low_h3_high_count']):,}/"
                    f"{int(row['tp_only_h2_low_count']):,} "
                    f"({float(row['tp_only_h2_low_h3_high_rate']):.2%})"
                ),
                "H1 SL-only AND high H2 > -SL n": (
                    f"{int(row['sl_only_h2_high_count']):,}"
                ),
                (
                    "P(high H2 > -SL AND low H3 < -SL | "
                    "H1 SL-only AND high H2 > -SL)"
                ): (
                    f"{int(row['sl_only_h2_high_h3_low_count']):,}/"
                    f"{int(row['sl_only_h2_high_count']):,} "
                    f"({float(row['sl_only_h2_high_h3_low_rate']):.2%})"
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_report(
    summary: pd.DataFrame,
    h1_both: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(27, 19),
        gridspec_kw={
            "height_ratios": [1.0, 1.0, 1.0, 0.9, 3.0, 1.7]
        },
        constrained_layout=True,
    )
    for axis, frame, table_title in (
        (axes[0], main_table(summary), "Fixed TP/SL strategy"),
        (
            axes[1],
            both_table(h1_both),
            "H1 touched both TP and SL",
        ),
        (
            axes[2],
            one_side_detail_table(h1_both),
            "H1 one-sided touch: counterfactual close path",
        ),
        (
            axes[3],
            one_side_follow_through_table(h1_both),
            "H1 one-sided touch: H2-H3 follow-through",
        ),
    ):
        axis.axis("off")
        table = axis.table(
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
        axis.set_title(table_title, fontsize=11, pad=8)

    colors = {"val": "#2563eb", "test": "#dc2626"}
    for split, frame in simulations.items():
        ordered = frame.sort_index()
        cumulative = ordered["net_return"].cumsum()
        axes[4].plot(
            ordered.index,
            cumulative * 100.0,
            color=colors[split],
            linewidth=1.1,
            label=(
                f"{split.upper()} n={len(ordered):,} | "
                f"end={float(cumulative.iloc[-1]) * 100.0:+.2f}%"
            ),
        )
    axes[4].axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    axes[4].set_title("Fixed TP/SL cumulative net return")
    axes[4].set_ylabel("Percentage points")
    axes[4].grid(True, alpha=0.5)
    axes[4].legend(frameon=False)

    combined = pd.concat(simulations.values()).sort_index()
    daily = pd.Series(1, index=combined.index).resample("D").sum()
    axes[5].bar(daily.index, daily.to_numpy(), width=0.9, color="#f59e0b")
    axes[5].set_title("Signals per day")
    axes[5].set_ylabel("Signals")
    axes[5].grid(True, axis="y", alpha=0.5)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def run(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    archive_path = Path(args.archive)
    data_path = Path(args.data)
    metadata = load_archive_metadata(archive_path)
    mode = config.canonical_label_mode(metadata.get("label_mode"))
    direction = config.canonical_label_direction(
        metadata.get("label_direction")
    )
    horizons = _archive_horizons(
        archive_path,
        fallback=[3],
        label="5m Long MFE fixed TP/SL",
    )
    if mode != "mfe" or direction != "long" or horizons != [3]:
        raise ValueError(
            "Archive must use mode=mfe, direction=long, horizons=[3]; got "
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
    path = make_price_path(raw_df, horizon=3)
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
    both_rows: list[dict[str, Any]] = []
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
                    take_profit=float(args.take_profit),
                    stop_loss=float(args.stop_loss),
                    trade_cost=float(args.trade_cost),
                    same_candle_policy=args.same_candle_policy,
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
        both_rows.append(
            summarize_h1_both(
                split=split,
                simulation=simulation,
                take_profit=float(args.take_profit),
                stop_loss=float(args.stop_loss),
            )
        )

    summary = pd.DataFrame(summary_rows)
    h1_both = pd.DataFrame(both_rows)
    run_name = (
        f"{archive_path.stem}_r{spec.rank:02d}_top"
        f"{spec.top_fraction * 100.0:.0f}_fixed_tp"
        f"{float(args.take_profit) * 100.0:.3f}pct_sl"
        f"{float(args.stop_loss) * 100.0:.3f}pct_"
        f"{args.same_candle_policy}"
        f"{'_next1m_oracle' if args.next_1m_tp_filter else ''}"
    ).replace(".", "p")
    output_path = Path(args.out_dir) / f"{run_name}.png"
    title = (
        "5m Long MFE H3 | fixed TP/SL through H1-H3 | "
        f"rank={spec.rank}, top={spec.top_fraction:.0%}, "
        f"TP=+{float(args.take_profit):.2%}, "
        f"SL=-{float(args.stop_loss):.2%}, "
        f"same-candle={args.same_candle_policy}, "
        f"cost={float(args.trade_cost):.3%}, "
        f"next-1m TP oracle={'ON' if args.next_1m_tp_filter else 'OFF'}"
    )
    draw_report(summary, h1_both, simulations, output_path, title)
    logger.info("Saved report: %s", output_path)
    return summary, h1_both, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_TAKE_PROFIT)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS)
    parser.add_argument("--trade-cost", type=float, default=DEFAULT_TRADE_COST)
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
            "Oracle diagnostic: use H1 minute 1 only to require a TP touch, "
            "then evaluate H1 TP/SL from minutes 2-5 before H2-H3."
        ),
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    summary, h1_both, output_path = run(parse_args())
    print("\n=== Fixed TP/SL strategy ===")
    print(main_table(summary).to_string(index=False))
    print("\n=== H1 touched both TP and SL ===")
    print(both_table(h1_both).to_string(index=False))
    print("\n=== H1 TP-only / SL-only close path ===")
    print(one_side_detail_table(h1_both).to_string(index=False))
    print("\n=== H1 TP-only / SL-only H2-H3 follow-through ===")
    print(one_side_follow_through_table(h1_both).to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
