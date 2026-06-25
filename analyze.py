"""
Evo_Finance — Analyze
──────────────────────
Load archive JSON → retrain mỗi individual → vẽ biểu đồ IC + Hit Rate theo ngày
(từ val đến hết test, đường rolling-5 mean) → lưu vào results/chart/

Usage
-----
    python analyze.py --data-dir data/raw --archive results/archive.json

    # Chỉ vẽ top-N individuals
    python analyze.py --data-dir data/raw --archive results/archive.json --top 10
    python analyze.py --data-dir data/raw --archive results/archive_test6_morongdomain.json --top 10    

    # Custom ngày split (phải khớp với lúc train)
    python analyze.py --data-dir data/raw --archive results/archive.json \\
        --val-start 2022-01-01 --test-start 2023-07-01

Output
------
    results/chart/rank_{N:02d}_score_{score:.4f}.png   — 1 file per individual
    results/chart/ic_summary.png                        — tất cả IC rolling trên 1 chart
    results/chart/hitrate_summary.png                   — tất cả hit rate rolling trên 1 chart
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    VAL_START, TEST_START, TEST_END, HIT_RATE_TOP_K,
    WF_END, WF_MIN_TRAIN_MONTHS, WF_VAL_MONTHS,
    WF_STEP_MONTHS, WF_PURGE_DAYS, FITNESS_WEIGHTS,
)
from data.loader       import load_from_dir
from model.data_utils  import (
    label_dataframe,
    make_walk_forward_folds,
    split_labeled_by_dates,
    validate_ohlcv,
)
from model.trainer     import Trainer
from mutator.gene      import Gene, Individual
from fitness.fitness   import _ic_per_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evo_finance.analyze")

ROLLING_WINDOW = 5


# ─── Load archive JSON ────────────────────────────────────────────────────────

def load_archive_json(path: Path) -> list[dict]:
    with open(path) as f:
        entries = json.load(f)
    logger.info("Loaded %d individuals từ %s", len(entries), path)
    return entries


def json_to_individual(entry: dict) -> Individual:
    genes = [Gene(formula=f) for f in entry["genes"]]
    return Individual(
        genes      = genes,
        score      = None,
        metrics    = {},
        generation = entry.get("generation", 0),
    )


# ─── Hit rate per date ───────────────────────────────────────────────────────

def _hitrate_per_date(
    pred:   pd.Series,
    label:  pd.Series,
    top_k:  int = HIT_RATE_TOP_K,
) -> pd.Series:
    """
    Hit rate per date: |top_k pred ∩ top_k label| / top_k
    Returns pd.Series index = date.
    """
    data = pd.DataFrame({"pred": pred, "label": label}).dropna()

    if isinstance(data.index, pd.MultiIndex):
        dates_arr = data.index.get_level_values("date")
    else:
        dates_arr = data.index

    hits = {}
    for date in np.unique(dates_arr):
        mask = dates_arr == date
        grp  = data[mask]
        if len(grp) < top_k:
            continue
        top_pred  = set(grp.nlargest(top_k, "pred").index)
        top_label = set(grp.nlargest(top_k, "label").index)
        hits[date] = len(top_pred & top_label) / top_k

    return pd.Series(hits).sort_index()


# ─── Compute IC + hit rate series ────────────────────────────────────────────

def compute_series(
    individual: Individual,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """
    Retrain trên train_df với early-stopping monitor = val_df (KHÔNG bao giờ
    là test_df — nếu test lọt vào đây, model overfit trực tiếp lên test
    qua early stopping, làm hit rate/IC test bị inflate giả tạo).

    Sau khi train xong, predict riêng trên val_df và test_df rồi nối lại
    để vẽ chart liên tục từ val → test.

    Returns (ic_series, hitrate_series, eval_dates). eval_dates contains every
    trading date in val + test so charts can break lines where a metric has no
    value instead of connecting across gaps.
    """
    trainer = Trainer()
    empty   = pd.Series(dtype=float)
    eval_dates = _date_index_from_frames(val_df, test_df)
    if feature_df is None:
        feature_df = pd.concat([train_df, val_df, test_df]).sort_index()

    try:
        # Early stopping CHỈ thấy val_df — đúng với cách main.py train thật
        booster, _, _, _, _ = trainer.train(
            individual, train_df, val_df, feature_df=feature_df, mode="final"
        )
    except Exception as exc:
        logger.warning("Train failed: %s", exc)
        return empty, empty, eval_dates

    try:
        ic_parts, hr_parts = [], []

        for split_df in (val_df, test_df):
            if split_df.empty:
                continue
            preds = trainer.predict(
                booster, individual, split_df, feature_df=feature_df
            )
            labels = split_df["label"]
            mask = preds.notna() & labels.notna()
            if mask.sum() == 0:
                continue

            ic_parts.append(_ic_per_date(preds[mask], labels[mask], split_df.index))
            hr_parts.append(_hitrate_per_date(preds[mask], labels[mask]))

        ic_series = pd.concat(ic_parts).sort_index() if ic_parts else empty
        hr_series = pd.concat(hr_parts).sort_index() if hr_parts else empty

        return ic_series, hr_series, eval_dates

    except Exception as exc:
        logger.warning("Predict failed: %s", exc)
        return empty, empty, eval_dates


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_walk_forward_metrics(
    individual: Individual,
    wf_folds: list,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """
    Re-run the individual on every walk-forward fold used by evolution.

    Returns one row per fold plus daily validation IC/hit-rate series.
    """
    trainer = Trainer()
    fold_rows: list[dict] = []
    ic_parts: list[pd.Series] = []
    hr_parts: list[pd.Series] = []
    date_parts: list[pd.DatetimeIndex] = []

    for fold in wf_folds:
        try:
            _, train_pred, val_pred, train_labels, val_labels = trainer.train(
                individual,
                fold.train_df,
                fold.val_df,
                feature_df=feature_df,
                mode="wf",
            )
        except Exception as exc:
            logger.warning("WF train failed on %s: %s", fold.name, exc)
            continue

        train_ic_series = _ic_per_date(
            train_pred, train_labels, fold.train_df.index
        )
        val_ic_series = _ic_per_date(
            val_pred, val_labels, fold.val_df.index
        )
        hr_series = _hitrate_per_date(val_pred, val_labels)

        train_mean_ic = _safe_mean(train_ic_series)
        val_mean_ic = _safe_mean(val_ic_series)
        hit_rate = _safe_mean(hr_series)
        baseline = _random_hit_rate_baseline(
            fold.val_df.index.get_level_values("ticker").nunique()
        )

        fold_rows.append(
            {
                "name": fold.name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "val_start": fold.val_start,
                "val_end": fold.val_end,
                "train_mean_ic": train_mean_ic,
                "val_mean_ic": val_mean_ic,
                "hit_rate": hit_rate,
                "hit_excess": hit_rate - baseline,
                "overfit_gap": max(0.0, train_mean_ic - val_mean_ic),
                "ic_dates": int(val_ic_series.notna().sum()),
                "hr_dates": int(hr_series.notna().sum()),
            }
        )

        if not val_ic_series.empty:
            ic_parts.append(val_ic_series.sort_index())
        if not hr_series.empty:
            hr_parts.append(hr_series.sort_index())
        date_parts.append(_date_index_from_frames(fold.val_df))

    fold_metrics = pd.DataFrame(fold_rows)
    wf_ic_series = (
        pd.concat(ic_parts).sort_index()
        if ic_parts
        else pd.Series(dtype=float)
    )
    wf_hr_series = (
        pd.concat(hr_parts).sort_index()
        if hr_parts
        else pd.Series(dtype=float)
    )
    wf_dates = _combine_date_indexes(date_parts)
    return fold_metrics, wf_ic_series, wf_hr_series, wf_dates


def _add_period_shading(ax, xmin, xmax, test_ts, val_ts=None):
    """Tô vùng val (xanh nhạt) và test (vàng nhạt), kẻ đường phân cách."""
    if val_ts is not None and xmin < val_ts:
        ax.axvspan(xmin, val_ts, alpha=0.035, color="#94A3B8")
        ax.axvspan(val_ts, test_ts, alpha=0.06, color="#3B82F6")
        ax.axvline(val_ts, color="#3B82F6", linewidth=1.0, linestyle=":", alpha=0.8)
    else:
        ax.axvspan(xmin, test_ts, alpha=0.06, color="#3B82F6")
    ax.axvspan(test_ts, xmax,   alpha=0.06, color="#F59E0B")
    ax.axvline(test_ts, color="#F59E0B", linewidth=1.2, linestyle=":", alpha=0.9)
    ax.axhline(0,       color="#6B7280", linewidth=0.7, linestyle="--", alpha=0.5)


def _add_xaxis_format(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def _period_means(series, test_ts):
    val_mean  = series[series.index < test_ts].mean()
    test_mean = series[series.index >= test_ts].mean()
    return val_mean, test_mean


def _safe_mean(series: pd.Series) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _combine_date_indexes(indexes: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    valid = [idx for idx in indexes if idx is not None and len(idx) > 0]
    if not valid:
        return pd.DatetimeIndex([])

    out = valid[0]
    for idx in valid[1:]:
        out = out.append(idx)
    return out.unique().sort_values()


def _date_index_from_frames(*dfs: pd.DataFrame) -> pd.DatetimeIndex:
    indexes = []
    for df in dfs:
        if df.empty:
            continue
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values("date")
        elif "date" in df.columns:
            dates = df["date"]
        else:
            dates = df.index
        indexes.append(pd.DatetimeIndex(pd.to_datetime(dates)).unique())

    if not indexes:
        return pd.DatetimeIndex([])

    out = indexes[0]
    for idx in indexes[1:]:
        out = out.append(idx)
    return out.unique().sort_values()


def _rolling_for_plot(
    series: pd.Series,
    date_index: Optional[pd.DatetimeIndex],
) -> pd.Series:
    if date_index is None or len(date_index) == 0:
        date_index = pd.DatetimeIndex(series.index).unique().sort_values()

    if series.empty:
        return pd.Series(index=date_index, dtype=float)

    rolling = series.sort_index().rolling(ROLLING_WINDOW, min_periods=1).mean()
    return rolling.reindex(date_index)


def _random_hit_rate_baseline(n_tickers: Optional[int]) -> float:
    if not n_tickers:
        return float("nan")
    return min(1.0, HIT_RATE_TOP_K / max(int(n_tickers), 1))


def _period_counts(
    series: pd.Series,
    date_index: pd.DatetimeIndex,
    test_ts: pd.Timestamp,
) -> tuple[int, int, int, int]:
    if date_index is None or len(date_index) == 0:
        date_index = pd.DatetimeIndex(series.index).unique().sort_values()

    observed = pd.DatetimeIndex(series.dropna().index).unique()
    val_total = int((date_index < test_ts).sum())
    test_total = int((date_index >= test_ts).sum())
    val_obs = int((observed < test_ts).sum())
    test_obs = int((observed >= test_ts).sum())
    return val_obs, val_total, test_obs, test_total


# ─── Plot individual (IC + Hit Rate) ─────────────────────────────────────────

def _as_finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _aggregate_wf_metrics(
    wf_metrics: Optional[pd.DataFrame],
    wf_ic_series: Optional[pd.Series],
) -> dict[str, float]:
    if wf_metrics is None or wf_metrics.empty:
        return {}

    clean_ic = (
        wf_ic_series.replace([np.inf, -np.inf], np.nan).dropna()
        if wf_ic_series is not None
        else pd.Series(dtype=float)
    )
    if len(clean_ic) > 1:
        wf_icir = float(clean_ic.mean()) / (float(clean_ic.std()) + 1e-9)
    else:
        wf_icir = 0.0
    wf_icir_scaled = float(np.clip(wf_icir, -3.0, 3.0)) / 3.0

    out = {
        "wf_mean_ic": float(wf_metrics["val_mean_ic"].mean()),
        "wf_icir": wf_icir,
        "wf_icir_scaled": wf_icir_scaled,
        "wf_hit_rate": float(wf_metrics["hit_rate"].mean()),
        "wf_hit_excess": float(wf_metrics["hit_excess"].mean()),
        "wf_ic_std": float(wf_metrics["val_mean_ic"].std(ddof=0)),
        "bad_fold_ratio": float((wf_metrics["val_mean_ic"] <= 0.0).mean()),
        "wf_overfit_gap": float(wf_metrics["overfit_gap"].mean()),
        "wf_train_mean_ic": float(wf_metrics["train_mean_ic"].mean()),
        "n_folds": float(len(wf_metrics)),
    }

    w = FITNESS_WEIGHTS
    out["score"] = (
        w["wf_mean_ic"] * out["wf_mean_ic"]
        + w["wf_icir"] * out["wf_icir_scaled"]
        + w["wf_hit_excess"] * out["wf_hit_excess"]
        + w["wf_ic_std"] * out["wf_ic_std"]
        + w["bad_fold_ratio"] * out["bad_fold_ratio"]
        + w["wf_overfit_gap"] * out["wf_overfit_gap"]
    )
    return out


def _wf_metric_summary_text(wf_aggregate: Optional[dict[str, float]]) -> str:
    values = wf_aggregate or {}
    if not values:
        return "WF metrics: n/a"

    def _fmt(key: str, spec: str) -> str:
        value = values.get(key)
        return "nan" if value is None else format(value, spec)

    return (
        "WF metrics: "
        f"wf_mean_ic={_fmt('wf_mean_ic', '+.4f')} | "
        f"wf_icir={_fmt('wf_icir', '+.4f')} | "
        f"wf_hit_rate={_fmt('wf_hit_rate', '.3f')} | "
        f"wf_hit_excess={_fmt('wf_hit_excess', '+.3f')} | "
        f"wf_ic_std={_fmt('wf_ic_std', '.4f')} | "
        f"bad_fold_ratio={_fmt('bad_fold_ratio', '.2f')} | "
        f"wf_overfit_gap={_fmt('wf_overfit_gap', '.4f')}"
    )


def _add_wf_fold_table(
    ax,
    wf_metrics: pd.DataFrame,
    wf_aggregate: Optional[dict[str, float]] = None,
) -> None:
    ax.axis("off")
    if wf_metrics is None or wf_metrics.empty:
        ax.text(
            0.5, 0.5, "WF fold metrics are not available",
            ha="center", va="center", fontsize=9, color="#6B7280",
        )
        return

    cols = ["Fold", "Val window", "Train IC", "Val IC", "Hit", "Gap", "Days"]
    rows = []
    for _, row in wf_metrics.iterrows():
        val_start = pd.Timestamp(row["val_start"]).strftime("%Y-%m-%d")
        val_end = pd.Timestamp(row["val_end"]).strftime("%Y-%m-%d")
        rows.append(
            [
                row["name"],
                f"{val_start} -> {val_end}",
                f"{row['train_mean_ic']:+.3f}",
                f"{row['val_mean_ic']:+.3f}",
                f"{row['hit_rate']:.3f}",
                f"{row['overfit_gap']:.3f}",
                f"{int(row['ic_dates'])}/{int(row['hr_dates'])}",
            ]
        )

    ax.set_title(
        _wf_metric_summary_text(wf_aggregate),
        fontsize=8,
        loc="left",
        pad=6,
    )
    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.08, 0.25, 0.11, 0.11, 0.10, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.18)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold", color="#111827")
            continue
        val_ic = float(wf_metrics.iloc[row_idx - 1]["val_mean_ic"])
        cell.set_edgecolor("#D1D5DB")
        if col_idx == 3:
            cell.set_facecolor("#DCFCE7" if val_ic > 0 else "#FEE2E2")


def plot_individual(
    ic_series:  pd.Series,
    hr_series:  pd.Series,
    entry:      dict,
    val_start:  str,
    test_start: str,
    out_path:   Path,
    date_index: Optional[pd.DatetimeIndex] = None,
    final_date_index: Optional[pd.DatetimeIndex] = None,
    n_tickers:  Optional[int] = None,
    wf_metrics: Optional[pd.DataFrame] = None,
    wf_aggregate: Optional[dict[str, float]] = None,
    wf_ic_series: Optional[pd.Series] = None,
    wf_hr_series: Optional[pd.Series] = None,
) -> None:
    """
    2 subplot trong cùng 1 figure:
      Top    — IC daily (bar) + rolling-5 mean (line)
      Bottom — Hit rate daily (bar) + rolling-5 mean (line) + random baseline
    """
    if ic_series.empty:
        logger.warning("IC series rỗng — bỏ qua rank %d", entry["rank"])
        return

    val_ts  = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)
    if date_index is None or len(date_index) == 0:
        date_index = pd.DatetimeIndex(ic_series.index).unique().sort_values()
    xmin    = date_index.min()
    xmax    = date_index.max()

    ic_roll = _rolling_for_plot(ic_series, date_index)
    hr_roll = _rolling_for_plot(hr_series, date_index)

    val_ic,   test_ic   = _period_means(ic_series, test_ts)
    val_hr,   test_hr   = _period_means(hr_series, test_ts) if not hr_series.empty else (float("nan"), float("nan"))
    n_ticker_baseline = _random_hit_rate_baseline(n_tickers)
    if final_date_index is None or len(final_date_index) == 0:
        final_date_index = date_index
    val_hr_obs, val_hr_total, test_hr_obs, test_hr_total = _period_counts(
        hr_series, final_date_index, test_ts
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    show_wf_table = wf_metrics is not None and not wf_metrics.empty
    fig_height = 9.0 + (min(3.0, 0.35 * len(wf_metrics)) if show_wf_table else 0.0)
    height_ratios = (
        [1, 1, 0.30 + 0.05 * len(wf_metrics)]
        if show_wf_table
        else [1, 1]
    )
    fig = plt.figure(figsize=(15, fig_height))
    gs  = gridspec.GridSpec(
        len(height_ratios), 1, figure=fig,
        hspace=0.42,
        height_ratios=height_ratios,
    )
    ax_ic = fig.add_subplot(gs[0])
    ax_hr = fig.add_subplot(gs[1])
    ax_wf = fig.add_subplot(gs[2]) if show_wf_table else None

    # ── Suptitle ─────────────────────────────────────────────────────────────
    rank   = entry["rank"]
    score  = _as_finite_float((wf_aggregate or {}).get("score"))
    n_gene = entry["n_genes"]
    genes_str = " | ".join(entry["genes"][:3])
    if len(entry["genes"]) > 3:
        genes_str += f"  … (+{len(entry['genes'])-3} more)"

    fig.suptitle(
        f"Rank #{rank}  |  Recomputed WF Score={(score if score is not None else float('nan')):.4f}  |  n_genes={n_gene}\n"
        f"Genes: {genes_str}",
        fontsize=10, fontweight="bold", y=0.98,
    )

    # ── Panel 1: IC ───────────────────────────────────────────────────────────
    _add_period_shading(ax_ic, xmin, xmax, test_ts, val_ts=val_ts)

    ax_ic.bar(
        ic_series.index, ic_series.values,
        color=np.where(ic_series.values >= 0, "#93C5FD", "#FCA5A5"),
        width=1.5, alpha=0.4, label="IC daily",
    )
    ax_ic.plot(
        ic_roll.index, ic_roll.values,
        color="#1D4ED8", linewidth=2.0,
        label=f"IC rolling-{ROLLING_WINDOW}",
    )
    if wf_ic_series is not None and not wf_ic_series.empty:
        wf_ic_roll = _rolling_for_plot(wf_ic_series, date_index)
        ax_ic.plot(
            wf_ic_roll.index, wf_ic_roll.values,
            color="#7C3AED", linewidth=1.4, linestyle="--", alpha=0.85,
            label=f"WF val IC rolling-{ROLLING_WINDOW}",
        )

    # Annotation IC
    ax_ic.set_title(
        f"IC (Spearman)    Val={val_ic:+.4f}  |  Test={test_ic:+.4f}",
        fontsize=9, pad=6,
    )
    ax_ic.set_ylabel("IC", fontsize=8)
    ax_ic.legend(loc="upper right", fontsize=7, framealpha=0.8)
    _add_xaxis_format(ax_ic)

    # Val/Test label
    y_top = ax_ic.get_ylim()[1]
    ax_ic.text(val_ts,  y_top * 0.88, "← VAL",  color="#3B82F6", fontsize=7, ha="left")
    ax_ic.text(test_ts, y_top * 0.88, "TEST →", color="#F59E0B", fontsize=7, ha="left")

    # ── Panel 2: Hit Rate ─────────────────────────────────────────────────────
    if not hr_series.empty:
        _add_period_shading(ax_hr, xmin, xmax, test_ts, val_ts=val_ts)

        # Random baseline
        ax_hr.axhline(
            n_ticker_baseline,
            color="#9CA3AF", linewidth=1.0, linestyle="--", alpha=0.8,
            label=f"Random baseline ({n_ticker_baseline:.2f})",
        )

        ax_hr.bar(
            hr_series.index, hr_series.values,
            color=np.where(hr_series.values >= n_ticker_baseline, "#6EE7B7", "#FCA5A5"),
            width=1.5, alpha=0.4, label="Hit rate daily",
        )
        ax_hr.plot(
            hr_roll.index, hr_roll.values,
            color="#059669", linewidth=2.0,
            label=f"Hit rate rolling-{ROLLING_WINDOW}",
        )
        if wf_hr_series is not None and not wf_hr_series.empty:
            wf_hr_roll = _rolling_for_plot(wf_hr_series, date_index)
            ax_hr.plot(
                wf_hr_roll.index, wf_hr_roll.values,
                color="#7C3AED", linewidth=1.4, linestyle="--", alpha=0.85,
                label=f"WF val hit rolling-{ROLLING_WINDOW}",
            )

        # Annotation hit rate
        lift_val  = val_hr  / n_ticker_baseline if n_ticker_baseline else float("nan")
        lift_test = test_hr / n_ticker_baseline if n_ticker_baseline else float("nan")
        ax_hr.set_title(
            f"Hit Rate (top-{HIT_RATE_TOP_K})    "
            f"Val={val_hr:.3f} ({lift_val:.1f}x)  |  "
            f"Test={test_hr:.3f} ({lift_test:.1f}x)    "
            f"dates={val_hr_obs}/{val_hr_total} val, {test_hr_obs}/{test_hr_total} test",
            fontsize=9, pad=6,
        )
        ax_hr.set_ylabel("Hit Rate", fontsize=8)
        ax_hr.set_ylim(0, 1.05)
        ax_hr.legend(loc="upper right", fontsize=7, framealpha=0.8)
        _add_xaxis_format(ax_hr)

        y_top2 = ax_hr.get_ylim()[1]
        ax_hr.text(val_ts,  y_top2 * 0.92, "← VAL",  color="#3B82F6", fontsize=7, ha="left")
        ax_hr.text(test_ts, y_top2 * 0.92, "TEST →", color="#F59E0B", fontsize=7, ha="left")
    else:
        ax_hr.text(0.5, 0.5, "Hit rate không có dữ liệu",
                   ha="center", va="center", transform=ax_hr.transAxes,
                   fontsize=10, color="#6B7280")

    ax_hr.set_xlabel("Date", fontsize=8)

    if ax_wf is not None:
        _add_wf_fold_table(
            ax_wf,
            wf_metrics,
            wf_aggregate=wf_aggregate,
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ─── Summary charts ───────────────────────────────────────────────────────────

def _plot_summary_generic(
    all_series:  dict[int, pd.Series],
    entries:     list[dict],
    val_start:   str,
    test_start:  str,
    out_path:    Path,
    title:       str,
    ylabel:      str,
    baseline:    Optional[float] = None,
    date_index:  Optional[pd.DatetimeIndex] = None,
) -> None:
    """Generic summary chart (reused for IC và Hit Rate)."""
    if not all_series:
        return

    test_ts = pd.Timestamp(test_start)
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap    = plt.get_cmap("tab20", len(all_series))
    xmin, xmax = None, None

    for i, (rank, series) in enumerate(sorted(all_series.items())):
        if series.empty:
            continue
        rolling = _rolling_for_plot(series, date_index)
        score   = next((e["score"] for e in entries if e["rank"] == rank), 0)
        val_m, test_m = _period_means(series, test_ts)
        ax.plot(
            rolling.index, rolling.values,
            color=cmap(i), linewidth=1.4, alpha=0.85,
            label=f"#{rank} s={score:.3f} v={val_m:.3f} t={test_m:.3f}",
        )
        xmin = rolling.index.min() if xmin is None else min(xmin, rolling.index.min())
        xmax = rolling.index.max() if xmax is None else max(xmax, rolling.index.max())

    if xmin is None:
        plt.close(fig)
        return

    _add_period_shading(ax, xmin, xmax, test_ts, val_ts=pd.Timestamp(val_start))

    if baseline is not None:
        ax.axhline(baseline, color="#9CA3AF", linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"Random baseline ({baseline:.2f})")

    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.1
    ax.text(pd.Timestamp(val_start), ymax * 0.92,
            "← VAL",  color="#3B82F6", fontsize=9, ha="left")
    ax.text(test_ts, ymax * 0.92,
            "TEST →", color="#F59E0B", fontsize=9, ha="left")

    ax.set_title(
        f"{title} Rolling-{ROLLING_WINDOW} — {len(all_series)} Individuals",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    ax.legend(
        loc="upper left", fontsize=6.5,
        ncol=max(1, len(all_series) // 15 + 1),
        framealpha=0.85,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary saved: %s", out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def analyze(
    data_dir:     str,
    archive_path: str,
    val_start:    str           = VAL_START,
    test_start:   str           = TEST_START,
    test_end:     Optional[str] = TEST_END,
    wf_end:       str           = WF_END,
    wf_min_train_months: int    = WF_MIN_TRAIN_MONTHS,
    wf_val_months: int          = WF_VAL_MONTHS,
    wf_step_months: int         = WF_STEP_MONTHS,
    wf_purge_days: int          = WF_PURGE_DAYS,
    top:          Optional[int] = None,
    out_dir:      str           = "results/chart",
    tickers:      Optional[list[str]] = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data from %s …", data_dir)
    df = load_from_dir(data_dir, tickers=tickers)
    validate_ohlcv(df)

    # ── Split ─────────────────────────────────────────────────────────────────
    labeled_df = label_dataframe(df)
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
    )
    feature_df = labeled_df.sort_index()
    logger.info(
        "train=%d rows | val=%d rows | test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    raw_dates = df.index.get_level_values("date")
    wf_raw_df = df[raw_dates < pd.Timestamp(wf_end)].copy().sort_index()
    wf_labeled_df = label_dataframe(wf_raw_df)
    wf_folds = make_walk_forward_folds(
        wf_labeled_df,
        wf_end=wf_end,
        min_train_months=wf_min_train_months,
        val_months=wf_val_months,
        step_months=wf_step_months,
        purge_days=wf_purge_days,
    )
    wf_feature_df = wf_labeled_df.sort_index()
    logger.info(
        "WF folds=%d | end=%s | min_train=%dm | val=%dm | step=%dm | purge=%dd",
        len(wf_folds),
        wf_end,
        wf_min_train_months,
        wf_val_months,
        wf_step_months,
        wf_purge_days,
    )
    for fold in wf_folds:
        logger.info(
            "WF %s: train [%s -> %s) %d rows | val [%s -> %s) %d rows",
            fold.name,
            fold.train_start.date(),
            fold.train_end.date(),
            len(fold.train_df),
            fold.val_start.date(),
            fold.val_end.date(),
            len(fold.val_df),
        )

    # ── Load archive ──────────────────────────────────────────────────────────
    entries = load_archive_json(Path(archive_path))
    if top is not None:
        entries = entries[:top]
        logger.info("Chỉ xử lý top %d individuals", top)

    # ── Process each individual ───────────────────────────────────────────────
    all_ic: dict[int, pd.Series] = {}
    all_hr: dict[int, pd.Series] = {}

    for entry in entries:
        rank = entry["rank"]
        logger.info("── Rank #%d (genes=%d) ──", rank, entry["n_genes"])

        ind = json_to_individual(entry)
        ic_series, hr_series, eval_dates = compute_series(
            ind, train_df, val_df, test_df, feature_df=feature_df
        )
        wf_metrics, wf_ic_series, wf_hr_series, wf_dates = compute_walk_forward_metrics(
            ind, wf_folds, wf_feature_df
        )
        wf_aggregate = _aggregate_wf_metrics(wf_metrics, wf_ic_series)
        recomputed_score = _as_finite_float(wf_aggregate.get("score"))
        if recomputed_score is not None:
            entry["score"] = recomputed_score
        entry.update(wf_aggregate)
        plot_dates = _combine_date_indexes([wf_dates, eval_dates])
        if len(plot_dates) == 0:
            plot_dates = eval_dates

        if ic_series.empty:
            logger.warning("  Rank #%d: empty — skip", rank)
            continue

        all_ic[rank] = ic_series
        all_hr[rank] = hr_series

        val_ic,  test_ic  = _period_means(ic_series, pd.Timestamp(test_start))
        val_hr,  test_hr  = _period_means(hr_series,  pd.Timestamp(test_start)) \
                            if not hr_series.empty else (float("nan"), float("nan"))
        n_tickers_est = val_df.index.get_level_values("ticker").nunique()
        baseline = _random_hit_rate_baseline(n_tickers_est)
        logger.info(
            "  IC  val=%.4f test=%.4f | Hit val=%.3f (%.1fx) test=%.3f (%.1fx) | Hit dates=%d/%d",
            val_ic, test_ic,
            val_hr,  val_hr  / baseline,
            test_hr, test_hr / baseline,
            len(hr_series), len(eval_dates),
        )
        if not wf_metrics.empty:
            logger.info(
                "  WF recomputed score=%.4f | folds=%d | meanIC=%.4f | ICIR=%.4f | hit=%.3f | gap=%.4f | bad=%.0f%%",
                recomputed_score if recomputed_score is not None else float("nan"),
                int(wf_aggregate.get("n_folds", len(wf_metrics))),
                wf_aggregate.get("wf_mean_ic", float("nan")),
                wf_aggregate.get("wf_icir", float("nan")),
                wf_aggregate.get("wf_hit_rate", float("nan")),
                wf_aggregate.get("wf_overfit_gap", float("nan")),
                100.0 * wf_aggregate.get("bad_fold_ratio", float("nan")),
            )

        fname = f"rank_{rank:02d}_score_{(recomputed_score if recomputed_score is not None else float('nan')):.4f}.png"
        plot_individual(ic_series, hr_series, entry, val_start, test_start,
                        out_path / fname, date_index=plot_dates,
                        final_date_index=eval_dates,
                        n_tickers=n_tickers_est,
                        wf_metrics=wf_metrics,
                        wf_aggregate=wf_aggregate,
                        wf_ic_series=wf_ic_series,
                        wf_hr_series=wf_hr_series)

    # ── Summary charts ────────────────────────────────────────────────────────
    if all_ic:
        _plot_summary_generic(
            all_ic, entries, val_start, test_start,
            out_path / "ic_summary.png",
            title="IC (Spearman)", ylabel=f"IC Rolling-{ROLLING_WINDOW}",
            date_index=eval_dates,
        )
        _plot_summary_generic(
            all_hr, entries, val_start, test_start,
            out_path / "hitrate_summary.png",
            title="Hit Rate", ylabel=f"Hit Rate Rolling-{ROLLING_WINDOW}",
            baseline=baseline, date_index=eval_dates,
        )

    total = len(all_ic)
    logger.info("=== Done. %d individuals → %d charts in %s ===",
                total, total + 2, out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evo_Finance Analyzer — IC + Hit Rate charts",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data-dir",   required=True, metavar="DIR")
    parser.add_argument("--archive",    required=True, metavar="FILE")
    parser.add_argument("--val-start",  default=VAL_START,  metavar="DATE",
                        help=f"default: {VAL_START}")
    parser.add_argument("--test-start", default=TEST_START, metavar="DATE",
                        help=f"default: {TEST_START}")
    parser.add_argument("--test-end",   default=TEST_END,   metavar="DATE")
    parser.add_argument("--wf-end",     default=WF_END,     metavar="DATE",
                        help=f"default: {WF_END}")
    parser.add_argument("--wf-min-train-months", type=int,
                        default=WF_MIN_TRAIN_MONTHS)
    parser.add_argument("--wf-val-months", type=int, default=WF_VAL_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=WF_STEP_MONTHS)
    parser.add_argument("--wf-purge-days", type=int, default=WF_PURGE_DAYS)
    parser.add_argument("--top",        type=int, default=None, metavar="N")
    parser.add_argument("--out-dir",    default="results/chart", metavar="DIR")
    parser.add_argument("--tickers",    nargs="+", default=None, metavar="TICKER")

    args = parser.parse_args()
    analyze(
        data_dir     = args.data_dir,
        archive_path = args.archive,
        val_start    = args.val_start,
        test_start   = args.test_start,
        test_end     = args.test_end,
        wf_end       = args.wf_end,
        wf_min_train_months = args.wf_min_train_months,
        wf_val_months       = args.wf_val_months,
        wf_step_months      = args.wf_step_months,
        wf_purge_days       = args.wf_purge_days,
        top          = args.top,
        out_dir      = args.out_dir,
        tickers      = args.tickers,
    )
