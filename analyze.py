"""
Evo_Finance — Analyze
──────────────────────
Load archive JSON → retrain mỗi individual → vẽ biểu đồ IC theo ngày
(từ val đến hết test, đường rolling-5 mean) → lưu vào results/chart/

Usage
-----
    python analyze.py --data-dir data/raw --archive results/archive.json

    # Chỉ vẽ top-N individuals
    python analyze.py --data-dir data/raw --archive results/archive.json --top 10

    # Custom ngày split (phải khớp với lúc train)
    python analyze.py --data-dir data/raw --archive results/archive.json \\
        --val-start 2022-01-01 --test-start 2023-07-01

Output
------
    results/chart/rank_{N:02d}_score_{score:.4f}.png   — 1 file per individual
    results/chart/ic_summary.png                        — tất cả individuals trên 1 chart
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — không cần display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    VAL_START, TEST_START, TEST_END, DEFAULT_WINDOW,
)
from data.loader       import load_from_dir
from model.data_utils  import split_and_label, validate_ohlcv
from model.trainer     import Trainer, build_feature_matrix
from mutator.gene      import Gene, Individual
from fitness.fitness   import _ic_per_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evo_finance.analyze")

ROLLING_WINDOW = 5   # rolling mean cho đường IC


# ─── Load archive JSON ────────────────────────────────────────────────────────

def load_archive_json(path: Path) -> list[dict]:
    with open(path) as f:
        entries = json.load(f)
    logger.info("Loaded %d individuals từ %s", len(entries), path)
    return entries


def json_to_individual(entry: dict) -> Individual:
    """Tạo Individual từ 1 entry trong archive JSON."""
    genes = [Gene(formula=f) for f in entry["genes"]]
    ind   = Individual(
        genes      = genes,
        score      = entry.get("score"),
        metrics    = {k: v for k, v in entry.items()
                      if k in ("val_mean_ic","val_icir","hit_rate","overfit_gap")},
        generation = entry.get("generation", 0),
    )
    return ind


# ─── Compute IC series val+test ───────────────────────────────────────────────

def compute_ic_series(
    individual: Individual,
    train_df:   pd.DataFrame,
    eval_df:    pd.DataFrame,   # val + test gộp lại
) -> pd.Series:
    """
    Retrain individual trên train_df, predict trên eval_df,
    tính IC per date → trả về pd.Series index = date.
    """
    trainer = Trainer()
    # Cần val_df để train (early stopping) — dùng phần đầu của eval_df làm val proxy
    # Thực tế: retrain với full train_df, dùng eval_df[val] làm val set
    # Ta tách eval_df ra val và test bằng cách dùng booster từ train
    try:
        # Dùng eval_df cả khối làm val (early stopping monitor)
        booster, _, _, _, _ = trainer.train(individual, train_df, eval_df)
    except Exception as exc:
        logger.warning("Train failed cho %s: %s", individual.formulas[:2], exc)
        return pd.Series(dtype=float)

    # Predict trên toàn eval_df (val + test)
    try:
        X_eval = build_feature_matrix(individual, eval_df)
        mask   = X_eval.notna().all(axis=1) & eval_df["label"].notna()
        preds  = pd.Series(
            booster.predict(X_eval[mask]),
            index=X_eval[mask].index,
            name="pred",
        )
        labels = eval_df.loc[mask, "label"]
        ic_series = _ic_per_date(preds, labels, eval_df.index)
        return ic_series.sort_index()
    except Exception as exc:
        logger.warning("Predict/IC failed: %s", exc)
        return pd.Series(dtype=float)


# ─── Plot individual ──────────────────────────────────────────────────────────

def plot_individual(
    ic_series:   pd.Series,
    entry:       dict,
    val_start:   str,
    test_start:  str,
    out_path:    Path,
) -> None:
    """
    Vẽ IC daily + rolling-5 mean cho 1 individual.
    Vùng val và test được tô màu khác nhau.
    """
    if ic_series.empty:
        logger.warning("IC series rỗng — bỏ qua rank %d", entry["rank"])
        return

    val_ts  = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)

    rolling = ic_series.rolling(ROLLING_WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))

    # Tô vùng val / test
    xmin = ic_series.index.min()
    xmax = ic_series.index.max()
    ax.axvspan(xmin,    test_ts, alpha=0.06, color="#3B82F6", label="Val period")
    ax.axvspan(test_ts, xmax,   alpha=0.06, color="#F59E0B", label="Test period")

    # Đường zero
    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.6)

    # Ranh giới val/test
    ax.axvline(test_ts, color="#F59E0B", linewidth=1.2, linestyle=":", alpha=0.9)

    # IC daily (mờ)
    ax.bar(
        ic_series.index, ic_series.values,
        color=np.where(ic_series.values >= 0, "#93C5FD", "#FCA5A5"),
        width=1.5, alpha=0.4, label="IC daily",
    )

    # Rolling-5 mean (đậm)
    ax.plot(
        rolling.index, rolling.values,
        color="#1D4ED8", linewidth=2.0,
        label=f"IC rolling-{ROLLING_WINDOW} mean",
    )

    # Metrics
    val_mask  = ic_series.index < test_ts
    test_mask = ic_series.index >= test_ts
    val_ic    = ic_series[val_mask].mean()
    test_ic   = ic_series[test_mask].mean()

    rank   = entry["rank"]
    score  = entry["score"]
    n_gene = entry["n_genes"]
    genes_str = " | ".join(entry["genes"][:4])
    if len(entry["genes"]) > 4:
        genes_str += f" … (+{len(entry['genes'])-4})"

    title = (
        f"Rank #{rank}  •  Score={score:.4f}  •  "
        f"Val IC={val_ic:.4f}  •  Test IC={test_ic:.4f}  •  "
        f"n_genes={n_gene}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("IC (Spearman)", fontsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # Annotation genes
    ax.text(
        0.01, 0.02, genes_str,
        transform=ax.transAxes,
        fontsize=7, color="#4B5563",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Legend
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Val/Test label trên trục x
    y_annot = ax.get_ylim()[1] * 0.92
    ax.text(val_ts,  y_annot, "← VAL",  color="#3B82F6", fontsize=8, ha="left")
    ax.text(test_ts, y_annot, "TEST →", color="#F59E0B", fontsize=8, ha="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ─── Summary chart (all individuals) ─────────────────────────────────────────

def plot_summary(
    all_ic:     dict[int, pd.Series],   # rank → ic_series
    entries:    list[dict],
    val_start:  str,
    test_start: str,
    out_path:   Path,
) -> None:
    """Vẽ tất cả rolling IC trên 1 chart để so sánh."""
    if not all_ic:
        return

    test_ts = pd.Timestamp(test_start)

    fig, ax = plt.subplots(figsize=(16, 7))

    cmap   = plt.cm.get_cmap("tab20", len(all_ic))
    xmin, xmax = None, None

    for i, (rank, ic_series) in enumerate(sorted(all_ic.items())):
        if ic_series.empty:
            continue
        rolling = ic_series.rolling(ROLLING_WINDOW, min_periods=1).mean()
        score   = next(e["score"] for e in entries if e["rank"] == rank)
        ax.plot(
            rolling.index, rolling.values,
            color=cmap(i), linewidth=1.4, alpha=0.85,
            label=f"#{rank} score={score:.4f}",
        )
        if xmin is None:
            xmin = rolling.index.min()
            xmax = rolling.index.max()
        else:
            xmin = min(xmin, rolling.index.min())
            xmax = max(xmax, rolling.index.max())

    if xmin is None:
        plt.close(fig)
        return

    ax.axvspan(xmin,    test_ts, alpha=0.05, color="#3B82F6")
    ax.axvspan(test_ts, xmax,   alpha=0.05, color="#F59E0B")
    ax.axvline(test_ts, color="#F59E0B", linewidth=1.5, linestyle=":", alpha=0.9)
    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.5)

    y_annot = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.1
    ax.text(pd.Timestamp(val_start),  y_annot * 0.92, "← VAL",
            color="#3B82F6", fontsize=9, ha="left")
    ax.text(test_ts, y_annot * 0.92, "TEST →",
            color="#F59E0B", fontsize=9, ha="left")

    ax.set_title(
        f"IC Rolling-{ROLLING_WINDOW} Mean — Tất cả {len(all_ic)} Individuals",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel(f"IC Rolling-{ROLLING_WINDOW}", fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    ax.legend(
        loc="upper left", fontsize=7,
        ncol=max(1, len(all_ic) // 15 + 1),
        framealpha=0.85,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary chart saved: %s", out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def analyze(
    data_dir:   str,
    archive_path: str,
    val_start:  str = VAL_START,
    test_start: str = TEST_START,
    test_end:   Optional[str] = TEST_END,
    top:        Optional[int] = None,
    out_dir:    str = "results/chart",
    tickers:    Optional[list[str]] = None,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data from %s …", data_dir)
    df = load_from_dir(data_dir, tickers=tickers)
    validate_ohlcv(df)

    # ── Split ─────────────────────────────────────────────────────────────────
    logger.info("Splitting: val_start=%s test_start=%s test_end=%s",
                val_start, test_start, test_end)
    train_df, val_df, test_df = split_and_label(
        df, val_start=val_start, test_start=test_start, test_end=test_end,
    )

    # Val + test gộp lại để compute IC liên tục từ val → hết test
    eval_df = pd.concat([val_df, test_df]).sort_index()
    logger.info(
        "train=%d rows | eval(val+test)=%d rows (%d val + %d test)",
        len(train_df), len(eval_df), len(val_df), len(test_df),
    )

    # ── Load archive ──────────────────────────────────────────────────────────
    entries = load_archive_json(Path(archive_path))
    if top is not None:
        entries = entries[:top]
        logger.info("Chỉ xử lý top %d individuals", top)

    # ── Process each individual ───────────────────────────────────────────────
    all_ic: dict[int, pd.Series] = {}

    for entry in entries:
        rank = entry["rank"]
        logger.info("── Processing rank #%d (score=%.4f, genes=%d) ──",
                    rank, entry["score"], entry["n_genes"])

        ind = json_to_individual(entry)
        ic_series = compute_ic_series(ind, train_df, eval_df)

        if ic_series.empty:
            logger.warning("  Rank #%d: IC series rỗng — bỏ qua", rank)
            continue

        all_ic[rank] = ic_series

        # Stats
        val_mask  = ic_series.index < pd.Timestamp(test_start)
        test_mask = ic_series.index >= pd.Timestamp(test_start)
        val_ic    = ic_series[val_mask].mean()
        test_ic   = ic_series[test_mask].mean()
        logger.info("  Val IC=%.4f | Test IC=%.4f | n_dates=%d",
                    val_ic, test_ic, len(ic_series))

        # Plot individual
        fname    = f"rank_{rank:02d}_score_{entry['score']:.4f}.png"
        out_file = out_path / fname
        plot_individual(ic_series, entry, val_start, test_start, out_file)

    # ── Summary chart ─────────────────────────────────────────────────────────
    if all_ic:
        plot_summary(all_ic, entries, val_start, test_start,
                     out_path / "ic_summary.png")

    logger.info("=== Done. %d charts saved to %s ===", len(all_ic) + 1, out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evo_Finance Analyzer — vẽ IC chart cho các individuals trong archive",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data-dir",   required=True, metavar="DIR",
                        help="Thư mục chứa CSV từ craw_data.py  (e.g. data/raw)")
    parser.add_argument("--archive",    required=True, metavar="FILE",
                        help="File archive JSON  (e.g. results/archive.json)")
    parser.add_argument("--val-start",  default=VAL_START,  metavar="DATE",
                        help=f"Ngày đầu val   (default: {VAL_START})")
    parser.add_argument("--test-start", default=TEST_START, metavar="DATE",
                        help=f"Ngày đầu test  (default: {TEST_START})")
    parser.add_argument("--test-end",   default=TEST_END,   metavar="DATE",
                        help=f"Ngày cuối test (default: hết data)")
    parser.add_argument("--top",        type=int, default=None, metavar="N",
                        help="Chỉ vẽ top-N individuals (default: tất cả)")
    parser.add_argument("--out-dir",    default="results/chart", metavar="DIR",
                        help="Thư mục lưu chart (default: results/chart)")
    parser.add_argument("--tickers",    nargs="+", default=None, metavar="TICKER",
                        help="Chỉ load một số ticker (default: tất cả)")

    args = parser.parse_args()
    analyze(
        data_dir     = args.data_dir,
        archive_path = args.archive,
        val_start    = args.val_start,
        test_start   = args.test_start,
        test_end     = args.test_end,
        top          = args.top,
        out_dir      = args.out_dir,
        tickers      = args.tickers,
    )