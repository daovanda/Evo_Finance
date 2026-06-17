"""
Evo_Finance — Main
───────────────────
Entry point for the evolutionary feature-selection loop.

Usage
-----
    python main.py --data path/to/ohlcv.parquet --budget 3600

The input file must be a Parquet (or CSV) with a MultiIndex (date, ticker)
and columns [open, high, close, low, volume].

At the end the archive is printed as a summary table and optionally saved.
"""

from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Project imports ────────────────────────────────────────────────────────────
from config.settings import (
    DEFAULT_WINDOW, RESTART_PROB, TIME_BUDGET_SECONDS,
    VAL_START, TEST_START, TEST_END,
    WF_END, WF_MIN_TRAIN_MONTHS, WF_VAL_MONTHS,
    WF_STEP_MONTHS, WF_PURGE_DAYS, DOMAIN_PRECOMPUTE_ON_START,
)
from mutator.gene       import Individual
from mutator.domain     import Domain
from mutator.mutator    import Mutator
from model.trainer      import Trainer
from model.data_utils   import (
    label_dataframe, make_walk_forward_folds,
    split_labeled_by_dates, validate_ohlcv,
)
from fitness.fitness    import (
    FitnessEvaluator, FoldPrediction, _hit_rate, _ic_per_date,
)
from archive.archive    import Archive

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("evo_finance.main")


# ─── Core loop ────────────────────────────────────────────────────────────────

def run(
    df:              pd.DataFrame,
    time_budget:     float = TIME_BUDGET_SECONDS,
    restart_prob:    float = RESTART_PROB,
    seed:            int   = 42,
    save_archive:    Optional[Path] = None,
    val_start:       str   = VAL_START,
    test_start:      str   = TEST_START,
    test_end:        Optional[str] = TEST_END,
    wf_end:          str   = WF_END,
    wf_min_train_months: int = WF_MIN_TRAIN_MONTHS,
    wf_val_months:   int   = WF_VAL_MONTHS,
    wf_step_months:  int   = WF_STEP_MONTHS,
    wf_purge_days:   int   = WF_PURGE_DAYS,
) -> Archive:
    """
    Run the evolutionary loop and return the final Archive.

    Parameters
    ----------
    df           : OHLCV DataFrame with MultiIndex (date, ticker).
    time_budget  : Wall-clock seconds to run.
    restart_prob : Probability of restarting from raw OHLCV each iteration.
    seed         : Master RNG seed.
    save_archive : If provided, write archive summary JSON to this path.
    val_start    : Ngày đầu val  (= kết thúc train). Ví dụ "2022-01-01"
    test_start   : Ngày đầu test (= kết thúc val).  Ví dụ "2023-07-01"
    test_end     : Ngày cuối test (None = hết data).
    """
    validate_ohlcv(df)
    rng = np.random.default_rng(seed)

    # ── Data split + labels ───────────────────────────────────────────────────
    logger.info("Labeling data and building walk-forward folds ...")
    labeled_df = label_dataframe(df)
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
    )
    logger.info(
        "Final sizes: train=%d rows | val=%d rows | test=%d rows",
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
    wf_feature_df = wf_labeled_df.sort_index()
    final_feature_df = labeled_df.sort_index()

    # ── Initialise components ─────────────────────────────────────────────────
    domain    = Domain()
    seed_genes = domain.seed(window=DEFAULT_WINDOW)
    if DOMAIN_PRECOMPUTE_ON_START:
        domain.precompute(wf_feature_df)
    else:
        logger.info(
            "Domain precompute skipped; formula cache will be filled lazily."
        )

    mutator   = Mutator(domain=domain, seed=int(rng.integers(1 << 31)))
    trainer   = Trainer()
    evaluator = FitnessEvaluator()
    archive   = Archive()

    # ── First individual: raw OHLCV ───────────────────────────────────────────
    logger.info("Evaluating seed individual (raw OHLCV) …")
    seed_ind = Individual.seed(window=DEFAULT_WINDOW)
    _evaluate_and_archive_wf(
        seed_ind, trainer, evaluator, archive,
        wf_folds, wf_feature_df,
    )

    # ── Evolutionary loop ─────────────────────────────────────────────────────
    iteration = 0
    t_start   = time.time()

    while time.time() - t_start < time_budget:
        iteration += 1
        elapsed = time.time() - t_start
        logger.info("── Iteration %d  (%.1fs / %.1fs) ──", iteration, elapsed, time_budget)

        # restart?
        if rng.random() < restart_prob:
            logger.info("RESTART: resetting to raw OHLCV individual.")
            parent = Individual.seed(window=DEFAULT_WINDOW)
        elif archive.is_empty():
            parent = Individual.seed(window=DEFAULT_WINDOW)
            _evaluate_and_archive_wf(parent, trainer, evaluator, archive, wf_folds, wf_feature_df)
            continue
        else:
            parent = archive.random_individual(rng)

        # mutate
        try:
            child = mutator.mutate(parent, wf_feature_df)
        except Exception as exc:
            logger.warning("Mutation failed: %s — skipping iteration.", exc)
            continue

        if len(child) == 0:
            logger.warning("Mutation produced empty individual — skipping.")
            continue

        # evaluate + archive
        admitted = _evaluate_and_archive_wf(
            child, trainer, evaluator, archive,
            wf_folds, wf_feature_df,
        )

        best_score = archive.best.score if archive.best else float("nan")
        logger.info(
            "Result: score=%.4f | admitted=%s | archive_size=%d | best=%.4f",
            child.score or float("nan"), admitted, len(archive), best_score,
        )

    # ── Test-set evaluation for all archived individuals ──────────────────────
    logger.info("=== Time budget exhausted. Running final validation/test evaluation ... ===")
    _final_evaluate_archive(
        archive, trainer, train_df, val_df, test_df, final_feature_df
    )

    # ── Output ────────────────────────────────────────────────────────────────
    _print_summary(archive, iteration, time.time() - t_start)

    if save_archive is not None:
        _save_json(archive, save_archive)

    return archive


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _evaluate_and_archive_wf(
    individual: Individual,
    trainer:    Trainer,
    evaluator:  FitnessEvaluator,
    archive:    Archive,
    wf_folds,
    feature_df: pd.DataFrame,
) -> bool:
    """Train and score an individual over all walk-forward folds."""
    fold_predictions = []
    last_booster = None

    for fold in wf_folds:
        try:
            booster, train_pred, val_pred, train_labels, val_labels = trainer.train(
                individual, fold.train_df, fold.val_df,
                feature_df=feature_df, mode="wf"
            )
        except Exception as exc:
            logger.warning("WF training failed on %s: %s", fold.name, exc)
            return False
        last_booster = booster
        fold_predictions.append(
            FoldPrediction(
                name=fold.name,
                train_pred=train_pred,
                val_pred=val_pred,
                train_labels=train_labels,
                val_labels=val_labels,
                train_df=fold.train_df,
                val_df=fold.val_df,
            )
        )

    try:
        evaluator.evaluate_walk_forward(individual, fold_predictions)
    except Exception as exc:
        logger.warning("WF fitness evaluation failed: %s", exc)
        return False

    return archive.try_add(individual, last_booster)


def _final_evaluate_archive(
    archive:   Archive,
    trainer:   Trainer,
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    test_df:   pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    """Retrain each archived individual on final split and evaluate val/test."""
    for entry in archive.entries:
        try:
            booster, _, val_pred, _, val_labels = trainer.train(
                entry.individual, train_df, val_df,
                feature_df=feature_df, mode="final"
            )
            entry.booster = booster
            entry.final_val_metrics = _prediction_metrics(
                val_pred, val_labels, val_df.index, prefix="final_val"
            )

            test_pred = trainer.predict(
                booster, entry.individual, test_df, feature_df=feature_df
            )
            test_labels = test_df["label"]
            entry.test_metrics = _prediction_metrics(
                test_pred, test_labels, test_df.index, prefix="test"
            )
            logger.info(
                "Final eval: score=%.4f | val_IC=%.4f | test_IC=%.4f | test_hit=%.4f",
                entry.score,
                entry.final_val_metrics.get("final_val_mean_ic", float("nan")),
                entry.test_metrics.get("test_mean_ic", float("nan")),
                entry.test_metrics.get("test_hit_rate", float("nan")),
            )
        except Exception as exc:
            logger.warning("Final eval failed for entry: %s", exc)


def _prediction_metrics(
    pred: pd.Series,
    labels: pd.Series,
    index: pd.Index,
    prefix: str,
) -> dict:
    ic = _ic_per_date(pred, labels, index)
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    icir = (
        mean_ic / (float(ic.std()) + 1e-9)
        if len(ic) > 1
        else float("nan")
    )
    hit = _hit_rate(pred, labels, index)
    return {
        f"{prefix}_mean_ic": mean_ic,
        f"{prefix}_icir": icir,
        f"{prefix}_hit_rate": hit,
    }


def _print_summary(archive: Archive, total_iters: int, elapsed: float) -> None:
    rows = archive.summary()
    print(f"\n{'='*100}")
    print(f"Evo_Finance - Final Archive  ({total_iters} iterations, {elapsed:.1f}s)")
    print(f"{'='*100}")
    header = f"{'Rank':>4}  {'Score':>7}  {'WF_IC':>7}  {'WF_IR':>7}  "
    header += f"{'HitEx':>7}  {'Std':>6}  {'Bad':>5}  {'Gap':>6}  "
    header += f"{'FValIC':>7}  {'TestIC':>7}  {'nGenes':>6}"
    print(header)
    print("-" * 100)
    for r in rows:
        tm = r.get("test_metrics", {})
        fvm = r.get("final_val_metrics", {})
        print(
            f"{r['rank']:>4}  {r['score']:>7.4f}  "
            f"{_fmt_metric(r.get('wf_mean_ic')):>7}  "
            f"{_fmt_metric(r.get('wf_icir')):>7}  "
            f"{_fmt_metric(r.get('wf_hit_excess')):>7}  "
            f"{_fmt_metric(r.get('wf_ic_std')):>6}  "
            f"{_fmt_metric(r.get('bad_fold_ratio')):>5}  "
            f"{_fmt_metric(r.get('wf_overfit_gap')):>6}  "
            f"{_fmt_metric(fvm.get('final_val_mean_ic')):>7}  "
            f"{tm.get('test_mean_ic', float('nan')):>7.4f}  "
            f"{r['n_genes']:>6}"
        )
    print(f"{'='*100}\n")
    return
    print(f"\n{'='*70}")
    print(f"Evo_Finance — Final Archive  ({total_iters} iterations, {elapsed:.1f}s)")
    print(f"{'='*70}")
    header = f"{'Rank':>4}  {'Score':>7}  {'ValIC':>6}  {'ICIR':>6}  "
    header += f"{'Hit':>5}  {'Gap':>5}  {'TestIC':>7}  {'nGenes':>6}"
    print(header)
    print("-" * 70)
    for r in rows:
        tm = r.get("test_metrics", {})
        print(
            f"{r['rank']:>4}  {r['score']:>7.4f}  "
            f"{r['val_mean_ic']:>6.4f}  {r['val_icir']:>6.4f}  "
            f"{r['hit_rate']:>5.4f}  {r['overfit_gap']:>5.4f}  "
            f"{tm.get('test_mean_ic', float('nan')):>7.4f}  "
            f"{r['n_genes']:>6}"
        )
    print(f"{'='*70}\n")


def _fmt_metric(value) -> str:
    if value is None:
        return "nan"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "nan"


def _save_json(archive: Archive, path: Path) -> None:
    rows = archive.summary()
    # make serialisable
    for r in rows:
        r.pop("val_ic_series", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info("Archive saved to %s", path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _load_data(path: str) -> pd.DataFrame:
    """Load a single parquet/csv file with MultiIndex (date, ticker)."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".gz"):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    if not isinstance(df.index, pd.MultiIndex):
        date_col   = next((c for c in df.columns if "date" in c.lower()), None)
        ticker_col = next(
            (c for c in df.columns if c.lower() in ("ticker", "symbol", "stock")),
            None,
        )
        if date_col is None or ticker_col is None:
            raise ValueError(
                "Cannot infer date/ticker columns. "
                "Please pass a DataFrame with MultiIndex (date, ticker)."
            )
        df[date_col]   = pd.to_datetime(df[date_col])
        df             = df.set_index([date_col, ticker_col])
        df.index.names = ["date", "ticker"]

    return df.sort_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evo_Finance — evolutionary feature selection cho mô hình LightGBM lambdarank",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Nguồn dữ liệu (chọn 1 trong 2) ──────────────────────────────────────
    data_grp = parser.add_mutually_exclusive_group(required=True)
    data_grp.add_argument(
        "--data-dir",
        metavar="DIR",
        help=(
            "Thư mục chứa các file <TICKER>.csv sinh ra bởi craw_data.py\n"
            "Ví dụ: --data-dir data/raw"
        ),
    )
    data_grp.add_argument(
        "--data",
        metavar="FILE",
        help=(
            "File parquet/csv duy nhất đã có MultiIndex (date, ticker)\n"
            "Ví dụ: --data data/all_stocks.parquet"
        ),
    )

    # ── Tùy chọn ticker (chỉ dùng với --data-dir) ───────────────────────────
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help=(
            "Chỉ load một số ticker nhất định (mặc định: tất cả file trong thư mục)\n"
            "Ví dụ: --tickers ACB VCB TCB HPG"
        ),
    )

    # ── Tham số chạy ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--budget",
        type=float,
        default=TIME_BUDGET_SECONDS,
        metavar="SECONDS",
        help=f"Ngân sách thời gian chạy (giây). Mặc định: {TIME_BUDGET_SECONDS}",
    )
    parser.add_argument(
        "--restart",
        type=float,
        default=RESTART_PROB,
        metavar="PROB",
        help=f"Xác suất restart về OHLCV gốc. Mặc định: {RESTART_PROB}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Mặc định: 42",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Lưu kết quả archive ra file JSON. Ví dụ: --save results/archive.json",
    )

    # ── Ngày split ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--val-start",
        type=str,
        default=VAL_START,
        metavar="DATE",
        help=(
            "Ngày đầu tiên của val = kết thúc train (YYYY-MM-DD)\n"
            f"Mặc định: {VAL_START}"
        ),
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=TEST_START,
        metavar="DATE",
        help=(
            "Ngày đầu tiên của test = kết thúc val (YYYY-MM-DD)\n"
            f"Mặc định: {TEST_START}"
        ),
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=TEST_END,
        metavar="DATE",
        help=(
            "Ngày cuối test (YYYY-MM-DD). None = lấy hết data còn lại\n"
            f"Mặc định: {TEST_END or 'hết data'}"
        ),
    )

    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data_dir:
        from data.loader import load_from_dir
        df = load_from_dir(args.data_dir, tickers=args.tickers)
    else:
        df = _load_data(args.data)

    save_path = Path(args.save) if args.save else None
    run(
        df           = df,
        time_budget  = args.budget,
        restart_prob = args.restart,
        seed         = args.seed,
        save_archive = save_path,
        val_start    = args.val_start,
        test_start   = args.test_start,
        test_end     = args.test_end,
    )
