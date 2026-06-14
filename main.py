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
)
from mutator.gene       import Individual
from mutator.domain     import Domain
from mutator.mutator    import Mutator
from model.trainer      import Trainer
from model.data_utils   import split_and_label, validate_ohlcv
from fitness.fitness    import FitnessEvaluator
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
    """
    validate_ohlcv(df)
    rng = np.random.default_rng(seed)

    # ── Data split + labels ───────────────────────────────────────────────────
    logger.info("Splitting data into train / val / test …")
    train_df, val_df, test_df = split_and_label(df)
    logger.info(
        "Sizes — train: %d rows, val: %d rows, test: %d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Initialise components ─────────────────────────────────────────────────
    domain    = Domain()
    seed_genes = domain.seed(window=DEFAULT_WINDOW)
    domain.precompute(train_df)

    mutator   = Mutator(domain=domain, seed=int(rng.integers(1 << 31)))
    trainer   = Trainer()
    evaluator = FitnessEvaluator()
    archive   = Archive()

    # ── First individual: raw OHLCV ───────────────────────────────────────────
    logger.info("Evaluating seed individual (raw OHLCV) …")
    seed_ind = Individual.seed(window=DEFAULT_WINDOW)
    _evaluate_and_archive(
        seed_ind, trainer, evaluator, archive,
        train_df, val_df,
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
            _evaluate_and_archive(parent, trainer, evaluator, archive, train_df, val_df)
            continue
        else:
            parent = archive.random_individual(rng)

        # mutate
        try:
            child = mutator.mutate(parent, train_df)
        except Exception as exc:
            logger.warning("Mutation failed: %s — skipping iteration.", exc)
            continue

        if len(child) == 0:
            logger.warning("Mutation produced empty individual — skipping.")
            continue

        # evaluate + archive
        admitted = _evaluate_and_archive(
            child, trainer, evaluator, archive,
            train_df, val_df,
        )

        best_score = archive.best.score if archive.best else float("nan")
        logger.info(
            "Result: score=%.4f | admitted=%s | archive_size=%d | best=%.4f",
            child.score or float("nan"), admitted, len(archive), best_score,
        )

    # ── Test-set evaluation for all archived individuals ──────────────────────
    logger.info("=== Time budget exhausted. Running test-set evaluation … ===")
    _test_evaluate_archive(archive, trainer, evaluator, test_df)

    # ── Output ────────────────────────────────────────────────────────────────
    _print_summary(archive, iteration, time.time() - t_start)

    if save_archive is not None:
        _save_json(archive, save_archive)

    return archive


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _evaluate_and_archive(
    individual: Individual,
    trainer:    Trainer,
    evaluator:  FitnessEvaluator,
    archive:    Archive,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
) -> bool:
    """Train, score, and optionally archive an individual. Returns admission bool."""
    try:
        booster, train_pred, val_pred, train_labels, val_labels = trainer.train(
            individual, train_df, val_df
        )
    except Exception as exc:
        logger.warning("Training failed: %s", exc)
        return False

    try:
        evaluator.evaluate(
            individual,
            train_pred, val_pred,
            train_labels, val_labels,
            train_df, val_df,
        )
    except Exception as exc:
        logger.warning("Fitness evaluation failed: %s", exc)
        return False

    return archive.try_add(individual, booster)


def _test_evaluate_archive(
    archive:   Archive,
    trainer:   Trainer,
    evaluator: FitnessEvaluator,
    test_df:   pd.DataFrame,
) -> None:
    """Run test-set predictions for every archived individual."""
    for entry in archive.entries:
        try:
            test_pred = trainer.predict(entry.booster, entry.individual, test_df)
            test_labels = test_df["label"]

            # reuse fitness helpers for test metrics
            from fitness.fitness import _ic_per_date, _hit_rate
            test_ic = _ic_per_date(test_pred, test_labels, test_df.index)
            test_mean_ic = float(test_ic.mean()) if len(test_ic) else float("nan")
            test_icir    = (
                test_mean_ic / (float(test_ic.std()) + 1e-9)
                if len(test_ic) > 1
                else float("nan")
            )
            test_hit = _hit_rate(test_pred, test_labels, test_df.index)

            entry.test_metrics = {
                "test_mean_ic": test_mean_ic,
                "test_icir":    test_icir,
                "test_hit_rate":test_hit,
            }
            logger.info(
                "Test eval: score=%.4f | IC=%.4f | ICIR=%.4f | hit=%.4f",
                entry.score, test_mean_ic, test_icir, test_hit,
            )
        except Exception as exc:
            logger.warning("Test eval failed for entry: %s", exc)


def _print_summary(archive: Archive, total_iters: int, elapsed: float) -> None:
    rows = archive.summary()
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
    )