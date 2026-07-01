"""Run crypto feature evolution.

Example:
    python -m crypto.main --data data/crypto/BTCUSDT_15m.csv --budget 3600 --seed 1
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from crypto import config
from crypto.data import (
    add_binary_labels,
    load_ohlcv,
    make_walk_forward_folds,
    split_labeled_by_dates,
)
from crypto.evolution import CryptoArchive, CryptoIndividual, CryptoMutator
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.fitness import CryptoFitnessEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.main")


def run(
    data_path: str | Path = config.DATA_PATH,
    time_budget: float = config.TIME_BUDGET_SECONDS,
    seed: int = 42,
    save_archive: str | Path | None = config.DEFAULT_ARCHIVE_PATH,
    resume_archive: str | Path | None = None,
    horizons: list[int] | tuple[int, ...] = tuple(config.HOLDING_HORIZONS),
    label_threshold: float = config.LABEL_THRESHOLD,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    wf_end: str | None = None,
    wf_min_train_months: int = config.WF_MIN_TRAIN_MONTHS,
    wf_val_months: int = config.WF_VAL_MONTHS,
    wf_step_months: int = config.WF_STEP_MONTHS,
    checkpoint_every: float = config.CHECKPOINT_EVERY_SECONDS,
) -> CryptoArchive:
    config.validate_config()
    horizons = [int(h) for h in horizons]
    purge_bars = config.purge_bars_for_horizons(horizons)
    wf_end = wf_end or test_start
    rng = np.random.default_rng(seed)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    labeled_df = add_binary_labels(raw_df, horizons=horizons, threshold=label_threshold)
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge_bars,
    )
    logger.info(
        "Final split: train=%d | val=%d | test=%d | purge=%d bars",
        len(train_df),
        len(val_df),
        len(test_df),
        purge_bars,
    )

    wf_raw_df = labeled_df[labeled_df.index < pd.Timestamp(wf_end)].copy()
    folds = make_walk_forward_folds(
        wf_raw_df,
        wf_end=wf_end,
        min_train_months=wf_min_train_months,
        val_months=wf_val_months,
        step_months=wf_step_months,
        purge_bars=purge_bars,
    )
    for fold in folds:
        logger.info(
            "WF %s: train [%s -> %s) %d | val [%s -> %s) %d",
            fold.name,
            fold.train_start,
            fold.train_end,
            len(fold.train_df),
            fold.val_start,
            fold.val_end,
            len(fold.val_df),
        )

    feature_quality_index = folds[0].train_df.index
    logger.info(
        "Building safe crypto feature matrix; quality filter uses %d first-fold train rows.",
        len(feature_quality_index),
    )
    feature_df = build_feature_frame(raw_df, quality_index=feature_quality_index)
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)
    logger.info("Feature pool: %d safe features.", len(feature_pool))
    if len(feature_pool) < config.FEATURE_MIN:
        raise ValueError("Feature pool is smaller than FEATURE_MIN.")

    mutator = CryptoMutator(
        feature_pool=feature_pool,
        feature_space=feature_space,
        train_index=feature_quality_index,
        seed=int(rng.integers(1 << 31)),
    )
    evaluator = CryptoFitnessEvaluator(horizons=horizons)
    archive = (
        CryptoArchive.load(resume_archive)
        if resume_archive is not None
        else CryptoArchive()
    )

    if archive.is_empty():
        logger.info("Evaluating seed crypto individual ...")
        seed_individual = mutator.seed_individual()
        evaluator.evaluate_walk_forward(seed_individual, folds, feature_space)
        archive.try_add(seed_individual)
    else:
        logger.info("Loaded resume archive with %d entries.", len(archive))

    save_path = Path(save_archive) if save_archive else None
    checkpoint_path = _checkpoint_path(save_path)
    start_time = time.time()
    last_checkpoint = start_time
    iteration = 0

    while time.time() - start_time < float(time_budget):
        iteration += 1
        elapsed = time.time() - start_time
        logger.info("Iteration %d (%.1fs / %.1fs)", iteration, elapsed, time_budget)

        if rng.random() < config.RESTART_PROB or archive.is_empty():
            parent = mutator.seed_individual()
        else:
            parent = archive.random_individual(rng)

        child = mutator.mutate(parent)
        if _signature(child.features) == _signature(parent.features):
            logger.info("Mutation unchanged; skip.")
            continue

        try:
            evaluator.evaluate_walk_forward(child, folds, feature_space)
        except Exception as exc:
            logger.warning("Evaluation failed: %s", exc)
            continue

        admitted = archive.try_add(child)
        best = archive.best.score if archive.best else float("nan")
        logger.info(
            "Result: score=%.4f | admitted=%s | archive=%d | best=%.4f",
            child.score,
            admitted,
            len(archive),
            best,
        )

        if checkpoint_path is not None and checkpoint_every > 0:
            now = time.time()
            if now - last_checkpoint >= float(checkpoint_every):
                _save_archive(archive, checkpoint_path, horizons, label_threshold)
                last_checkpoint = now

    if save_path is not None:
        _save_archive(archive, save_path, horizons, label_threshold)
        logger.info("Saved crypto archive to %s", save_path)
    if checkpoint_path is not None:
        _save_archive(archive, checkpoint_path, horizons, label_threshold)
    return archive


def _save_archive(
    archive: CryptoArchive,
    path: Path,
    horizons: list[int],
    label_threshold: float,
) -> None:
    archive.save(
        path,
        metadata={
            "pipeline": "crypto",
            "horizons": horizons,
            "label_threshold": label_threshold,
            "fitness": config.FITNESS_WEIGHTS,
            "trade_top_fraction": config.TRADE_TOP_FRACTION,
            "return_score_scale": config.RETURN_SCORE_SCALE,
        },
    )


def _checkpoint_path(save_path: Path | None) -> Path | None:
    if save_path is None:
        return None
    return save_path.with_name(save_path.stem + ".checkpoint" + save_path.suffix)


def _signature(features: list[str]) -> tuple[str, ...]:
    return tuple(sorted(features))


def _parse_horizons(text: str) -> list[int]:
    horizons = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not horizons:
        raise argparse.ArgumentTypeError("horizons must not be empty.")
    if any(h < 1 for h in horizons):
        raise argparse.ArgumentTypeError("all horizons must be positive.")
    return horizons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path.")
    parser.add_argument("--budget", type=float, default=config.TIME_BUDGET_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save",
        default=str(config.DEFAULT_ARCHIVE_PATH),
        help=f"Output archive JSON path. Default: {config.DEFAULT_ARCHIVE_PATH}",
    )
    parser.add_argument("--resume", default=None, help="Resume from a crypto archive JSON.")
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=list(config.HOLDING_HORIZONS),
        help="Comma-separated horizons, default: 3,7,10.",
    )
    parser.add_argument("--label-threshold", type=float, default=config.LABEL_THRESHOLD)
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--wf-end", default=None)
    parser.add_argument("--wf-min-train-months", type=int, default=config.WF_MIN_TRAIN_MONTHS)
    parser.add_argument("--wf-val-months", type=int, default=config.WF_VAL_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=config.WF_STEP_MONTHS)
    parser.add_argument("--checkpoint-every", type=float, default=config.CHECKPOINT_EVERY_SECONDS)
    args = parser.parse_args()

    run(
        data_path=args.data,
        time_budget=args.budget,
        seed=args.seed,
        save_archive=args.save,
        resume_archive=args.resume,
        horizons=args.horizons,
        label_threshold=args.label_threshold,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        wf_end=args.wf_end,
        wf_min_train_months=args.wf_min_train_months,
        wf_val_months=args.wf_val_months,
        wf_step_months=args.wf_step_months,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
