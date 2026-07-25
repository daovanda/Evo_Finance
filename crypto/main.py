"""Run crypto feature evolution.

PowerShell, start a new 12-hour payoff run:
    python -m crypto.main `
      --data data/crypto/BTCUSDT_15m.csv `
      --budget 43200 `
      --seed 1 `
      --horizons 5 `
      --label-mode payoff `
      --label-direction Long `
      --label-threshold 0.002 `
      --trade-top-fraction 0.10 `
      --save crypto/results/crypto_btc_long_payoff_h5_seed1_12h.json `
      --checkpoint-every 3600

PowerShell, continue an archive for another 12 hours with a new RNG seed:
    python -m crypto.main `
      --data data/crypto/BTCUSDT_15m.csv `
      --budget 43200 `
      --seed 2 `
      --horizons 5 `
      --label-mode payoff `
      --label-direction Long `
      --label-threshold 0.002 `
      --trade-top-fraction 0.10 `
      --resume crypto/results/crypto_btc_long_payoff_h5_seed1_12h.json `
      --save crypto/results/crypto_btc_long_payoff_h5_seed1_24h.json `
      --checkpoint-every 3600

Bash/VM, start the same run and save a log:
    python -m crypto.main \
      --data data/crypto/BTCUSDT_15m.csv \
      --budget 43200 \
      --seed 1 \
      --horizons 5 \
      --label-mode payoff \
      --label-direction Long \
      --label-threshold 0.002 \
      --trade-top-fraction 0.10 \
      --save crypto/results/crypto_btc_long_payoff_h5_seed1_12h.json \
      --checkpoint-every 3600 \
      2>&1 | tee crypto/results/run_crypto_btc_long_payoff_h5_seed1_12h.log

Mode-specific constants such as PAYOFF_TP, PAYOFF_ADVERSE_FLOOR, TP_SAFE_PATH,
TRADE_COST, and the
fitness weights are read from crypto/config.py. Resume metadata must match the
effective CLI and config values.
"""

from __future__ import annotations

import argparse
import gc
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
    label_threshold: float | None = None,
    label_mode: str = config.LABEL_MODE,
    label_direction: str = config.LABEL_DIRECTION,
    trade_top_fraction: float | None = None,
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
    label_mode = config.canonical_label_mode(label_mode)
    label_direction = config.canonical_label_direction(label_direction)
    label_threshold = config.default_label_threshold(label_mode, label_threshold)
    if not np.isfinite(float(label_threshold)):
        raise ValueError("label_threshold must be finite.")
    if label_mode == "adverse_floor" and float(label_threshold) <= 0.0:
        raise ValueError(
            "adverse_floor label_threshold must be positive; pass for example "
            "--label-threshold 0.003."
        )
    if label_mode == "slope_slowdown" and float(label_threshold) <= 0.0:
        raise ValueError(
            "slope_slowdown label_threshold must be positive; pass for example "
            "--label-threshold 0.0003."
        )
    if trade_top_fraction is not None:
        trade_top_fraction = float(trade_top_fraction)
        if not np.isfinite(trade_top_fraction) or not 0.0 < trade_top_fraction <= 1.0:
            raise ValueError("trade_top_fraction must be finite and in (0, 1].")
        config.TRADE_TOP_FRACTION = trade_top_fraction
    label_return_fn = config.get_label_return_fn(label_mode)
    purge_bars = config.purge_bars_for_horizons(horizons)
    wf_end = wf_end or test_start
    rng = np.random.default_rng(seed)

    logger.info(
        "Run labels: mode=%s | direction=%s | threshold=%g | trade_top_fraction=%.2f%%",
        label_mode,
        label_direction,
        label_threshold,
        100.0 * float(config.TRADE_TOP_FRACTION),
    )
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    labeled_df = add_binary_labels(
        raw_df,
        horizons=horizons,
        threshold=label_threshold,
        return_fn=label_return_fn,
        label_mode=label_mode,
        label_direction=label_direction,
    )
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
    evaluator = CryptoFitnessEvaluator(
        horizons=horizons,
        precision_only=config.is_precision_only_label_mode(label_mode),
    )
    archive = (
        CryptoArchive.load(resume_archive)
        if resume_archive is not None
        else CryptoArchive()
    )
    if resume_archive is not None:
        _validate_resume_metadata(
            archive=archive,
            resume_path=Path(resume_archive),
            horizons=horizons,
            label_mode=label_mode,
            label_direction=label_direction,
            label_threshold=label_threshold,
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

        if config.EVOLUTION_GC_EVERY > 0 and iteration % config.EVOLUTION_GC_EVERY == 0:
            collected = gc.collect()
            logger.info(
                "Memory cleanup: expression_cache=%d/%d | collected=%d",
                feature_space.cache_size,
                config.EXPR_CACHE_MAX_ITEMS,
                collected,
            )

        if checkpoint_path is not None and checkpoint_every > 0:
            now = time.time()
            if now - last_checkpoint >= float(checkpoint_every):
                _save_archive(
                    archive,
                    checkpoint_path,
                    horizons,
                    label_threshold,
                    label_mode,
                    label_direction,
                )
                last_checkpoint = now

    if not archive.is_empty():
        _evaluate_final_archive(
            archive=archive,
            evaluator=evaluator,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_space=feature_space,
        )

    if save_path is not None:
        _save_archive(
            archive, save_path, horizons, label_threshold, label_mode, label_direction
        )
        logger.info("Saved crypto archive to %s", save_path)
    if checkpoint_path is not None:
        _save_archive(
            archive,
            checkpoint_path,
            horizons,
            label_threshold,
            label_mode,
            label_direction,
        )
    return archive


def _evaluate_final_archive(
    archive: CryptoArchive,
    evaluator: CryptoFitnessEvaluator,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
) -> None:
    logger.info(
        "Budget ended; running final val/test evaluation for %d archive entries.",
        len(archive),
    )
    ok = 0
    for rank, individual in enumerate(archive.entries, start=1):
        logger.info("Final evaluation rank %d/%d started.", rank, len(archive))
        try:
            evaluator.evaluate_final(
                individual=individual,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_data=feature_space,
            )
        except Exception as exc:
            logger.warning("Final evaluation failed for rank %d: %s", rank, exc)
            continue
        ok += 1
    logger.info(
        "Final evaluation completed: %d/%d entries evaluated.", ok, len(archive)
    )


def _save_archive(
    archive: CryptoArchive,
    path: Path,
    horizons: list[int],
    label_threshold: float,
    label_mode: str,
    label_direction: str,
) -> None:
    archive.save(
        path,
        metadata={
            "pipeline": "crypto",
            "horizons": horizons,
            "label_mode": label_mode,
            "label_direction": label_direction,
            "direction_neutral": config.is_direction_neutral_label_mode(label_mode),
            "label_threshold": label_threshold,
            "payoff_tp": config.PAYOFF_TP,
            "payoff_adverse_floor": config.PAYOFF_ADVERSE_FLOOR,
            "tp_safe_path": config.TP_SAFE_PATH,
            "safe_adverse_floor": (
                float(label_threshold)
                if label_mode == "safe_path_mfe"
                else float(config.SAFE_ADVERSE_FLOOR)
            ),
            "safe_path_rule": config.SAFE_PATH_RULE,
            "slope_lookback": config.SLOPE_LOOKBACK,
            "slope_min_initial": config.SLOPE_MIN_INITIAL,
            "slope_price_column": config.SLOPE_PRICE_COLUMN,
            "slope_slowdown_rule": config.SLOPE_SLOWDOWN_RULE,
            "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
            "fitness": config.FITNESS_WEIGHTS,
            "trade_top_fraction": config.TRADE_TOP_FRACTION,
            "trade_cost": config.TRADE_COST,
            "return_score_scale": config.RETURN_SCORE_SCALE,
            "precision_only": config.is_precision_only_label_mode(label_mode),
        },
    )


def _validate_resume_metadata(
    archive: CryptoArchive,
    resume_path: Path,
    horizons: list[int],
    label_mode: str,
    label_direction: str,
    label_threshold: float,
) -> None:
    metadata = getattr(archive, "metadata", {}) or {}
    if not metadata:
        logger.warning(
            "Resume archive %s has no metadata; cannot verify label/fitness compatibility.",
            resume_path,
        )
        return

    metadata_label_mode = metadata.get("label_mode", "")
    archive_label_mode = (
        config.canonical_label_mode(metadata_label_mode)
        if metadata_label_mode not in (None, "")
        else ""
    )
    metadata_label_direction = metadata.get("label_direction")
    # Archives created before direction support were all long-only.
    archive_label_direction = config.canonical_label_direction(
        metadata_label_direction
        if metadata_label_direction not in (None, "")
        else "long"
    )
    checks: list[tuple[str, object, object]] = [
        (
            "horizons",
            [int(h) for h in metadata.get("horizons", [])],
            [int(h) for h in horizons],
        ),
        ("label_mode", archive_label_mode, label_mode),
        ("label_threshold", metadata.get("label_threshold"), float(label_threshold)),
        (
            "fitness_horizon_mode",
            str(metadata.get("fitness_horizon_mode", "")).strip().lower(),
            config.FITNESS_HORIZON_MODE,
        ),
        (
            "trade_top_fraction",
            metadata.get("trade_top_fraction"),
            float(config.TRADE_TOP_FRACTION),
        ),
        ("trade_cost", metadata.get("trade_cost"), float(config.TRADE_COST)),
    ]
    if not (
        config.is_direction_neutral_label_mode(label_mode)
        and config.is_direction_neutral_label_mode(archive_label_mode)
    ):
        checks.append(("label_direction", archive_label_direction, label_direction))
    if label_mode == "payoff" and archive_label_mode == "payoff":
        checks.append(("payoff_tp", metadata.get("payoff_tp"), float(config.PAYOFF_TP)))
        checks.append(
            (
                "payoff_adverse_floor",
                metadata.get("payoff_adverse_floor"),
                float(config.PAYOFF_ADVERSE_FLOOR),
            )
        )
    if label_mode == "safe_path_mfe" and archive_label_mode == "safe_path_mfe":
        archive_rule = metadata.get("safe_path_rule")
        if archive_rule != config.SAFE_PATH_RULE:
            raise ValueError(
                "Resume archive uses an incompatible safe_path_mfe rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.SAFE_PATH_RULE!r}. Start a new archive."
            )
        checks.append(
            ("tp_safe_path", metadata.get("tp_safe_path"), float(config.TP_SAFE_PATH))
        )
    if label_mode == "slope_slowdown" and archive_label_mode == "slope_slowdown":
        archive_rule = metadata.get("slope_slowdown_rule")
        if archive_rule != config.SLOPE_SLOWDOWN_RULE:
            raise ValueError(
                "Resume archive uses an incompatible slope_slowdown rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.SLOPE_SLOWDOWN_RULE!r}. "
                "Start a new archive."
            )
        checks.extend(
            [
                (
                    "slope_lookback",
                    metadata.get("slope_lookback"),
                    int(config.SLOPE_LOOKBACK),
                ),
                (
                    "slope_min_initial",
                    metadata.get("slope_min_initial"),
                    float(config.SLOPE_MIN_INITIAL),
                ),
                (
                    "slope_price_column",
                    metadata.get("slope_price_column"),
                    config.SLOPE_PRICE_COLUMN,
                ),
            ]
        )

    mismatches: list[str] = []
    for name, archive_value, current_value in checks:
        if archive_value in (None, "", []):
            logger.warning(
                "Resume archive %s metadata is missing %s; cannot verify that field.",
                resume_path,
                name,
            )
            continue
        if isinstance(current_value, float):
            try:
                archive_float = float(archive_value)
            except (TypeError, ValueError):
                mismatches.append(
                    f"{name}: archive={archive_value!r}, current={current_value!r}"
                )
                continue
            if not np.isclose(archive_float, current_value, rtol=0.0, atol=1e-12):
                mismatches.append(
                    f"{name}: archive={archive_float!r}, current={current_value!r}"
                )
        elif archive_value != current_value:
            mismatches.append(
                f"{name}: archive={archive_value!r}, current={current_value!r}"
            )

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "Resume archive config does not match current run. "
            f"Archive={resume_path}. Mismatches: {joined}. "
            "Use matching --label-mode/--label-threshold/config values, or start a new archive."
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
    parser.add_argument(
        "--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path."
    )
    parser.add_argument("--budget", type=float, default=config.TIME_BUDGET_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save",
        default=str(config.DEFAULT_ARCHIVE_PATH),
        help=f"Output archive JSON path. Default: {config.DEFAULT_ARCHIVE_PATH}",
    )
    parser.add_argument(
        "--resume", default=None, help="Resume from a crypto archive JSON."
    )
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=list(config.HOLDING_HORIZONS),
        help=(
            "Comma-separated horizons. Default: "
            f"{','.join(str(h) for h in config.HOLDING_HORIZONS)}."
        ),
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help=(
            "Label threshold. Default is LABEL_THRESHOLD for ordinary modes, "
            "TRADE_COST for payoff, and SAFE_ADVERSE_FLOOR for safe_path_mfe. "
            "For safe_path_mfe this is the stop-first adverse low/high floor; "
            "TP is config.TP_SAFE_PATH. For adverse_floor use a positive "
            "distance, for example 0.003 means the path must stay above -0.003. "
            "For high_exit this is the directional return threshold of the "
            "exact H candle high (Long) or low (Short). For slope_slowdown "
            "this is the minimum high-price OLS slope change per candle; its "
            "mode default is config.SLOPE_SLOWDOWN_THRESHOLD. For "
            "two_sided_tp it is the positive absolute TP used on both sides."
        ),
    )
    parser.add_argument(
        "--label-mode",
        default=config.LABEL_MODE,
        help=(
            f"Label mode. Allowed: {', '.join(sorted(config.LABEL_RETURN_FNS))}. "
            "Aliases accepted: exit_all -> exit_after_h1, "
            "first_hit_safe_close/safe_close -> safe_path_mfe. "
            f"Default: {config.LABEL_MODE}."
        ),
    )
    parser.add_argument(
        "--label-direction",
        default=config.LABEL_DIRECTION,
        help=(
            "Label direction. Long means price up is favorable; Short means "
            "price down is favorable. It is ignored by two_sided_tp. "
            f"Default: {config.LABEL_DIRECTION}."
        ),
    )
    parser.add_argument(
        "--trade-top-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of highest predictions selected for fitness/trading. "
            f"Default: config.TRADE_TOP_FRACTION={config.TRADE_TOP_FRACTION}."
        ),
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--wf-end", default=None)
    parser.add_argument(
        "--wf-min-train-months", type=int, default=config.WF_MIN_TRAIN_MONTHS
    )
    parser.add_argument("--wf-val-months", type=int, default=config.WF_VAL_MONTHS)
    parser.add_argument("--wf-step-months", type=int, default=config.WF_STEP_MONTHS)
    parser.add_argument(
        "--checkpoint-every", type=float, default=config.CHECKPOINT_EVERY_SECONDS
    )
    args = parser.parse_args()

    run(
        data_path=args.data,
        time_budget=args.budget,
        seed=args.seed,
        save_archive=args.save,
        resume_archive=args.resume,
        horizons=args.horizons,
        label_threshold=args.label_threshold,
        label_mode=args.label_mode,
        label_direction=args.label_direction,
        trade_top_fraction=args.trade_top_fraction,
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
