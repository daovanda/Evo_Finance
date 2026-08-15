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

PowerShell, evolve an OOF dynamic-TP meta learner from MFE Q20 Rank 1:
    python -m crypto.main `
      --data data/crypto/BTCUSDT_5m.csv `
      --budget 3600 `
      --seed 1 `
      --horizons 3 `
      --label-mode meta_learner `
      --label-direction Long `
      --meta-base-archive crypto/results/crypto_btc_5m_quantile_mfe_q20_h3_seed1_1h.json `
      --meta-base-rank 1 `
      --meta-feature-data data/crypto/BTCUSDT_1m.csv `
      --meta-feature-lookahead-bars 1 `
      --meta-min-prediction 0.0002 `
      --meta-val-fraction 0.20 `
      --trade-top-fraction 0.20 `
      --save crypto/results/crypto_btc_5m_meta_mfe_q20_h3_seed1_1h.json `
      --checkpoint-every 0

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
from crypto.meta_targets import (
    align_meta_feature_frame,
    build_meta_feature_alignment,
    build_meta_learner_data,
    build_observed_mfe,
    build_post_observation_mfe,
    load_meta_base,
    required_feature_windows,
)
from crypto.quantile_fitness import QuantileFitnessEvaluator


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
    quantile_target: str = config.QUANTILE_TARGET,
    quantile: float = config.QUANTILE_ALPHA,
    meta_base_archive: str | Path = config.META_LEARNER_BASE_ARCHIVE,
    meta_base_rank: int = config.META_LEARNER_BASE_RANK,
    meta_min_prediction: float = config.META_LEARNER_MIN_PREDICTION,
    meta_tp_offset: float = config.META_LEARNER_TP_OFFSET,
    meta_val_fraction: float = config.META_LEARNER_META_VAL_FRACTION,
    meta_feature_data: str | Path | None = config.META_LEARNER_FEATURE_DATA,
    meta_feature_lookahead_bars: int = config.META_LEARNER_FEATURE_LOOKAHEAD_BARS,
    meta_feature_include_h1: bool = config.META_LEARNER_FEATURE_INCLUDE_H1,
    exit_after_k: int | None = None,
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
    quantile_target = config.canonical_quantile_target(quantile_target)
    quantile = config.validate_quantile_alpha(quantile)
    if label_mode == "quantile_trade":
        config.QUANTILE_TARGET = quantile_target
        config.QUANTILE_ALPHA = quantile
    if config.is_meta_learner_label_mode(label_mode):
        config.META_LEARNER_BASE_ARCHIVE = Path(meta_base_archive)
        config.META_LEARNER_BASE_RANK = int(meta_base_rank)
        config.META_LEARNER_MIN_PREDICTION = float(meta_min_prediction)
        config.META_LEARNER_TP_OFFSET = float(meta_tp_offset)
        config.META_LEARNER_META_VAL_FRACTION = float(meta_val_fraction)
        config.META_LEARNER_FEATURE_DATA = (
            Path(meta_feature_data) if meta_feature_data not in (None, "") else None
        )
        config.META_LEARNER_FEATURE_LOOKAHEAD_BARS = int(
            meta_feature_lookahead_bars
        )
        config.META_LEARNER_FEATURE_INCLUDE_H1 = bool(meta_feature_include_h1)
        config.META_LEARNER_TARGET_START_STEP = (
            2 if config.META_LEARNER_FEATURE_INCLUDE_H1 else 1
        )
        config.META_LEARNER_FEATURE_ALIGNMENT_RULE = (
            "target_open_plus_two_target_intervals_minus_feature_interval_v1"
            if config.META_LEARNER_FEATURE_INCLUDE_H1
            else (
                "target_open_plus_lower_timeframe_lookahead_v1"
                if config.META_LEARNER_FEATURE_LOOKAHEAD_BARS
                else "target_open_plus_target_interval_minus_feature_interval_v1"
            )
        )
        config.META_LEARNER_TARGET_INTERVAL_SECONDS = None
        config.META_LEARNER_FEATURE_INTERVAL_SECONDS = None
        if int(meta_base_rank) < 1:
            raise ValueError("meta_base_rank must be positive.")
        if not np.isfinite(meta_min_prediction) or meta_min_prediction < 0.0:
            raise ValueError("meta_min_prediction must be finite and non-negative.")
        if not np.isfinite(meta_tp_offset) or meta_tp_offset < 0.0:
            raise ValueError("meta_tp_offset must be finite and non-negative.")
        if not 0.0 < meta_val_fraction < 0.5:
            raise ValueError("meta_val_fraction must be in (0, 0.5).")
        if label_direction != "long":
            raise ValueError(
                "Meta modes currently model Long MFE-based trades; "
                "use --label-direction Long."
            )
        if (
            (
                config.META_LEARNER_FEATURE_INCLUDE_H1
                or config.META_LEARNER_FEATURE_LOOKAHEAD_BARS > 0
            )
            and config.META_LEARNER_FEATURE_DATA is None
        ):
            raise ValueError(
                "Meta feature lookahead requires --meta-feature-data."
            )
        if config.META_LEARNER_FEATURE_LOOKAHEAD_BARS < 0:
            raise ValueError("--meta-feature-lookahead-bars must be non-negative.")
        if (
            config.META_LEARNER_FEATURE_INCLUDE_H1
            and config.META_LEARNER_FEATURE_LOOKAHEAD_BARS
        ):
            raise ValueError(
                "Use either --meta-feature-include-h1 or "
                "--meta-feature-lookahead-bars, not both."
            )
        if config.META_LEARNER_FEATURE_INCLUDE_H1 and min(horizons) < 2:
            raise ValueError(
                "--meta-feature-include-h1 requires every horizon to be at least 2."
            )
    exit_after_k = config.resolve_exit_after_k(label_mode, exit_after_k)
    if exit_after_k is not None:
        config.EXIT_AFTER_K = int(exit_after_k)
    if (
        label_mode in {"ma_slope_reversal", "quantile_trade", "meta_learner"}
        and label_threshold is not None
    ):
        logger.warning(
            "--label-threshold is ignored by %s; using 0.0 metadata value.",
            label_mode,
        )
        label_threshold = 0.0
    else:
        label_threshold = config.default_label_threshold(label_mode, label_threshold)
    if not np.isfinite(float(label_threshold)):
        raise ValueError("label_threshold must be finite.")
    if label_mode == "adverse_floor" and float(label_threshold) <= 0.0:
        raise ValueError(
            "adverse_floor label_threshold must be positive; pass for example "
            "--label-threshold 0.003."
        )
    if (
        label_mode in {"slope_slowdown", "slope_slowdown_all"}
        and float(label_threshold) <= 0.0
    ):
        raise ValueError(
            f"{label_mode} label_threshold must be positive; pass for example "
            "--label-threshold 0.0003."
        )
    if trade_top_fraction is not None:
        trade_top_fraction = float(trade_top_fraction)
        if not np.isfinite(trade_top_fraction) or not 0.0 < trade_top_fraction <= 1.0:
            raise ValueError("trade_top_fraction must be finite and in (0, 1].")
        config.TRADE_TOP_FRACTION = trade_top_fraction
    purge_bars = config.purge_bars_for_horizons(horizons)
    # Keep final validation independent from evolutionary feature selection.
    # Callers can still opt into a later boundary explicitly with --wf-end.
    wf_end = wf_end or val_start
    rng = np.random.default_rng(seed)

    if label_mode == "quantile_trade":
        logger.info(
            "Run quantile target: mode=%s | target=%s | Q=%.2f | "
            "trading/threshold/top-fraction disabled",
            label_mode,
            quantile_target,
            quantile,
        )
    elif config.is_meta_learner_label_mode(label_mode):
        logger.info(
            "Run meta learner: mode=%s | base=%s rank=%d | min TP=%.4f%% | "
            "label threshold=%.4f%% | "
            "meta val=%.1f%% | feature_data=%s | lookahead_bars=%d | "
            "include_h1=%s | "
            "trade_top_fraction=%.2f%%",
            label_mode,
            config.META_LEARNER_BASE_ARCHIVE,
            config.META_LEARNER_BASE_RANK,
            100.0 * config.META_LEARNER_MIN_PREDICTION,
            100.0 * float(label_threshold),
            100.0 * config.META_LEARNER_META_VAL_FRACTION,
            config.META_LEARNER_FEATURE_DATA or data_path,
            config.META_LEARNER_FEATURE_LOOKAHEAD_BARS,
            config.META_LEARNER_FEATURE_INCLUDE_H1,
            100.0 * float(config.TRADE_TOP_FRACTION),
        )
    else:
        logger.info(
            "Run labels: mode=%s | direction=%s | threshold=%g | "
            "exit_after_k=%s | trade_top_fraction=%.2f%%",
            label_mode,
            label_direction,
            label_threshold,
            exit_after_k,
            100.0 * float(config.TRADE_TOP_FRACTION),
        )
    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    meta_base = None
    if config.is_meta_learner_label_mode(label_mode):
        meta_base = load_meta_base(
            config.META_LEARNER_BASE_ARCHIVE,
            config.META_LEARNER_BASE_RANK,
            horizons,
        )
        labeled_df = add_binary_labels(
            raw_df,
            horizons=horizons,
            label_mode="quantile_trade",
            label_direction="long",
        )
    else:
        label_return_fn = config.get_label_return_fn(label_mode)
        labeled_df = add_binary_labels(
            raw_df,
            horizons=horizons,
            threshold=label_threshold,
            return_fn=label_return_fn,
            label_mode=label_mode,
            label_direction=label_direction,
            exit_after_k=exit_after_k,
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
    feature_windows = list(config.WINDOWS)
    base_feature_space: CryptoFeatureSpace | None = None
    if meta_base is not None and config.META_LEARNER_FEATURE_DATA is not None:
        base_windows = required_feature_windows(meta_base.individual)
        logger.info("Building 5m base feature space | windows=%s", base_windows)
        base_feature_df = build_feature_frame(
            raw_df,
            windows=base_windows,
            quality_filter=False,
        )
        base_feature_space = CryptoFeatureSpace(
            base_feature_df,
            selectable_features(base_feature_df),
        )

        logger.info(
            "Loading lower-timeframe meta features from %s",
            config.META_LEARNER_FEATURE_DATA,
        )
        meta_feature_raw_df = load_ohlcv(config.META_LEARNER_FEATURE_DATA)
        alignment = build_meta_feature_alignment(
            labeled_df.index,
            meta_feature_raw_df.index,
            include_h1=config.META_LEARNER_FEATURE_INCLUDE_H1,
            lookahead_bars=config.META_LEARNER_FEATURE_LOOKAHEAD_BARS,
        )
        config.META_LEARNER_TARGET_INTERVAL_SECONDS = (
            alignment.target_interval.total_seconds()
        )
        config.META_LEARNER_FEATURE_INTERVAL_SECONDS = (
            alignment.feature_interval.total_seconds()
        )
        if (
            config.META_LEARNER_FEATURE_LOOKAHEAD_BARS > 0
            and not config.META_LEARNER_FEATURE_INCLUDE_H1
        ):
            logger.warning(
                "Meta lookahead test mode: %d observed lower-timeframe bars "
                "remain included in the original H1-H%d TP-hit target.",
                config.META_LEARNER_FEATURE_LOOKAHEAD_BARS,
                meta_base.horizon,
            )
        meta_path_mfe_column: str | None = None
        meta_observed_mfe_column: str | None = None
        if (
            alignment.lookahead_bars > 0
            and config.META_LEARNER_FEATURE_INCLUDE_H1
        ):
            meta_path_mfe_column = (
                f"meta_post_observation_mfe_h{meta_base.horizon}_"
                f"lb{alignment.lookahead_bars}"
            )
            post_observation_mfe = build_post_observation_mfe(
                raw_df,
                meta_feature_raw_df,
                alignment,
                horizon=meta_base.horizon,
            )
            meta_observed_mfe_column = (
                f"meta_observed_mfe_lb{alignment.lookahead_bars}"
            )
            observed_mfe = build_observed_mfe(
                raw_df,
                meta_feature_raw_df,
                alignment,
            )
            labeled_df[meta_path_mfe_column] = post_observation_mfe
            labeled_df[meta_observed_mfe_column] = observed_mfe
            train_df[meta_path_mfe_column] = post_observation_mfe.reindex(
                train_df.index
            )
            train_df[meta_observed_mfe_column] = observed_mfe.reindex(train_df.index)
            val_df[meta_path_mfe_column] = post_observation_mfe.reindex(val_df.index)
            val_df[meta_observed_mfe_column] = observed_mfe.reindex(val_df.index)
            test_df[meta_path_mfe_column] = post_observation_mfe.reindex(
                test_df.index
            )
            test_df[meta_observed_mfe_column] = observed_mfe.reindex(test_df.index)
            for fold in folds:
                fold.train_df[meta_path_mfe_column] = post_observation_mfe.reindex(
                    fold.train_df.index
                )
                fold.val_df[meta_path_mfe_column] = post_observation_mfe.reindex(
                    fold.val_df.index
                )
                fold.train_df[meta_observed_mfe_column] = observed_mfe.reindex(
                    fold.train_df.index
                )
                fold.val_df[meta_observed_mfe_column] = observed_mfe.reindex(
                    fold.val_df.index
                )
            logger.info(
                "Meta post-observation target: observed=%d lower-timeframe bars | "
                "remaining path through H%d | valid=%d/%d",
                alignment.lookahead_bars,
                meta_base.horizon,
                int(post_observation_mfe.notna().sum()),
                len(post_observation_mfe),
            )
        native_quality_index = alignment.source_index_for_targets(
            feature_quality_index
        )
        native_feature_df = build_feature_frame(
            meta_feature_raw_df,
            windows=feature_windows,
            quality_index=native_quality_index,
            output_index=alignment.source_index,
        )
        feature_df = align_meta_feature_frame(native_feature_df, alignment)
        del native_feature_df, meta_feature_raw_df
        gc.collect()
    else:
        meta_path_mfe_column = None
        meta_observed_mfe_column = None
        if meta_base is not None:
            feature_windows = sorted(
                set(feature_windows)
                | set(required_feature_windows(meta_base.individual))
            )
            logger.info("Meta/base required feature windows: %s", feature_windows)
        feature_df = build_feature_frame(
            raw_df,
            windows=feature_windows,
            quality_index=feature_quality_index,
        )
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)
    if base_feature_space is None:
        base_feature_space = feature_space
    logger.info("Feature pool: %d safe features.", len(feature_pool))
    if len(feature_pool) < config.FEATURE_MIN:
        raise ValueError("Feature pool is smaller than FEATURE_MIN.")

    if meta_base is not None:
        meta_data = build_meta_learner_data(
            base_labeled_df=labeled_df,
            original_folds=folds,
            final_train_df=train_df,
            final_val_df=val_df,
            final_test_df=test_df,
            feature_space=base_feature_space,
            base=meta_base,
            min_prediction=config.META_LEARNER_MIN_PREDICTION,
            target_mode=label_mode,
            label_threshold=float(label_threshold),
            tp_offset=config.META_LEARNER_TP_OFFSET,
            meta_val_fraction=config.META_LEARNER_META_VAL_FRACTION,
            target_start_step=config.META_LEARNER_TARGET_START_STEP,
            path_mfe_column=meta_path_mfe_column,
            observed_mfe_column=meta_observed_mfe_column,
            purge_bars=purge_bars,
            test_start=test_start,
        )
        if base_feature_space is not feature_space:
            base_feature_space.clear_expression_cache()
            del base_feature_space, base_feature_df
            gc.collect()
        folds = meta_data.folds
        train_df = meta_data.train_df
        val_df = meta_data.val_df
        test_df = meta_data.test_df
        if config.meta_prediction_is_feature(label_mode):
            tp_feature = f"meta_dynamic_tp_h{meta_base.horizon}"
            tp_parts = [
                frame[tp_feature]
                for frame in (train_df, val_df, test_df)
                if tp_feature in frame
            ]
            if len(tp_parts) != 3:
                raise ValueError(
                    f"Missing OOF dynamic-TP feature for {label_mode}: {tp_feature}."
                )
            tp_values = pd.concat(tp_parts).sort_index()
            tp_values = tp_values.loc[~tp_values.index.duplicated(keep="last")]
            previous_feature_space = feature_space
            feature_df = feature_df.copy()
            feature_df[tp_feature] = tp_values.reindex(feature_df.index)
            feature_pool = selectable_features(feature_df)
            feature_space = CryptoFeatureSpace(feature_df, feature_pool)
            previous_feature_space.clear_expression_cache()
            logger.info(
                "Added leakage-safe OOF dynamic TP to %s features: %s | "
                "valid=%d/%d",
                label_mode,
                tp_feature,
                int(feature_df[tp_feature].notna().sum()),
                len(feature_df),
            )
        feature_quality_index = folds[0].train_df.index
        logger.info(
            "Meta final split: OOF train=%d | val=%d | test=%d | folds=%d",
            len(train_df),
            len(val_df),
            len(test_df),
            len(folds),
        )

    mutator = CryptoMutator(
        feature_pool=feature_pool,
        feature_space=feature_space,
        train_index=feature_quality_index,
        seed=int(rng.integers(1 << 31)),
    )
    evaluator = _make_fitness_evaluator(
        label_mode,
        horizons,
        quantile_target=quantile_target,
        quantile=quantile,
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
            exit_after_k=exit_after_k,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            wf_end=wf_end,
            wf_min_train_months=wf_min_train_months,
            wf_val_months=wf_val_months,
            wf_step_months=wf_step_months,
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
                    exit_after_k,
                    val_start=val_start,
                    test_start=test_start,
                    test_end=test_end,
                    wf_end=wf_end,
                    wf_min_train_months=wf_min_train_months,
                    wf_val_months=wf_val_months,
                    wf_step_months=wf_step_months,
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
            archive,
            save_path,
            horizons,
            label_threshold,
            label_mode,
            label_direction,
            exit_after_k,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            wf_end=wf_end,
            wf_min_train_months=wf_min_train_months,
            wf_val_months=wf_val_months,
            wf_step_months=wf_step_months,
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
            exit_after_k,
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            wf_end=wf_end,
            wf_min_train_months=wf_min_train_months,
            wf_val_months=wf_val_months,
            wf_step_months=wf_step_months,
        )
    return archive


def _make_fitness_evaluator(
    label_mode: str,
    horizons: list[int] | tuple[int, ...],
    quantile_target: str = config.QUANTILE_TARGET,
    quantile: float = config.QUANTILE_ALPHA,
) -> CryptoFitnessEvaluator | QuantileFitnessEvaluator:
    """Select the mode-specific evaluator without changing split construction."""
    selected_mode = config.canonical_label_mode(label_mode)
    if selected_mode == "quantile_trade":
        return QuantileFitnessEvaluator(
            horizons=horizons,
            target=quantile_target,
            quantile=quantile,
        )
    return CryptoFitnessEvaluator(
        horizons=horizons,
        precision_only=config.is_precision_only_label_mode(selected_mode),
    )


def _evaluate_final_archive(
    archive: CryptoArchive,
    evaluator: CryptoFitnessEvaluator | QuantileFitnessEvaluator,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
) -> None:
    logger.info(
        "Budget ended; running final val/test evaluation for %d archive entries.",
        len(archive),
    )
    logger.warning(
        "Final test metrics are diagnostic only and do not change archive ranking; "
        "selecting a rank after inspecting test_metrics would leak the holdout."
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


def _timestamp_text(value: str | pd.Timestamp | None) -> str | None:
    if value in (None, ""):
        return None
    return pd.Timestamp(value).isoformat()


def _path_text(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    return Path(value).as_posix()


def _save_archive(
    archive: CryptoArchive,
    path: Path,
    horizons: list[int],
    label_threshold: float,
    label_mode: str,
    label_direction: str,
    exit_after_k: int | None,
    *,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    wf_end: str = config.WF_END,
    wf_min_train_months: int = config.WF_MIN_TRAIN_MONTHS,
    wf_val_months: int = config.WF_VAL_MONTHS,
    wf_step_months: int = config.WF_STEP_MONTHS,
) -> None:
    metadata = {
        "pipeline": "crypto",
        "horizons": horizons,
        "label_mode": label_mode,
        "label_direction": label_direction,
        "exit_after_k": exit_after_k,
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
        "slope_slowdown_all_rule": config.SLOPE_SLOWDOWN_ALL_RULE,
        "ma_slope_fast_window": config.MA_SLOPE_FAST_WINDOW,
        "ma_slope_fast_shift": config.MA_SLOPE_FAST_SHIFT,
        "ma_slope_future_shift": config.MA_SLOPE_FUTURE_SHIFT,
        "ma_slope_reversal_rule": config.MA_SLOPE_REVERSAL_RULE,
        "bear_zigzag_tolerance": config.BEAR_ZIGZAG_TOLERANCE,
        "bear_min_drop": config.BEAR_MIN_DROP,
        "bear_min_bars": config.BEAR_MIN_BARS,
        "bear_label_rule": config.BEAR_LABEL_RULE,
        "bull_zigzag_tolerance": config.BULL_ZIGZAG_TOLERANCE,
        "bull_min_rise": config.BULL_MIN_RISE,
        "bull_min_bars": config.BULL_MIN_BARS,
        "bull_label_rule": config.BULL_LABEL_RULE,
        "fitness_horizon_mode": (
            "mean" if label_mode == "quantile_trade" else config.FITNESS_HORIZON_MODE
        ),
        "fitness": (
            config.quantile_trade_fitness_weights(config.QUANTILE_TARGET)
            if label_mode == "quantile_trade"
            else config.FITNESS_WEIGHTS
        ),
        "trade_top_fraction": config.TRADE_TOP_FRACTION,
        "trade_cost": config.TRADE_COST,
        "return_score_scale": config.RETURN_SCORE_SCALE,
        "precision_only": config.is_precision_only_label_mode(label_mode),
        "split_policy": {
            "val_start": _timestamp_text(val_start),
            "test_start": _timestamp_text(test_start),
            "test_end": _timestamp_text(test_end),
            "wf_end": _timestamp_text(wf_end),
            "wf_min_train_months": int(wf_min_train_months),
            "wf_val_months": int(wf_val_months),
            "wf_step_months": int(wf_step_months),
        },
    }
    if label_mode == "quantile_trade":
        metadata.pop("trade_top_fraction", None)
        metadata.pop("trade_cost", None)
        metadata.update(
            {
                "quantile_trade_rule": config.QUANTILE_TRADE_RULE,
                "quantile_target": config.QUANTILE_TARGET,
                "quantile_alpha": config.QUANTILE_ALPHA,
            }
        )
    elif config.is_meta_learner_label_mode(label_mode):
        base = load_meta_base(
            config.META_LEARNER_BASE_ARCHIVE,
            config.META_LEARNER_BASE_RANK,
            horizons,
        )
        metadata.update(
            {
                "meta_learner_rule": config.meta_learner_rule(label_mode),
                "meta_target_mode": label_mode,
                "meta_base_archive": str(config.META_LEARNER_BASE_ARCHIVE),
                "meta_base_archive_sha256": base.archive_sha256,
                "meta_base_rank": int(config.META_LEARNER_BASE_RANK),
                "meta_base_horizon": int(base.horizon),
                "meta_base_quantile": float(base.quantile),
                "meta_min_prediction": float(
                    config.META_LEARNER_MIN_PREDICTION
                ),
                "meta_tp_offset": float(config.META_LEARNER_TP_OFFSET),
                "meta_val_fraction": float(
                    config.META_LEARNER_META_VAL_FRACTION
                ),
                "meta_prediction_is_feature": config.meta_prediction_is_feature(
                    label_mode
                ),
                "meta_feature_data": (
                    _path_text(config.META_LEARNER_FEATURE_DATA)
                ),
                "meta_feature_include_h1": bool(
                    config.META_LEARNER_FEATURE_INCLUDE_H1
                ),
                "meta_feature_lookahead_bars": int(
                    config.META_LEARNER_FEATURE_LOOKAHEAD_BARS
                ),
                "meta_feature_alignment_rule": (
                    config.META_LEARNER_FEATURE_ALIGNMENT_RULE
                ),
                "meta_target_interval_seconds": (
                    config.META_LEARNER_TARGET_INTERVAL_SECONDS
                ),
                "meta_feature_interval_seconds": (
                    config.META_LEARNER_FEATURE_INTERVAL_SECONDS
                ),
                "meta_target_start_step": int(
                    config.META_LEARNER_TARGET_START_STEP
                ),
                "meta_target_path_rule": (
                    "observed_feature_inclusive_original_mfe_target_v3"
                    if config.META_LEARNER_FEATURE_LOOKAHEAD_BARS > 0
                    else (
                        "exclude_observed_hits_then_lower_timeframe_to_horizon_v2"
                        if config.META_LEARNER_FEATURE_INCLUDE_H1
                        else "target_candle_steps_v1"
                    )
                ),
                "meta_final_train_source": "original_wf_validation_oof",
                "meta_final_val_base_train_end": _timestamp_text(val_start),
                "meta_final_test_base_train_end": _timestamp_text(test_start),
            }
        )
    archive.save(path, metadata=metadata)


def _validate_resume_metadata(
    archive: CryptoArchive,
    resume_path: Path,
    horizons: list[int],
    label_mode: str,
    label_direction: str,
    label_threshold: float,
    exit_after_k: int | None = None,
    *,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
    wf_end: str = config.WF_END,
    wf_min_train_months: int = config.WF_MIN_TRAIN_MONTHS,
    wf_val_months: int = config.WF_VAL_MONTHS,
    wf_step_months: int = config.WF_STEP_MONTHS,
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
    archive_exit_after_k = config.resolve_exit_after_k(
        metadata_label_mode or archive_label_mode,
        metadata.get("exit_after_k"),
    )
    checks: list[tuple[str, object, object]] = [
        (
            "horizons",
            [int(h) for h in metadata.get("horizons", [])],
            [int(h) for h in horizons],
        ),
        (
            "label_mode",
            archive_label_mode,
            label_mode,
        ),
        ("label_threshold", metadata.get("label_threshold"), float(label_threshold)),
        (
            "fitness_horizon_mode",
            str(metadata.get("fitness_horizon_mode", "")).strip().lower(),
            "mean" if label_mode == "quantile_trade" else config.FITNESS_HORIZON_MODE,
        ),
    ]
    split_policy = metadata.get("split_policy")
    if isinstance(split_policy, dict):
        checks.append(
            (
                "split_policy",
                split_policy,
                {
                    "val_start": _timestamp_text(val_start),
                    "test_start": _timestamp_text(test_start),
                    "test_end": _timestamp_text(test_end),
                    "wf_end": _timestamp_text(wf_end),
                    "wf_min_train_months": int(wf_min_train_months),
                    "wf_val_months": int(wf_val_months),
                    "wf_step_months": int(wf_step_months),
                },
            )
        )
    else:
        logger.warning(
            "Resume archive %s has no split_policy metadata; split compatibility "
            "cannot be verified.",
            resume_path,
        )
    if label_mode != "quantile_trade":
        checks.extend(
            [
                (
                    "trade_top_fraction",
                    metadata.get("trade_top_fraction"),
                    float(config.TRADE_TOP_FRACTION),
                ),
                ("trade_cost", metadata.get("trade_cost"), float(config.TRADE_COST)),
            ]
        )
    if exit_after_k is not None or archive_exit_after_k is not None:
        checks.append(("exit_after_k", archive_exit_after_k, exit_after_k))
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
    if (
        label_mode == "slope_slowdown_all"
        and archive_label_mode == "slope_slowdown_all"
    ):
        archive_rule = metadata.get("slope_slowdown_all_rule")
        if archive_rule != config.SLOPE_SLOWDOWN_ALL_RULE:
            raise ValueError(
                "Resume archive uses an incompatible slope_slowdown_all rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.SLOPE_SLOWDOWN_ALL_RULE!r}. "
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
                    "slope_price_column",
                    metadata.get("slope_price_column"),
                    config.SLOPE_PRICE_COLUMN,
                ),
            ]
        )
    if (
        label_mode == "ma_slope_reversal"
        and archive_label_mode == "ma_slope_reversal"
    ):
        archive_rule = metadata.get("ma_slope_reversal_rule")
        if archive_rule != config.MA_SLOPE_REVERSAL_RULE:
            raise ValueError(
                "Resume archive uses an incompatible ma_slope_reversal rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.MA_SLOPE_REVERSAL_RULE!r}. "
                "Start a new archive."
            )
        checks.extend(
            [
                (
                    "ma_slope_fast_window",
                    metadata.get("ma_slope_fast_window"),
                    int(config.MA_SLOPE_FAST_WINDOW),
                ),
                (
                    "ma_slope_fast_shift",
                    metadata.get("ma_slope_fast_shift"),
                    int(config.MA_SLOPE_FAST_SHIFT),
                ),
                (
                    "ma_slope_future_shift",
                    metadata.get("ma_slope_future_shift"),
                    int(config.MA_SLOPE_FUTURE_SHIFT),
                ),
            ]
        )
    if label_mode == "bear" and archive_label_mode == "bear":
        archive_rule = metadata.get("bear_label_rule")
        if archive_rule != config.BEAR_LABEL_RULE:
            raise ValueError(
                "Resume archive uses an incompatible bear labeling rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.BEAR_LABEL_RULE!r}. Start a new archive."
            )
        checks.extend(
            [
                (
                    "bear_zigzag_tolerance",
                    metadata.get("bear_zigzag_tolerance"),
                    float(config.BEAR_ZIGZAG_TOLERANCE),
                ),
                (
                    "bear_min_drop",
                    metadata.get("bear_min_drop"),
                    float(config.BEAR_MIN_DROP),
                ),
                (
                    "bear_min_bars",
                    metadata.get("bear_min_bars"),
                    int(config.BEAR_MIN_BARS),
                ),
            ]
        )
    if label_mode == "bull" and archive_label_mode == "bull":
        archive_rule = metadata.get("bull_label_rule")
        if archive_rule != config.BULL_LABEL_RULE:
            raise ValueError(
                "Resume archive uses an incompatible bull labeling rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.BULL_LABEL_RULE!r}. Start a new archive."
            )
        checks.extend(
            [
                (
                    "bull_zigzag_tolerance",
                    metadata.get("bull_zigzag_tolerance"),
                    float(config.BULL_ZIGZAG_TOLERANCE),
                ),
                (
                    "bull_min_rise",
                    metadata.get("bull_min_rise"),
                    float(config.BULL_MIN_RISE),
                ),
                (
                    "bull_min_bars",
                    metadata.get("bull_min_bars"),
                    int(config.BULL_MIN_BARS),
                ),
            ]
        )
    if label_mode == "quantile_trade" and archive_label_mode == "quantile_trade":
        archive_rule = metadata.get("quantile_trade_rule")
        if archive_rule != config.QUANTILE_TRADE_RULE:
            raise ValueError(
                "Resume archive uses an incompatible quantile_trade rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={config.QUANTILE_TRADE_RULE!r}. Start a new archive."
            )
        checks.extend(
            [
                (
                    "quantile_target",
                    metadata.get("quantile_target"),
                    config.QUANTILE_TARGET,
                ),
                (
                    "quantile_alpha",
                    metadata.get("quantile_alpha"),
                    float(config.QUANTILE_ALPHA),
                ),
            ]
        )
    if (
        config.is_meta_learner_label_mode(label_mode)
        and archive_label_mode == label_mode
    ):
        archive_rule = metadata.get("meta_learner_rule")
        required_rule = config.meta_learner_rule(label_mode)
        if archive_rule != required_rule:
            raise ValueError(
                "Resume archive uses an incompatible meta_learner rule. "
                f"Archive={resume_path}, archive rule={archive_rule!r}, "
                f"required rule={required_rule!r}. Start a new archive."
            )
        base = load_meta_base(
            config.META_LEARNER_BASE_ARCHIVE,
            config.META_LEARNER_BASE_RANK,
            horizons,
        )
        if config.META_LEARNER_FEATURE_DATA is not None:
            required_feature_fields = {
                "meta_feature_data",
                "meta_feature_alignment_rule",
                "meta_target_interval_seconds",
                "meta_feature_interval_seconds",
                "meta_feature_include_h1",
                "meta_feature_lookahead_bars",
                "meta_target_start_step",
                "meta_target_path_rule",
            }
            missing_feature_fields = sorted(
                field
                for field in required_feature_fields
                if metadata.get(field) in (None, "")
            )
            if missing_feature_fields:
                raise ValueError(
                    "Resume archive predates or is missing multi-timeframe meta "
                    f"metadata {missing_feature_fields}. Start a new archive."
                )
        checks.extend(
            [
                (
                    "meta_target_mode",
                    metadata.get("meta_target_mode", "meta_learner"),
                    label_mode,
                ),
                (
                    "meta_base_archive_sha256",
                    metadata.get("meta_base_archive_sha256"),
                    base.archive_sha256,
                ),
                (
                    "meta_base_rank",
                    metadata.get("meta_base_rank"),
                    int(config.META_LEARNER_BASE_RANK),
                ),
                (
                    "meta_base_horizon",
                    metadata.get("meta_base_horizon"),
                    int(base.horizon),
                ),
                (
                    "meta_base_quantile",
                    metadata.get("meta_base_quantile"),
                    float(base.quantile),
                ),
                (
                    "meta_min_prediction",
                    metadata.get("meta_min_prediction"),
                    float(config.META_LEARNER_MIN_PREDICTION),
                ),
                (
                    "meta_tp_offset",
                    metadata.get("meta_tp_offset", 0.0),
                    float(config.META_LEARNER_TP_OFFSET),
                ),
                (
                    "meta_val_fraction",
                    metadata.get("meta_val_fraction"),
                    float(config.META_LEARNER_META_VAL_FRACTION),
                ),
                (
                    "meta_prediction_is_feature",
                    metadata.get("meta_prediction_is_feature"),
                    config.meta_prediction_is_feature(label_mode),
                ),
                (
                    "meta_feature_data",
                    metadata.get("meta_feature_data"),
                    _path_text(config.META_LEARNER_FEATURE_DATA),
                ),
                (
                    "meta_feature_include_h1",
                    metadata.get("meta_feature_include_h1"),
                    bool(config.META_LEARNER_FEATURE_INCLUDE_H1),
                ),
                (
                    "meta_feature_lookahead_bars",
                    metadata.get("meta_feature_lookahead_bars"),
                    int(config.META_LEARNER_FEATURE_LOOKAHEAD_BARS),
                ),
                (
                    "meta_feature_alignment_rule",
                    metadata.get("meta_feature_alignment_rule"),
                    config.META_LEARNER_FEATURE_ALIGNMENT_RULE,
                ),
                (
                    "meta_target_interval_seconds",
                    metadata.get("meta_target_interval_seconds"),
                    config.META_LEARNER_TARGET_INTERVAL_SECONDS,
                ),
                (
                    "meta_feature_interval_seconds",
                    metadata.get("meta_feature_interval_seconds"),
                    config.META_LEARNER_FEATURE_INTERVAL_SECONDS,
                ),
                (
                    "meta_target_start_step",
                    metadata.get("meta_target_start_step"),
                    int(config.META_LEARNER_TARGET_START_STEP),
                ),
                (
                    "meta_target_path_rule",
                    metadata.get("meta_target_path_rule"),
                    (
                        "observed_feature_inclusive_original_mfe_target_v3"
                    )
                    if config.META_LEARNER_FEATURE_LOOKAHEAD_BARS > 0
                    else (
                        "exclude_observed_hits_then_lower_timeframe_to_horizon_v2"
                        if config.META_LEARNER_FEATURE_INCLUDE_H1
                        else "target_candle_steps_v1"
                    ),
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
            "Comma-separated horizons. quantile_trade fits one quantile model "
            "per horizon and averages accuracy across them. Default: "
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
            "and slope_slowdown_all "
            "this is the minimum high-price OLS slope change per candle; its "
            "mode default is config.SLOPE_SLOWDOWN_THRESHOLD. For "
            "two_sided_tp it is the positive absolute TP used on both sides."
            " For meta_close_exit it is the minimum close-H return. For "
            "meta_strategy_profit it is the minimum TP-or-close strategy "
            "return and defaults to TRADE_COST. It is ignored by bear/bull, "
            "ma_slope_reversal, quantile_trade, and the original meta_learner."
        ),
    )
    parser.add_argument(
        "--label-mode",
        default=config.LABEL_MODE,
        help=(
            f"Label mode. Allowed: {', '.join(sorted(config.LABEL_RETURN_FNS))}. "
            "first_hit_safe_close/safe_close -> safe_path_mfe. "
            f"Default: {config.LABEL_MODE}."
        ),
    )
    parser.add_argument(
        "--label-direction",
        default=config.LABEL_DIRECTION,
        help=(
            "Label direction. Long means price up is favorable; Short means "
            "price down is favorable. It is ignored by bear, bull, and "
            "two_sided_tp, and quantile_trade. quantile_trade always defines "
            "MFE as upward excursion and MAE as downward excursion. "
            f"Default: {config.LABEL_DIRECTION}."
        ),
    )
    parser.add_argument(
        "--quantile-target",
        choices=("mfe", "mae", "close"),
        default=config.QUANTILE_TARGET,
        help=(
            "Regression target for --label-mode quantile_trade: mfe, mae, or close. "
            f"Default: {config.QUANTILE_TARGET}."
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=config.QUANTILE_ALPHA,
        help=(
            "Quantile alpha strictly between 0 and 1 for quantile_trade, "
            f"for example 0.20 or 0.80. Default: {config.QUANTILE_ALPHA}."
        ),
    )
    parser.add_argument(
        "--meta-base-archive",
        default=str(config.META_LEARNER_BASE_ARCHIVE),
        help=(
            "Ranked quantile-MFE archive used only to construct OOF dynamic-TP "
            "targets for --label-mode meta_learner."
            " The same base is used by meta_close_exit and "
            "meta_strategy_profit."
        ),
    )
    parser.add_argument(
        "--meta-base-rank",
        type=int,
        default=config.META_LEARNER_BASE_RANK,
        help="Rank from --meta-base-archive. Default: 1.",
    )
    parser.add_argument(
        "--meta-feature-data",
        default=(
            str(config.META_LEARNER_FEATURE_DATA)
            if config.META_LEARNER_FEATURE_DATA is not None
            else None
        ),
        help=(
            "Optional lower-timeframe OHLCV used only by the evolved meta "
            "model. Base prediction and labels continue to use --data."
        ),
    )
    parser.add_argument(
        "--meta-feature-lookahead-bars",
        type=int,
        default=config.META_LEARNER_FEATURE_LOOKAHEAD_BARS,
        help=(
            "Number of lower-timeframe H1 candles observed before meta "
            "prediction. For 1m features, use 1 to observe only minute one."
        ),
    )
    parser.add_argument(
        "--meta-feature-include-h1",
        action=argparse.BooleanOptionalAction,
        default=config.META_LEARNER_FEATURE_INCLUDE_H1,
        help=(
            "Let the meta model observe the complete H1 lower-timeframe candle. "
            "The TP remains anchored to open H1 and target hits start at H2."
        ),
    )
    parser.add_argument(
        "--meta-min-prediction",
        type=float,
        default=config.META_LEARNER_MIN_PREDICTION,
        help=(
            "Keep meta samples only when the OOF predicted TP is strictly "
            "above this return. Default: 0.0002."
        ),
    )
    parser.add_argument(
        "--meta-tp-offset",
        type=float,
        default=config.META_LEARNER_TP_OFFSET,
        help=(
            "Non-negative fixed return added to each MFE base prediction "
            "before constructing the meta TP label and hit return. "
            "For example 0.0005 adds 0.05%%. Default: 0.0."
        ),
    )
    parser.add_argument(
        "--meta-val-fraction",
        type=float,
        default=config.META_LEARNER_META_VAL_FRACTION,
        help=(
            "Chronological tail of each original WF validation block reserved "
            "for meta validation. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--exit-after-k",
        type=int,
        default=None,
        help=(
            "Decision candle k for --label-mode exit_after_k. "
            f"Default: config.EXIT_AFTER_K={config.EXIT_AFTER_K}. "
            "Requires 1 <= k < every holding horizon."
        ),
    )
    parser.add_argument(
        "--trade-top-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of highest predictions selected for fitness/trading. "
            "Ignored by quantile_trade, which evaluates quantile accuracy only. "
            f"Default: config.TRADE_TOP_FRACTION={config.TRADE_TOP_FRACTION}."
        ),
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument(
        "--wf-end",
        default=None,
        help=(
            "Exclusive end of data available to walk-forward evolution. "
            "Default: --val-start, keeping final validation independent."
        ),
    )
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
        quantile_target=args.quantile_target,
        quantile=args.quantile,
        meta_base_archive=args.meta_base_archive,
        meta_base_rank=args.meta_base_rank,
        meta_feature_data=args.meta_feature_data,
        meta_feature_lookahead_bars=args.meta_feature_lookahead_bars,
        meta_feature_include_h1=args.meta_feature_include_h1,
        meta_min_prediction=args.meta_min_prediction,
        meta_tp_offset=args.meta_tp_offset,
        meta_val_fraction=args.meta_val_fraction,
        exit_after_k=args.exit_after_k,
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
