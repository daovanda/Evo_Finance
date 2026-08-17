"""Compare Rank 1 score bands for the after-1m/2m/3m meta archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto import config
from crypto.data import (
    add_binary_labels,
    load_ohlcv,
    make_walk_forward_folds,
    split_labeled_by_dates,
)
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.fitness import CryptoFitnessEvaluator
from crypto.meta_targets import (
    align_meta_feature_frame,
    build_meta_feature_alignment,
    build_meta_learner_data,
    load_meta_base,
    required_feature_windows,
)
from crypto.quantile_fitness import QuantileFitnessEvaluator
from temp.plot_5m_meta_after_1m_signals import (
    _build_direct_meta_features,
    _load_rank,
    _valid_meta_frame,
)


ARCHIVES = (
    Path("crypto/results/crypto_btc_5m_meta_after_1m_mfe_q20_h3_top80_seed1_8h.json"),
    Path("crypto/results/crypto_btc_5m_meta_after_2m_mfe_q20_h3_top80_seed1_8h.json"),
    Path("crypto/results/crypto_btc_5m_meta_after_3m_mfe_q20_h3_top80_seed1_8h.json"),
)


def _split_bands(
    prediction: pd.Series,
    meta: pd.DataFrame,
    horizon: int,
    raw: pd.DataFrame,
    minute: pd.DataFrame,
    target_interval: pd.Timedelta,
) -> pd.DataFrame:
    prediction = prediction.dropna().sort_index()
    meta = meta.reindex(prediction.index)
    order = prediction.rank(method="first", ascending=False).astype(int) - 1
    band = np.minimum((order.to_numpy() * 20) // len(prediction), 19)
    entry_index = pd.DatetimeIndex(prediction.index) + target_interval
    open_h1 = pd.to_numeric(raw["open"], errors="coerce").reindex(entry_index)
    close_minute_1 = pd.to_numeric(minute["close"], errors="coerce").reindex(
        entry_index
    )
    close_minute_1_return = close_minute_1.to_numpy(float) / open_h1.to_numpy(float) - 1.0
    return pd.DataFrame(
        {
            "timestamp": prediction.index,
            "band": band,
            "hit": pd.to_numeric(meta[f"label_h{horizon}"], errors="coerce").to_numpy(),
            "tp": pd.to_numeric(
                meta[f"meta_dynamic_tp_h{horizon}"], errors="coerce"
            ).to_numpy(),
            "mfe": pd.to_numeric(
                meta[f"quantile_up_mfe_h{horizon}"], errors="coerce"
            ).to_numpy(),
            "close_h": pd.to_numeric(
                meta[f"quantile_close_return_h{horizon}"], errors="coerce"
            ).to_numpy(),
            "close_minute_1": close_minute_1_return,
        }
    )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_metadata, _ = _load_rank(args.archives[0], args.rank)
    horizons = [int(value) for value in first_metadata["horizons"]]
    if len(horizons) != 1:
        raise ValueError("This analysis requires exactly one horizon.")
    horizon = horizons[0]
    split = dict(first_metadata.get("split_policy", {}))
    val_start = str(split.get("val_start", config.VAL_START))
    test_start = str(split.get("test_start", config.TEST_START))
    test_end = split.get("test_end", config.TEST_END)
    wf_end = str(split.get("wf_end", val_start))
    purge = config.purge_bars_for_horizons(horizons)

    raw = load_ohlcv(args.data)
    labeled = add_binary_labels(
        raw, horizons=horizons, label_mode="quantile_trade", label_direction="long"
    )
    train, val, test = split_labeled_by_dates(
        labeled,
        val_start=val_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=purge,
    )
    folds = make_walk_forward_folds(
        labeled[labeled.index < pd.Timestamp(wf_end)],
        wf_end=wf_end,
        min_train_months=int(split.get("wf_min_train_months", config.WF_MIN_TRAIN_MONTHS)),
        val_months=int(split.get("wf_val_months", config.WF_VAL_MONTHS)),
        step_months=int(split.get("wf_step_months", config.WF_STEP_MONTHS)),
        purge_bars=purge,
    )

    fingerprint = hashlib.sha256(
        repr(
            (
                args.data.resolve(),
                args.data.stat().st_size,
                args.data.stat().st_mtime_ns,
                Path(str(first_metadata["meta_base_archive"])).resolve(),
                Path(str(first_metadata["meta_base_archive"])).stat().st_mtime_ns,
                first_metadata.get("meta_base_rank", 1),
                first_metadata.get("meta_min_prediction", 0.0002),
                first_metadata.get("meta_tp_offset", 0.0),
                first_metadata.get("meta_val_fraction", 0.20),
                first_metadata.get("meta_target_start_step", 1),
                tuple(horizons),
                tuple(sorted(split.items())),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    args.model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.model_dir / "common_meta_oof.pkl"
    if cache_path.exists() and not args.rebuild_feature_cache:
        meta_data = pd.read_pickle(cache_path)
        print(f"Loaded common OOF cache: {cache_path}", flush=True)
    else:
        base = load_meta_base(
            Path(str(first_metadata["meta_base_archive"])),
            int(first_metadata.get("meta_base_rank", 1)),
            horizons,
        )
        base_features = build_feature_frame(
            raw,
            windows=required_feature_windows(base.individual),
            quality_filter=False,
        )
        base_space = CryptoFeatureSpace(
            base_features, selectable_features(base_features)
        )
        meta_data = build_meta_learner_data(
            base_labeled_df=labeled,
            original_folds=folds,
            final_train_df=train,
            final_val_df=val,
            final_test_df=test,
            feature_space=base_space,
            base=base,
            min_prediction=float(first_metadata.get("meta_min_prediction", 0.0002)),
            tp_offset=float(first_metadata.get("meta_tp_offset", 0.0)),
            meta_val_fraction=float(first_metadata.get("meta_val_fraction", 0.20)),
            target_start_step=int(first_metadata.get("meta_target_start_step", 1)),
            purge_bars=purge,
            test_start=test_start,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(meta_data, cache_path)
        print(f"Saved common OOF cache: {cache_path}", flush=True)
    train_meta = _valid_meta_frame(meta_data.train_df, horizon)
    val_meta = _valid_meta_frame(meta_data.val_df, horizon)
    test_meta = _valid_meta_frame(meta_data.test_df, horizon)
    minute = load_ohlcv(args.data_1m)
    target_interval = pd.Timedelta(minutes=5)
    manifest_path = args.model_dir / "manifest.json"
    try:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        existing_manifest = {}
    existing_records = existing_manifest.get("models", [])

    def cached_model_matches(
        *, kind: str, model_path: Path, model_fingerprint: str, lookahead: int | None = None
    ) -> bool:
        for record in existing_records:
            if record.get("kind") != kind:
                continue
            if record.get("model_path") != model_path.name:
                continue
            if record.get("fingerprint") != model_fingerprint:
                continue
            if lookahead is not None and int(record.get("lookahead_bars", -1)) != lookahead:
                continue
            return model_path.exists()
        return False

    model_records: list[dict] = []

    base_model_fingerprint = hashlib.sha256(
        f"{fingerprint}|full_history_internal_stop_v1".encode("utf-8")
    ).hexdigest()[:16]
    base_model_path = args.model_dir / (
        f"mfe_q20_h{horizon}_rank{int(meta_data.base.rank)}.txt"
    )
    base_cache_valid = cached_model_matches(
        kind="base_quantile_mfe",
        model_path=base_model_path,
        model_fingerprint=base_model_fingerprint,
    )
    if args.retrain_models or not base_cache_valid:
        base = meta_data.base
        base_features = build_feature_frame(
            raw,
            windows=required_feature_windows(base.individual),
            quality_filter=False,
        )
        base_space = CryptoFeatureSpace(
            base_features, selectable_features(base_features)
        )
        base_evaluator = QuantileFitnessEvaluator(
            horizons=horizons,
            target="mfe",
            quantile=base.quantile,
        )
        base_train = base_evaluator._valid_frame(labeled, horizon)
        base_x = base_space.matrix(base.individual.features, base_train.index)
        base_y = base_train[base_evaluator.target_column(horizon)].astype(float)
        base_booster = base_evaluator._train_booster(
            base_x, base_y, horizon=horizon
        )
        base_booster.save_model(str(base_model_path))
        base_booster.free_dataset()
        print(f"Saved base MFE model: {base_model_path}", flush=True)
    else:
        print(f"Using cached base MFE model: {base_model_path}", flush=True)
    model_records.append(
        {
            "kind": "base_quantile_mfe",
            "model_path": base_model_path.name,
            "fingerprint": base_model_fingerprint,
            "archive": str(meta_data.base.archive_path),
            "rank": int(meta_data.base.rank),
            "horizon": int(horizon),
            "quantile": float(meta_data.base.quantile),
            "features": list(meta_data.base.individual.features),
            "training_scope": "all_valid_rows_with_internal_chronological_stop",
        }
    )

    summary_rows: list[dict] = []
    band_rows: list[pd.DataFrame] = []
    top50_sets: dict[str, set[pd.Timestamp]] = {}
    band_by_model: dict[str, pd.Series] = {}
    universe: set[pd.Timestamp] = set()
    for archive in args.archives:
        metadata, entry = _load_rank(archive, args.rank)
        lookahead = int(metadata.get("meta_feature_lookahead_bars", 0))
        alignment = build_meta_feature_alignment(
            labeled.index,
            minute.index,
            include_h1=bool(metadata.get("meta_feature_include_h1", False)),
            lookahead_bars=lookahead,
        )
        individual = CryptoIndividual(
            features=[str(value) for value in entry["features"]],
            generation=int(entry.get("generation", 0) or 0),
            score=float(entry.get("score", np.nan)),
        )
        feature_fingerprint = hashlib.sha256(
            repr(
                (
                    archive.resolve(),
                    archive.stat().st_size,
                    archive.stat().st_mtime_ns,
                    args.data_1m.resolve(),
                    args.data_1m.stat().st_size,
                    args.data_1m.stat().st_mtime_ns,
                    args.rank,
                    lookahead,
                    tuple(individual.features),
                    len(alignment.source_index),
                    alignment.source_index.min(),
                    alignment.source_index.max(),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        feature_cache_path = args.model_dir / f"features_after_{lookahead}m.pkl"
        feature_cache_record = next(
            (
                record
                for record in existing_records
                if record.get("kind") == "meta_rank1_analysis_replay"
                and int(record.get("lookahead_bars", -1)) == lookahead
            ),
            {},
        )
        feature_cache_valid = (
            feature_cache_path.exists()
            and feature_cache_record.get("feature_fingerprint")
            == feature_fingerprint
        )
        if feature_cache_valid and not args.rebuild_feature_cache:
            selected_features = pd.read_pickle(feature_cache_path)
            print(f"Loaded selected feature cache: {feature_cache_path}", flush=True)
        else:
            native = _build_direct_meta_features(
                minute, individual.features, alignment.source_index
            )
            if native is None:
                native = build_feature_frame(
                    minute,
                    windows=required_feature_windows(individual),
                    quality_filter=False,
                    output_index=alignment.source_index,
                )
            features = align_meta_feature_frame(native, alignment)
            space = CryptoFeatureSpace(features, selectable_features(features))
            selected_index = train_meta.index.append(val_meta.index).append(
                test_meta.index
            )
            selected_matrix = space.matrix(individual.features, selected_index)
            selected_features = pd.DataFrame(
                selected_matrix,
                index=selected_index,
                columns=individual.features,
            )
            selected_features.to_pickle(feature_cache_path)
            print(f"Saved selected feature cache: {feature_cache_path}", flush=True)
        x_train = selected_features.reindex(train_meta.index).to_numpy(float)
        x_val = selected_features.reindex(val_meta.index).to_numpy(float)
        x_test = selected_features.reindex(test_meta.index).to_numpy(float)
        evaluator = CryptoFitnessEvaluator(horizons=horizons)
        meta_fingerprint = hashlib.sha256(
            repr(
                (
                    fingerprint,
                    archive.resolve(),
                    archive.stat().st_size,
                    archive.stat().st_mtime_ns,
                    args.data_1m.resolve(),
                    args.data_1m.stat().st_size,
                    args.data_1m.stat().st_mtime_ns,
                    args.rank,
                    lookahead,
                    "final_val_early_stop_v1",
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        model_path = args.model_dir / f"meta_after_{lookahead}m_rank{args.rank}.txt"
        meta_cache_valid = cached_model_matches(
            kind="meta_rank1_analysis_replay",
            model_path=model_path,
            model_fingerprint=meta_fingerprint,
            lookahead=lookahead,
        )
        if meta_cache_valid and not args.retrain_models:
            booster = lgb.Booster(model_file=str(model_path))
            print(f"Loaded meta model: {model_path}", flush=True)
        else:
            booster = evaluator._train_booster_final(
                x_train,
                train_meta[f"label_h{horizon}"].astype(int),
                x_val,
                val_meta[f"label_h{horizon}"].astype(int),
            )
            booster.save_model(str(model_path))
            print(f"Saved meta model: {model_path}", flush=True)
        val_prediction = pd.Series(booster.predict(x_val), index=val_meta.index)
        test_prediction = pd.Series(booster.predict(x_test), index=test_meta.index)
        booster.free_dataset()
        model_records.append(
            {
                "kind": "meta_rank1_analysis_replay",
                "model_path": model_path.name,
                "fingerprint": meta_fingerprint,
                "archive": str(archive),
                "rank": int(args.rank),
                "lookahead_bars": int(lookahead),
                "horizon": int(horizon),
                "features": list(individual.features),
                "feature_cache_path": feature_cache_path.name,
                "feature_fingerprint": feature_fingerprint,
                "training_scope": "meta_oof_train_with_final_val_early_stopping",
            }
        )

        combined = pd.concat(
            [
                _split_bands(
                    val_prediction, val_meta, horizon, raw, minute, target_interval
                ),
                _split_bands(
                    test_prediction, test_meta, horizon, raw, minute, target_interval
                ),
            ],
            ignore_index=True,
        )
        selected = combined[combined["band"] < 16]
        rejected = combined[combined["band"] >= 16]
        name = f"after_{lookahead}m"
        universe.update(pd.to_datetime(combined["timestamp"]))
        top50_sets[name] = set(
            pd.to_datetime(combined.loc[combined["band"] < 10, "timestamp"])
        )
        band_by_model[name] = pd.Series(
            combined["band"].to_numpy(int),
            index=pd.DatetimeIndex(pd.to_datetime(combined["timestamp"])),
            name=f"{name}_band",
        ).sort_index()
        summary_rows.append(
            {
                "archive": name,
                "all": len(combined),
                "selected": len(selected),
                "precision": selected["hit"].mean(),
                "tp_mean": selected["tp"].mean(),
                "mfe_mean": selected["mfe"].mean(),
                "selected_close_h3_mean": selected["close_h"].mean(),
                "rejected": len(rejected),
                "rejected_close_h3_mean": rejected["close_h"].mean(),
                "rejected_mfe_mean": rejected["mfe"].mean(),
            }
        )
        grouped = combined.groupby("band", sort=True).agg(
            n=("hit", "size"),
            precision=("hit", "mean"),
            tp_mean=("tp", "mean"),
            mfe_mean=("mfe", "mean"),
            close_minute_1_mean=("close_minute_1", "mean"),
            close_h3_mean=("close_h", "mean"),
        )
        grouped.insert(0, "archive", name)
        band_rows.append(grouped.reset_index())
        partial_summary = pd.DataFrame(summary_rows)
        partial_bands = pd.concat(band_rows, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial_summary.to_csv(
            args.out.with_name(args.out.stem + "_summary.partial.csv"), index=False
        )
        partial_bands.to_csv(
            args.out.with_name(args.out.stem + ".partial.csv"), index=False
        )
        print(f"Finished {name}", flush=True)

    names = ["after_1m", "after_2m", "after_3m"]
    membership = pd.DataFrame(
        {
            name: [timestamp in top50_sets[name] for timestamp in sorted(universe)]
            for name in names
        },
        index=pd.DatetimeIndex(sorted(universe), name="timestamp"),
    )
    for name in names:
        membership[f"{name}_band"] = band_by_model[name].reindex(membership.index)
    membership.to_csv(args.out.with_name("meta_rank1_top50_membership.csv"))
    print("TOP50_OVERLAP", flush=True)
    for left, right in ((names[0], names[1]), (names[0], names[2]), (names[1], names[2])):
        intersection = top50_sets[left] & top50_sets[right]
        union = top50_sets[left] | top50_sets[right]
        print(
            f"{left}&{right}: intersection={len(intersection)} "
            f"overlap_each={len(intersection)/min(len(top50_sets[left]), len(top50_sets[right])):.8f} "
            f"jaccard={len(intersection)/len(union):.8f}",
            flush=True,
        )
    all_three = set.intersection(*(top50_sets[name] for name in names))
    print(
        f"all_three: intersection={len(all_three)} "
        f"share_each={len(all_three)/min(len(top50_sets[name]) for name in names):.8f} "
        f"share_universe={len(all_three)/len(universe):.8f}",
        flush=True,
    )
    patterns = membership.value_counts().sort_index()
    for pattern, count in patterns.items():
        key = "".join("1" if value else "0" for value in pattern)
        print(
            f"pattern_{key}={int(count)} share_universe={int(count)/len(universe):.8f}",
            flush=True,
        )

    manifest = {
        "pipeline": "meta_rank1_cached_bundle",
        "data": str(args.data),
        "data_1m": str(args.data_1m),
        "common_oof_cache": str(cache_path),
        "fingerprint": fingerprint,
        "models": model_records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return pd.DataFrame(summary_rows), pd.concat(band_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs=3, type=Path, default=list(ARCHIVES))
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--data", type=Path, default=Path("data/crypto/BTCUSDT_5m.csv"))
    parser.add_argument("--data-1m", type=Path, default=Path("data/crypto/BTCUSDT_1m.csv"))
    parser.add_argument(
        "--out", type=Path, default=Path("temp/output/meta_rank1_band_stats.csv")
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("temp/model/meta_mfe_q20_after_1m_2m_3m"),
    )
    parser.add_argument(
        "--retrain-models",
        action="store_true",
        help="Overwrite and retrain the cached base and three Rank 1 boosters.",
    )
    parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help=(
            "Rebuild the shared meta/OOF data and the compact selected-feature "
            "matrices stored beside the cached models."
        ),
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    summary, bands = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    bands.to_csv(args.out, index=False)
    print("SUMMARY")
    print(summary.to_csv(index=False), end="")
    print("BANDS")
    print(bands.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
