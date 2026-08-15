"""Measure overlap between Rank-1 MFE Long and MAE Short selections."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.data import add_binary_labels, load_ohlcv, make_walk_forward_folds, split_labeled_by_dates
from crypto.evolution import CryptoIndividual
from crypto.features import build_feature_frame, selectable_features
from crypto.quantile_fitness import QuantileFitnessEvaluator
from temp.backtest_quantile_mfe_dynamic_tp import _load_archive, _split_policy


def _fit_prediction(entry, target, horizon, train, val, test, feature_space):
    individual = CryptoIndividual(
        features=[str(value) for value in entry["features"]],
        generation=int(entry.get("generation", 0) or 0),
        score=float(entry.get("score", np.nan)),
    )
    evaluator = QuantileFitnessEvaluator(horizons=[horizon], target=target, quantile=0.2)
    valid_train = evaluator._valid_frame(train, horizon)
    valid_val = evaluator._valid_frame(val, horizon)
    valid_test = evaluator._valid_frame(test, horizon)
    _, val_pred, test_pred = evaluator._fit_predict(
        individual, feature_space, horizon, valid_train, valid_val, valid_test
    )
    return valid_val.index, val_pred, valid_test.index, test_pred


def _row(split, universe, long_selected, short_selected):
    total = len(universe)
    both = long_selected & short_selected
    only_long = long_selected - short_selected
    only_short = short_selected - long_selected
    neither = universe - long_selected - short_selected
    return {
        "split": split,
        "n": total,
        "both": len(both),
        "both_pct": len(both) / total if total else np.nan,
        "only_long": len(only_long),
        "only_long_pct": len(only_long) / total if total else np.nan,
        "only_short": len(only_short),
        "only_short_pct": len(only_short) / total if total else np.nan,
        "neither": len(neither),
        "neither_pct": len(neither) / total if total else np.nan,
        "long_any": len(long_selected),
        "short_any": len(short_selected),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--long-archive", type=Path, required=True)
    p.add_argument("--short-archive", type=Path, required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--min-prediction", type=float, default=0.0002)
    p.add_argument("--data", type=Path, default=Path("data/crypto/BTCUSDT_5m.csv"))
    args = p.parse_args()

    long_meta, long_entry = _load_archive(args.long_archive, args.rank)
    short_meta, short_entry = _load_archive(args.short_archive, args.rank)
    for meta, target, label in ((long_meta, "mfe", "long"), (short_meta, "mae", "short")):
        if config.canonical_label_mode(meta.get("label_mode")) != "quantile_trade":
            raise ValueError(f"{label} archive is not quantile_trade")
        if config.canonical_quantile_target(meta.get("quantile_target")) != target:
            raise ValueError(f"{label} archive target must be {target}")

    long_h = [int(x) for x in long_meta["horizons"]]
    short_h = [int(x) for x in short_meta["horizons"]]
    if long_h != short_h or len(long_h) != 1:
        raise ValueError(f"Archives must share one horizon: long={long_h}, short={short_h}")
    horizon = long_h[0]
    long_policy = _split_policy(long_meta)
    short_policy = _split_policy(short_meta)
    if {long_policy[k] for k in ("val_start", "test_start", "test_end")} != {
        short_policy[k] for k in ("val_start", "test_start", "test_end")
    }:
        raise ValueError("Archives must use the same split policy")

    raw = load_ohlcv(args.data)
    labeled = add_binary_labels(raw, horizons=[horizon], label_mode="quantile_trade")
    purge = config.purge_bars_for_horizons([horizon])
    train, val, test = split_labeled_by_dates(
        labeled, val_start=long_policy["val_start"], test_start=long_policy["test_start"],
        test_end=long_policy["test_end"], purge_bars=purge
    )
    wf_source = labeled.loc[labeled.index < pd.Timestamp(long_policy["wf_end"])]
    folds = make_walk_forward_folds(
        wf_source, wf_end=long_policy["wf_end"],
        min_train_months=long_policy["wf_min_train_months"],
        val_months=long_policy["wf_val_months"],
        step_months=long_policy["wf_step_months"], purge_bars=purge
    )
    windows = _required_windows_for_entries([long_entry, short_entry])
    feature_frame = build_feature_frame(raw, windows=windows, quality_index=folds[0].train_df.index)
    feature_space = __import__("crypto.expression", fromlist=["CryptoFeatureSpace"]).CryptoFeatureSpace(
        feature_frame, selectable_features(feature_frame)
    )
    lv_idx, lv_pred, lt_idx, lt_pred = _fit_prediction(long_entry, "mfe", horizon, train, val, test, feature_space)
    sv_idx, sv_pred, st_idx, st_pred = _fit_prediction(short_entry, "mae", horizon, train, val, test, feature_space)

    rows = []
    combined_sets = {"both": set(), "only_long": set(), "only_short": set(), "neither": set()}
    for split, li, lp, si, sp in (("val", lv_idx, lv_pred, sv_idx, sv_pred), ("test", lt_idx, lt_pred, st_idx, st_pred)):
        li = pd.Index(li).intersection(si)
        lp = pd.Series(lp, index=lv_idx if split == "val" else lt_idx).reindex(li)
        sp = pd.Series(sp, index=sv_idx if split == "val" else st_idx).reindex(li)
        long_set = set(li[lp.ge(args.min_prediction)])
        short_set = set(li[sp.ge(args.min_prediction)])
        row = _row(split, set(li), long_set, short_set)
        rows.append(row)
        for key, value in (("both", long_set & short_set), ("only_long", long_set - short_set),
                           ("only_short", short_set - long_set), ("neither", set(li) - long_set - short_set)):
            combined_sets[key].update(value)
    universe = set().union(
        *(set(pd.Index(lv_idx if split == "val" else lt_idx)) for split in ("val", "test"))
    )
    # Recompute combined counts directly so percentages use the global sample.
    rows.append(_row("all", universe, combined_sets["both"] | combined_sets["only_long"],
                     combined_sets["both"] | combined_sets["only_short"]))
    result = pd.DataFrame(rows)
    for col in ("both_pct", "only_long_pct", "only_short_pct", "neither_pct"):
        result[col] = result[col].map(lambda x: f"{100*x:.2f}%")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
