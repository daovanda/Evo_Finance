"""Analyze crypto archive individuals on final val/test splits.

Example:
    python -m crypto.analyze --archive crypto/results/crypto_btc_seed1_12h.json --top 5
    python -m crypto.analyze --archive crypto/results/crypto_btc_seed1_12h.json --rank 1 3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.data import add_binary_labels, load_ohlcv, split_labeled_by_dates
from crypto.evolution import CryptoIndividual
from crypto.expression import CryptoFeatureSpace
from crypto.features import build_feature_frame, selectable_features
from crypto.fitness import _binary_auc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crypto.analyze")


DEFAULT_CHART_DIR = config.RESULTS_DIR / "chart"
DAILY_ROLLING_WINDOW_DAYS = 10


@dataclass(frozen=True)
class SplitPrediction:
    split: str
    data: pd.DataFrame
    selected: pd.DataFrame
    auc: float
    precision_at_trade: float
    base_rate: float
    precision_excess: float
    selected_pred_threshold: float
    selected_mfe_hit_rate: float
    baseline_mfe_hit_rate: float
    selected_mfe_excess_mean: float
    selected_mfe_mean: float
    selected_close_return_mean: float
    selected_trades_per_day: float
    chart_days: int


@dataclass(frozen=True)
class HorizonAnalysis:
    horizon: int
    val: SplitPrediction
    test: SplitPrediction
    label: str | None = None


def analyze(
    archive_path: str | Path,
    data_path: str | Path = config.DATA_PATH,
    output_dir: str | Path = DEFAULT_CHART_DIR,
    top: int | None = None,
    ranks: list[int] | None = None,
    val_start: str = config.VAL_START,
    test_start: str = config.TEST_START,
    test_end: str | None = config.TEST_END,
) -> list[Path]:
    entries = _filter_entries(_load_archive_entries(Path(archive_path)), top=top, ranks=ranks)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading crypto data from %s", data_path)
    raw_df = load_ohlcv(data_path)
    labeled_df = add_binary_labels(
        raw_df,
        horizons=config.HOLDING_HORIZONS,
        threshold=config.LABEL_THRESHOLD,
    )
    purge_bars = config.purge_bars_for_horizons(config.HOLDING_HORIZONS)
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

    logger.info("Building crypto feature matrix; quality filter uses final train rows.")
    feature_df = build_feature_frame(raw_df, quality_index=train_df.index)
    feature_pool = selectable_features(feature_df)
    feature_space = CryptoFeatureSpace(feature_df, feature_pool)
    mfe_by_horizon = {
        int(horizon): _max_high_return(raw_df, int(horizon))
        for horizon in config.HOLDING_HORIZONS
    }
    close_return_by_horizon = {
        int(horizon): _close_exit_return(raw_df, int(horizon))
        for horizon in config.HOLDING_HORIZONS
    }

    charts: list[Path] = []
    for entry in entries:
        individual = _entry_to_individual(entry)
        logger.info(
            "Analyzing rank %s | score=%s | features=%d",
            entry.get("rank", "?"),
            entry.get("score"),
            len(individual.features),
        )
        horizon_results: list[HorizonAnalysis] = []
        for horizon in config.HOLDING_HORIZONS:
            result = _analyze_horizon(
                individual=individual,
                horizon=int(horizon),
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_space=feature_space,
                mfe=mfe_by_horizon[int(horizon)],
                close_return=close_return_by_horizon[int(horizon)],
            )
            if result is not None:
                horizon_results.append(result)

        if not horizon_results:
            logger.warning("Rank %s produced no horizon charts; skipped.", entry.get("rank", "?"))
            continue

        ensemble_result = _build_ensemble_analysis(horizon_results)
        chart_path = _plot_individual(entry, horizon_results, output_path, ensemble_result)
        charts.append(chart_path)
        logger.info("Saved chart: %s", chart_path)

    return charts


def _load_archive_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        raise ValueError(f"Archive has no entries list: {path}")
    logger.info("Loaded %d archive entries from %s", len(entries), path)
    return [dict(entry) for entry in entries]


def _filter_entries(
    entries: list[dict[str, Any]],
    top: int | None = None,
    ranks: list[int] | None = None,
) -> list[dict[str, Any]]:
    if ranks:
        wanted = {int(rank) for rank in ranks}
        selected = [entry for entry in entries if int(entry.get("rank", -1)) in wanted]
        found = {int(entry.get("rank", -1)) for entry in selected}
        missing = sorted(wanted - found)
        if missing:
            logger.warning("Archive does not contain requested rank(s): %s", missing)
        if not selected:
            raise ValueError(f"No archive entries matched rank(s): {sorted(wanted)}")
        return selected
    if top is not None:
        return entries[: int(top)]
    return entries


def _entry_to_individual(entry: dict[str, Any]) -> CryptoIndividual:
    features = entry.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no features list.")
    clean_features: list[str] = []
    for feature in features:
        feature = str(feature).strip()
        if feature and feature not in clean_features:
            clean_features.append(feature)
    if not clean_features:
        raise ValueError(f"Archive rank {entry.get('rank')} has no valid features.")
    return CryptoIndividual(
        features=clean_features,
        generation=int(entry.get("generation", 0) or 0),
        score=float(entry.get("score", float("nan"))),
        metrics=dict(entry.get("metrics", {})),
    )


def _analyze_horizon(
    individual: CryptoIndividual,
    horizon: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_space: CryptoFeatureSpace,
    mfe: pd.Series,
    close_return: pd.Series,
) -> HorizonAnalysis | None:
    label_col = f"label_h{horizon}"
    ret_col = f"future_return_h{horizon}"
    train = _valid_frame(train_df, label_col, ret_col)
    val = _valid_frame(val_df, label_col, ret_col)
    test = _valid_frame(test_df, label_col, ret_col)
    if train.empty or val.empty or test.empty:
        logger.warning("h%d skipped: empty train/val/test after label filtering.", horizon)
        return None

    X_train = feature_space.matrix(individual.features, train.index)
    X_val = feature_space.matrix(individual.features, val.index)
    X_test = feature_space.matrix(individual.features, test.index)
    y_train = train[label_col].astype(int)
    y_val = val[label_col].astype(int)
    y_test = test[label_col].astype(int)

    if y_train.nunique() < 2:
        logger.warning("h%d skipped: train label is constant.", horizon)
        return None

    booster = _train_final_booster(X_train, y_train, X_val, y_val)
    val_pred = pd.Series(booster.predict(X_val), index=val.index, name="pred")
    test_pred = pd.Series(booster.predict(X_test), index=test.index, name="pred")

    val_result = _split_prediction(
        split="val",
        y_true=y_val,
        pred=val_pred,
        close_return=close_return,
        mfe=mfe,
    )
    test_result = _split_prediction(
        split="test",
        y_true=y_test,
        pred=test_pred,
        close_return=close_return,
        mfe=mfe,
    )
    return HorizonAnalysis(horizon=horizon, val=val_result, test=test_result)


def _train_final_booster(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.Booster:
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=-1)]
    valid_sets = None
    if config.LGBM_EARLY_STOPPING > 0 and len(X_val) > 0 and y_val.nunique() >= 2:
        valid_sets = [
            lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_set,
                free_raw_data=False,
            )
        ]
        callbacks.insert(
            0,
            lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False),
        )
    return lgb.train(
        params=dict(config.LGBM_PARAMS),
        train_set=train_set,
        num_boost_round=int(config.LGBM_NUM_BOOST_ROUND),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _split_prediction(
    split: str,
    y_true: pd.Series,
    pred: pd.Series,
    close_return: pd.Series,
    mfe: pd.Series,
) -> SplitPrediction:
    data = (
        pd.DataFrame(
            {
                "label": y_true,
                "pred": pred,
                "close_return": close_return.reindex(pred.index),
                "mfe": mfe.reindex(pred.index),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if data.empty:
        return _empty_split_prediction(split, data)

    n_select = min(
        len(data),
        max(
            int(config.MIN_TRADES_PER_SPLIT),
            int(np.ceil(len(data) * float(config.TRADE_TOP_FRACTION))),
        ),
    )
    selected = data.nlargest(n_select, "pred")
    return _split_prediction_from_data(split=split, data=data, selected=selected)


def _split_prediction_from_data(
    split: str,
    data: pd.DataFrame,
    selected: pd.DataFrame,
) -> SplitPrediction:
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["label", "pred", "close_return", "mfe"]
    )
    if data.empty:
        return _empty_split_prediction(split, data)

    selected = selected.reindex(data.index.intersection(selected.index)).dropna(
        subset=["label", "pred", "close_return", "mfe"]
    )
    chart_days = _unique_calendar_days(data.index)
    base_rate = float(data["label"].mean())
    precision = float(selected["label"].mean()) if len(selected) else 0.0
    selected_pred_threshold = float(selected["pred"].min()) if len(selected) else 0.0
    threshold = float(config.LABEL_THRESHOLD)
    baseline_mfe_hit_rate = _daily_baseline_mfe_hit_rate(data, threshold)
    return SplitPrediction(
        split=split,
        data=data,
        selected=selected,
        auc=_binary_auc(data["label"].astype(int), data["pred"]),
        precision_at_trade=precision,
        base_rate=base_rate,
        precision_excess=precision - base_rate,
        selected_pred_threshold=selected_pred_threshold,
        selected_mfe_hit_rate=(
            float((selected["mfe"] > threshold).mean())
            if len(selected)
            else 0.0
        ),
        baseline_mfe_hit_rate=baseline_mfe_hit_rate,
        selected_mfe_excess_mean=(
            float((selected["mfe"] - threshold).clip(lower=0.0).mean())
            if len(selected)
            else 0.0
        ),
        selected_mfe_mean=float(selected["mfe"].mean()) if len(selected) else 0.0,
        selected_close_return_mean=(
            float(selected["close_return"].mean()) if len(selected) else 0.0
        ),
        selected_trades_per_day=(float(len(selected)) / chart_days if chart_days > 0 else 0.0),
        chart_days=chart_days,
    )


def _empty_split_prediction(split: str, data: pd.DataFrame) -> SplitPrediction:
    return SplitPrediction(
        split=split,
        data=data,
        selected=data.iloc[0:0].copy(),
        auc=0.5,
        precision_at_trade=0.0,
        base_rate=0.0,
        precision_excess=0.0,
        selected_pred_threshold=0.0,
        selected_mfe_hit_rate=0.0,
        baseline_mfe_hit_rate=0.0,
        selected_mfe_excess_mean=0.0,
        selected_mfe_mean=0.0,
        selected_close_return_mean=0.0,
        selected_trades_per_day=0.0,
        chart_days=0,
    )


def _build_ensemble_analysis(
    horizons: list[HorizonAnalysis],
) -> HorizonAnalysis | None:
    if len(horizons) < 2:
        return None
    ordered = sorted(horizons, key=lambda item: int(item.horizon))
    exit_result = ordered[-1]
    horizon_names = "+".join(f"h{item.horizon}" for item in ordered)
    label = f"ensemble {horizon_names} -> close h{exit_result.horizon}"
    return HorizonAnalysis(
        horizon=exit_result.horizon,
        val=_ensemble_split_prediction(
            split="val",
            split_results=[item.val for item in ordered],
            exit_split=exit_result.val,
        ),
        test=_ensemble_split_prediction(
            split="test",
            split_results=[item.test for item in ordered],
            exit_split=exit_result.test,
        ),
        label=label,
    )


def _ensemble_split_prediction(
    split: str,
    split_results: list[SplitPrediction],
    exit_split: SplitPrediction,
) -> SplitPrediction:
    if not split_results or exit_split.data.empty:
        return _empty_split_prediction(split, exit_split.data)

    common_index: pd.Index | None = None
    for result in split_results:
        selected_index = pd.Index(result.selected.index)
        common_index = selected_index if common_index is None else common_index.intersection(
            selected_index
        )
    if common_index is None:
        common_index = pd.Index([])

    data = exit_split.data.copy()
    pred_frame = pd.concat(
        [result.data["pred"].reindex(data.index) for result in split_results],
        axis=1,
    )
    data["pred"] = pred_frame.mean(axis=1)
    data = data.dropna(subset=["pred"])
    selected = data.reindex(data.index.intersection(common_index)).dropna()
    return _split_prediction_from_data(split=split, data=data, selected=selected)


def _unique_calendar_days(index: pd.Index) -> int:
    if len(index) == 0:
        return 0
    dt_index = pd.DatetimeIndex(index)
    return int(dt_index.normalize().nunique())


def _valid_frame(df: pd.DataFrame, label_col: str, ret_col: str) -> pd.DataFrame:
    if label_col not in df.columns or ret_col not in df.columns:
        raise ValueError(f"Missing required columns: {label_col}, {ret_col}")
    return df.dropna(subset=[label_col, ret_col]).copy()


def _max_high_return(raw_df: pd.DataFrame, horizon: int) -> pd.Series:
    data = raw_df.sort_index()
    entry = pd.to_numeric(data["open"], errors="coerce").shift(-1)
    high = pd.to_numeric(data["high"], errors="coerce")
    max_high = pd.concat(
        [high.shift(-offset) for offset in range(1, int(horizon) + 1)],
        axis=1,
    ).max(axis=1, skipna=False)
    return (max_high / entry - 1.0).replace([np.inf, -np.inf], np.nan)


def _close_exit_return(raw_df: pd.DataFrame, horizon: int) -> pd.Series:
    data = raw_df.sort_index()
    entry = pd.to_numeric(data["open"], errors="coerce").shift(-1)
    close = pd.to_numeric(data["close"], errors="coerce").shift(-int(horizon))
    return (close / entry - 1.0).replace([np.inf, -np.inf], np.nan)


def _plot_individual(
    entry: dict[str, Any],
    horizons: list[HorizonAnalysis],
    output_dir: Path,
    ensemble: HorizonAnalysis | None = None,
) -> Path:
    rank = int(entry.get("rank", 0) or 0)
    score = float(entry.get("score", float("nan")))
    sections = list(horizons)
    if ensemble is not None:
        sections.append(ensemble)
    nrows = len(sections) * 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(16, max(7.0, 4.7 * len(sections))),
        sharex=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.95, 2.8] * len(sections)},
    )
    if nrows == 1:
        axes = [axes]
    axes = list(np.ravel(axes))

    for idx, result in enumerate(sections):
        _plot_metrics_table(axes[idx * 2], result)
        _plot_daily_axis(axes[idx * 2 + 1], result)

    feature_count = int(entry.get("n_features", len(entry.get("features", []))) or 0)
    fig.suptitle(
        f"Crypto rank {rank:02d} | score={score:.4f} | features={feature_count} | "
        f"top={config.TRADE_TOP_FRACTION:.0%}",
        fontsize=13,
    )
    filename = f"rank_{rank:02d}_score_{score:.4f}_summary.png"
    path = output_dir / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_metrics_table(ax: plt.Axes, result: HorizonAnalysis) -> None:
    ax.axis("off")
    threshold_pct = float(config.LABEL_THRESHOLD) * 100.0
    columns = [
        "Split",
        "AUC",
        "Base",
        "Top precision",
        "PE",
        "Threshold",
        f"MFE>{threshold_pct:.2f}%",
        "Base MFE",
        "Excess",
        "Top MFE",
        "Close",
        "Trades",
        "Trades/day",
        "Days",
    ]
    rows = [_metrics_table_row("val", result.val), _metrics_table_row("test", result.test)]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#222831")
            cell.set_text_props(color="white", weight="bold")
        elif row == 1:
            cell.set_facecolor("#eef5ff")
        elif row == 2:
            cell.set_facecolor("#fff4e6")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")
    ax.set_title(
        f"{_analysis_label(result)} final val/test metrics",
        loc="left",
        fontsize=10,
        pad=2,
    )


def _metrics_table_row(split_name: str, split: SplitPrediction) -> list[str]:
    return [
        split_name,
        f"{split.auc:.3f}",
        _fmt_pct(split.base_rate),
        _fmt_pct(split.precision_at_trade),
        _fmt_pct(split.precision_excess, signed=True),
        f"{split.selected_pred_threshold:.6f}",
        _fmt_pct(split.selected_mfe_hit_rate),
        _fmt_pct(split.baseline_mfe_hit_rate),
        _fmt_pct(split.selected_mfe_excess_mean),
        _fmt_pct(split.selected_mfe_mean),
        _fmt_pct(split.selected_close_return_mean, signed=True),
        f"{len(split.selected):,}",
        f"{split.selected_trades_per_day:.1f}",
        f"{split.chart_days:,}",
    ]


def _fmt_pct(value: float, signed: bool = False) -> str:
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _plot_daily_axis(ax: plt.Axes, result: HorizonAnalysis) -> None:
    threshold_pct = float(config.LABEL_THRESHOLD) * 100.0
    label_prefix = _analysis_label(result)
    twin = ax.twinx()
    for split_result, color, label in [
        (result.val, "#1f77b4", "val"),
        (result.test, "#ff7f0e", "test"),
    ]:
        daily = _daily_hit_stats(split_result)
        if daily.empty:
            continue
        ax.plot(
            daily.index,
            daily["selected_mfe_hit_rate"] * 100.0,
            color=color,
            marker="o",
            markersize=2.5,
            linewidth=0.9,
            alpha=0.90,
            label=f"{label} top MFE hit",
        )
        ax.plot(
            daily.index,
            daily["baseline_mfe_hit_rate"] * 100.0,
            color=color,
            linestyle="--",
            marker=".",
            markersize=2.0,
            linewidth=0.8,
            alpha=0.45,
            label=f"{label} baseline",
        )
        twin.plot(
            daily.index,
            daily["trade_count"],
            color=color,
            linestyle=":",
            marker="x",
            markersize=2.5,
            linewidth=0.8,
            alpha=0.70,
            label=f"{label} trades",
        )

    ax.set_ylabel(f"{label_prefix} hit rate %")
    twin.set_ylabel("trades/day")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"{label_prefix} daily MFE>{threshold_pct:.2f}% rate vs baseline and trade count",
        loc="left",
        fontsize=9,
    )
    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="upper right", fontsize=8, ncol=3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _daily_hit_stats(split_result: SplitPrediction) -> pd.DataFrame:
    if split_result.data.empty:
        return pd.DataFrame(
            columns=["selected_mfe_hit_rate", "baseline_mfe_hit_rate", "trade_count"]
        )

    threshold = float(config.LABEL_THRESHOLD)
    data = split_result.data.sort_index()
    data_days = pd.DatetimeIndex(data.index).normalize()
    baseline_rate = (data["mfe"] > threshold).groupby(data_days).mean()
    daily = pd.DataFrame({"baseline_mfe_hit_rate": baseline_rate})

    selected = split_result.selected.sort_index()
    if selected.empty:
        daily["selected_mfe_hit_rate"] = np.nan
        daily["trade_count"] = 0
        return daily

    selected_days = pd.DatetimeIndex(selected.index).normalize()
    selected_rate = (selected["mfe"] > threshold).groupby(selected_days).mean()
    trade_count = selected.groupby(selected_days).size()
    daily["selected_mfe_hit_rate"] = selected_rate.reindex(daily.index)
    daily["trade_count"] = trade_count.reindex(daily.index).fillna(0).astype(int)
    daily = daily.rolling(
        window=int(DAILY_ROLLING_WINDOW_DAYS),
        min_periods=1,
    ).mean()
    return daily


def _daily_baseline_mfe_hit_rate(data: pd.DataFrame, threshold: float) -> float:
    if data.empty:
        return 0.0
    days = pd.DatetimeIndex(data.index).normalize()
    daily_rate = (data["mfe"] > threshold).groupby(days).mean()
    return float(daily_rate.mean()) if len(daily_rate) else 0.0


def _analysis_label(result: HorizonAnalysis) -> str:
    return result.label or f"h{result.horizon}"


def _parse_ranks(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    return [int(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Crypto archive JSON path.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="Crypto OHLCV CSV path.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_CHART_DIR),
        help=f"Chart output directory. Default: {DEFAULT_CHART_DIR}",
    )
    parser.add_argument("--top", type=int, default=None, help="Analyze only top N entries.")
    parser.add_argument(
        "--rank",
        nargs="+",
        default=None,
        help="Analyze specific archive rank(s), for example --rank 1 3 10.",
    )
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    args = parser.parse_args()

    charts = analyze(
        archive_path=args.archive,
        data_path=args.data,
        output_dir=args.out_dir,
        top=args.top,
        ranks=_parse_ranks(args.rank),
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    logger.info("Done. Saved %d chart(s).", len(charts))


if __name__ == "__main__":
    main()
