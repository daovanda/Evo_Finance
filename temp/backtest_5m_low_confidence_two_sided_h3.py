"""Backtest two-sided trades outside both Long and Short top fractions.

Signal flow:
1. Train/load one Long MFE H3 model and one Short MFE H3 model.
2. Learn each top-fraction threshold on Final Val and apply it to Test.
3. Select the complement of their union:
       NOT (Long top fraction OR Short top fraction)
4. Open one Long and one Short at open H1.
5. Keep independent TP/SL barriers active through H1, H2, and H3.
6. Close every unfilled leg at close H3.

Trade cost is specified per leg, so each two-sided signal pays twice the
configured value. OHLC does not reveal whether High or Low occurred first.
The default ``worst_case`` evaluates all H1-H3 High/Low order combinations
and uses the least favorable feasible result for each signal.

PowerShell:
    python -m temp.backtest_5m_low_confidence_two_sided_h3 `
      --long-archive crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json `
      --short-archive crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json `
      --rank 1 `
      --top-fraction 0.40 `
      --take-profit 0.0005 `
      --stop-loss 0.002 `
      --trade-cost-per-leg 0.00016 `
      --same-candle-policy worst_case `
      --data data/crypto/BTCUSDT_5m.csv `
      --out-dir temp/output
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto import config
from crypto.analyze import _required_windows_for_entries
from crypto.backtest import (
    BundleSignals,
    ModelSpec,
    _archive_horizons,
    _cached_feature_space,
    _load_rank_entry,
    _quality_train_index,
    _train_spec_bundle,
)
from crypto.data import load_ohlcv
from temp.backtest_5m_long_mfe_fixed_tp_sl_h3 import (
    _bundle_with_top_fraction,
    _load_trained_bundle_cache,
    _save_trained_bundle_cache,
    _trained_bundle_cache_path,
    load_archive_metadata,
    make_price_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(
    "temp.backtest_5m_low_confidence_two_sided_h3"
)

DEFAULT_LONG_ARCHIVE = Path(
    "crypto/results/crypto_btc_5m_long_mfe_h3_tp01_top40_seed1_8h.json"
)
DEFAULT_SHORT_ARCHIVE = Path(
    "crypto/results/crypto_btc_short_mfe_h3_tp01_top40_seed1_8h.json"
)
DEFAULT_DATA = Path("data/crypto/BTCUSDT_5m.csv")
DEFAULT_OUT_DIR = Path("temp/output")
DEFAULT_MODEL_CACHE_DIR = Path("temp/model")


@dataclass(frozen=True)
class ModelInput:
    name: str
    archive_path: Path
    direction: str
    spec: ModelSpec
    horizons: list[int]
    entry: dict[str, Any]
    cache_path: Path
    legacy_cache_path: Path


def _prepare_model_input(
    *,
    name: str,
    archive_path: Path,
    direction: str,
    rank: int,
    top_fraction: float,
    data_path: Path,
    raw_df: pd.DataFrame,
    cache_dir: Path,
    val_start: str,
    test_start: str,
    test_end: str | None,
) -> ModelInput:
    metadata = load_archive_metadata(archive_path)
    mode = config.canonical_label_mode(metadata.get("label_mode"))
    archive_direction = config.canonical_label_direction(
        metadata.get("label_direction")
    )
    expected_direction = config.canonical_label_direction(direction)
    horizons = _archive_horizons(
        archive_path,
        fallback=[3],
        label=f"5m {expected_direction.title()} MFE complement",
    )
    if (
        mode != "mfe"
        or archive_direction != expected_direction
        or horizons != [3]
    ):
        raise ValueError(
            f"{name} archive must use mode=mfe, "
            f"direction={expected_direction}, horizons=[3]; got "
            f"mode={mode}, direction={archive_direction}, horizons={horizons}."
        )
    spec = ModelSpec(
        archive_path=archive_path,
        rank=int(rank),
        label_mode=mode,
        label_threshold=float(metadata["label_threshold"]),
        top_fraction=float(top_fraction),
        label_direction=archive_direction,
    )
    entry = _load_rank_entry(archive_path, spec.rank)
    purge_bars = config.purge_bars_for_horizons(horizons)
    cache_kwargs = {
        "cache_dir": cache_dir,
        "archive_path": archive_path,
        "data_path": data_path,
        "raw_df": raw_df,
        "spec": spec,
        "horizons": horizons,
        "val_start": val_start,
        "test_start": test_start,
        "test_end": test_end,
        "purge_bars": purge_bars,
    }
    return ModelInput(
        name=name,
        archive_path=archive_path,
        direction=archive_direction,
        spec=spec,
        horizons=horizons,
        entry=entry,
        cache_path=_trained_bundle_cache_path(**cache_kwargs),
        legacy_cache_path=_trained_bundle_cache_path(
            **cache_kwargs,
            include_top_fraction=True,
        ),
    )


def _load_or_train_bundles(
    *,
    models: list[ModelInput],
    raw_df: pd.DataFrame,
    data_path: Path,
    val_start: str,
    test_start: str,
    test_end: str | None,
    use_cache: bool,
    rebuild_cache: bool,
) -> dict[str, BundleSignals]:
    bundles: dict[str, BundleSignals] = {}
    missing: list[ModelInput] = []
    for model in models:
        bundle = None
        if use_cache and not rebuild_cache:
            bundle = _load_trained_bundle_cache(model.cache_path)
            if bundle is None:
                bundle = _load_trained_bundle_cache(model.legacy_cache_path)
                if bundle is not None:
                    _save_trained_bundle_cache(model.cache_path, bundle)
        if bundle is None:
            missing.append(model)
        else:
            bundles[model.name] = bundle

    if missing:
        purge_bars = config.purge_bars_for_horizons([3])
        quality_index = _quality_train_index(
            raw_df=raw_df,
            spec=missing[0].spec,
            horizons=[3],
            val_start=val_start,
            test_start=test_start,
            test_end=test_end,
            purge_bars=purge_bars,
        )
        feature_space = _cached_feature_space(
            raw_df=raw_df,
            data_path=data_path,
            required_windows=_required_windows_for_entries(
                [model.entry for model in missing]
            ),
            quality_index=quality_index,
        )
        for model in missing:
            logger.info("Training missing %s model bundle.", model.name)
            bundle = _train_spec_bundle(
                spec=model.spec,
                entry=model.entry,
                raw_df=raw_df,
                feature_space=feature_space,
                horizons=model.horizons,
                val_start=val_start,
                test_start=test_start,
                test_end=test_end,
                purge_bars=purge_bars,
            )
            bundles[model.name] = bundle
            if use_cache:
                _save_trained_bundle_cache(model.cache_path, bundle)

    return {
        model.name: _bundle_with_top_fraction(
            bundles[model.name],
            model.spec.top_fraction,
        )
        for model in models
    }


def _complement_index(
    long_bundle: BundleSignals,
    short_bundle: BundleSignals,
    split: str,
) -> tuple[pd.Index, dict[str, int]]:
    long_signals = getattr(long_bundle, split)
    short_signals = getattr(short_bundle, split)
    universe = long_signals.data.index.intersection(short_signals.data.index)
    long_top = universe.intersection(long_signals.selected_index)
    short_top = universe.intersection(short_signals.selected_index)
    union = long_top.union(short_top)
    complement = universe.difference(union)
    return complement, {
        "universe": len(universe),
        "long_top": len(long_top),
        "short_top": len(short_top),
        "union": len(union),
        "complement": len(complement),
    }


def _simulate_intrabar_order(
    selected_path: pd.DataFrame,
    *,
    take_profit: float,
    stop_loss: float,
    high_first_by_step: tuple[bool, bool, bool],
) -> pd.DataFrame:
    result = selected_path.copy()
    n = len(result)
    long_active = np.ones(n, dtype=bool)
    short_active = np.ones(n, dtype=bool)
    long_return = np.full(n, np.nan, dtype=float)
    short_return = np.full(n, np.nan, dtype=float)
    long_outcome = np.full(n, "close_h3", dtype=object)
    short_outcome = np.full(n, "close_h3", dtype=object)

    def high_event(step: int, high: np.ndarray) -> None:
        long_tp = long_active & (high >= take_profit)
        short_sl = short_active & (high >= stop_loss)
        long_return[long_tp] = take_profit
        short_return[short_sl] = -stop_loss
        long_outcome[long_tp] = f"tp_h{step}"
        short_outcome[short_sl] = f"sl_h{step}"
        long_active[long_tp] = False
        short_active[short_sl] = False

    def low_event(step: int, low: np.ndarray) -> None:
        long_sl = long_active & (low <= -stop_loss)
        short_tp = short_active & (low <= -take_profit)
        long_return[long_sl] = -stop_loss
        short_return[short_tp] = take_profit
        long_outcome[long_sl] = f"sl_h{step}"
        short_outcome[short_tp] = f"tp_h{step}"
        long_active[long_sl] = False
        short_active[short_tp] = False

    for step, high_first in enumerate(high_first_by_step, start=1):
        high = pd.to_numeric(
            result[f"high_h{step}"], errors="coerce"
        ).to_numpy(dtype=float)
        low = pd.to_numeric(
            result[f"low_h{step}"], errors="coerce"
        ).to_numpy(dtype=float)
        events = (high_event, low_event) if high_first else (low_event, high_event)
        events[0](step, high if high_first else low)
        events[1](step, low if high_first else high)

    close_h3 = pd.to_numeric(
        result["close_h3"], errors="coerce"
    ).to_numpy(dtype=float)
    long_return[long_active] = close_h3[long_active]
    short_return[short_active] = -close_h3[short_active]
    result["long_outcome"] = long_outcome
    result["short_outcome"] = short_outcome
    result["long_return"] = long_return
    result["short_return"] = short_return
    result["gross_return"] = long_return + short_return
    return result


def simulate_two_sided(
    selected_path: pd.DataFrame,
    *,
    take_profit: float,
    stop_loss: float,
    trade_cost_per_leg: float,
    same_candle_policy: str,
) -> pd.DataFrame:
    required = [
        *(f"{side}_h{step}" for step in range(1, 4) for side in ("high", "low")),
        "close_h3",
    ]
    clean = selected_path.dropna(subset=required).copy()
    tp = float(take_profit)
    sl = float(stop_loss)
    cost = float(trade_cost_per_leg)
    if tp <= 0.0 or sl <= 0.0:
        raise ValueError("take_profit and stop_loss must be positive.")
    if cost < 0.0:
        raise ValueError("trade_cost_per_leg cannot be negative.")

    policy = str(same_candle_policy).strip().lower()
    if policy == "high_first":
        candidates = [
            _simulate_intrabar_order(
                clean,
                take_profit=tp,
                stop_loss=sl,
                high_first_by_step=(True, True, True),
            )
        ]
    elif policy == "low_first":
        candidates = [
            _simulate_intrabar_order(
                clean,
                take_profit=tp,
                stop_loss=sl,
                high_first_by_step=(False, False, False),
            )
        ]
    elif policy == "worst_case":
        candidates = [
            _simulate_intrabar_order(
                clean,
                take_profit=tp,
                stop_loss=sl,
                high_first_by_step=tuple(
                    bool(mask & (1 << step)) for step in range(3)
                ),
            )
            for mask in range(8)
        ]
    else:
        raise ValueError(
            "same_candle_policy must be worst_case, high_first, or low_first."
        )

    if len(candidates) == 1:
        result = candidates[0]
    else:
        gross_matrix = np.column_stack(
            [candidate["gross_return"].to_numpy() for candidate in candidates]
        )
        selected_candidate = np.argmin(gross_matrix, axis=1)
        result = candidates[0].copy()
        for column in (
            "long_outcome",
            "short_outcome",
            "long_return",
            "short_return",
            "gross_return",
        ):
            values = np.column_stack(
                [candidate[column].to_numpy() for candidate in candidates]
            )
            result[column] = values[np.arange(len(result)), selected_candidate]
        result["worst_path"] = selected_candidate

    result["net_return"] = result["gross_return"] - (2.0 * cost)
    return result


def _summary_row(
    *,
    split: str,
    simulation: pd.DataFrame,
    counts: dict[str, int],
) -> dict[str, Any]:
    n = len(simulation)
    long_tp = simulation["long_outcome"].str.startswith("tp")
    short_tp = simulation["short_outcome"].str.startswith("tp")
    long_sl = simulation["long_outcome"].str.startswith("sl")
    short_sl = simulation["short_outcome"].str.startswith("sl")
    both_tp = long_tp & short_tp
    one_tp = long_tp ^ short_tp
    neither_tp = ~(long_tp | short_tp)
    days = max(
        (simulation.index.max() - simulation.index.min()).total_seconds()
        / 86400.0
        if n > 1
        else 0.0,
        1.0,
    )
    return {
        "split": split,
        **counts,
        "complement_rate": (
            counts["complement"] / counts["universe"]
            if counts["universe"]
            else 0.0
        ),
        "both_tp": int(both_tp.sum()),
        "one_tp": int(one_tp.sum()),
        "neither_tp": int(neither_tp.sum()),
        "long_sl": int(long_sl.sum()),
        "short_sl": int(short_sl.sum()),
        "gross_mean": float(simulation["gross_return"].mean()) if n else np.nan,
        "net_mean": float(simulation["net_return"].mean()) if n else np.nan,
        "win_rate": float((simulation["net_return"] > 0.0).mean()) if n else np.nan,
        "signals_per_day": n / days,
    }


def display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        n = int(row["complement"])

        def count_pct(column: str) -> str:
            count = int(row[column])
            return f"{count:,} ({count / n:.2%})" if n else "0 (n/a)"

        rows.append(
            {
                "split": row["split"],
                "universe": f"{int(row['universe']):,}",
                "Long top": f"{int(row['long_top']):,}",
                "Short top": f"{int(row['short_top']):,}",
                "union": f"{int(row['union']):,}",
                "complement": (
                    f"{n:,} ({float(row['complement_rate']):.2%})"
                ),
                "both TP": count_pct("both_tp"),
                "one TP": count_pct("one_tp"),
                "neither TP": count_pct("neither_tp"),
                "Long SL": count_pct("long_sl"),
                "Short SL": count_pct("short_sl"),
                "gross mean": f"{float(row['gross_mean']):+.3%}",
                "E[net]": f"{float(row['net_mean']):+.3%}",
                "win rate": f"{float(row['win_rate']):.2%}",
                "signals/day": f"{float(row['signals_per_day']):.1f}",
            }
        )
    return pd.DataFrame(rows)


def _plot_report(
    *,
    summary: pd.DataFrame,
    simulations: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(22, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.4, 1.2])
    table_axis = fig.add_subplot(grid[0])
    equity_axis = fig.add_subplot(grid[1])
    count_axis = fig.add_subplot(grid[2])

    table_axis.axis("off")
    shown = display_table(summary)
    table = table_axis.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#9ca3af")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")

    colors = {"val": "#2563eb", "test": "#dc2626"}
    for split, simulation in simulations.items():
        ordered = simulation.sort_index()
        cumulative = ordered["net_return"].cumsum() * 100.0
        equity_axis.plot(
            ordered.index,
            cumulative,
            linewidth=1.1,
            color=colors[split],
            label=(
                f"{split.upper()} n={len(ordered):,} | "
                f"end={float(cumulative.iloc[-1]):+.2f}%"
            ),
        )
    equity_axis.axhline(0.0, color="#4b5563", linestyle="--", linewidth=0.8)
    equity_axis.set_title("Cumulative two-leg net return")
    equity_axis.set_ylabel("Percentage points")
    equity_axis.grid(True, alpha=0.4)
    equity_axis.legend(frameon=False)

    combined = pd.concat(simulations.values()).sort_index()
    daily = pd.Series(1, index=combined.index).resample("D").sum()
    count_axis.bar(daily.index, daily.to_numpy(), width=0.9, color="#f59e0b")
    count_axis.set_title("Complement signals per day")
    count_axis.set_ylabel("Signals")
    count_axis.grid(True, axis="y", alpha=0.4)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _fraction_token(value: float) -> str:
    return f"{float(value) * 100.0:.0f}"


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    long_archive = Path(args.long_archive)
    short_archive = Path(args.short_archive)
    data_path = Path(args.data)
    raw_df = load_ohlcv(data_path)
    cache_dir = Path(args.model_cache_dir)
    model_kwargs = {
        "rank": int(args.rank),
        "top_fraction": float(args.top_fraction),
        "data_path": data_path,
        "raw_df": raw_df,
        "cache_dir": cache_dir,
        "val_start": args.val_start,
        "test_start": args.test_start,
        "test_end": args.test_end,
    }
    models = [
        _prepare_model_input(
            name="long",
            archive_path=long_archive,
            direction="long",
            **model_kwargs,
        ),
        _prepare_model_input(
            name="short",
            archive_path=short_archive,
            direction="short",
            **model_kwargs,
        ),
    ]
    bundles = _load_or_train_bundles(
        models=models,
        raw_df=raw_df,
        data_path=data_path,
        val_start=args.val_start,
        test_start=args.test_start,
        test_end=args.test_end,
        use_cache=not args.no_model_cache,
        rebuild_cache=args.rebuild_model_cache,
    )
    price_path = make_price_path(raw_df, horizon=3, direction="long")
    simulations: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for split in ("val", "test"):
        complement, counts = _complement_index(
            bundles["long"],
            bundles["short"],
            split,
        )
        selected_path = price_path.reindex(complement)
        simulation = simulate_two_sided(
            selected_path,
            take_profit=float(args.take_profit),
            stop_loss=float(args.stop_loss),
            trade_cost_per_leg=float(args.trade_cost_per_leg),
            same_candle_policy=args.same_candle_policy,
        )
        simulations[split] = simulation
        summary_rows.append(
            _summary_row(
                split=split,
                simulation=simulation,
                counts=counts,
            )
        )
    summary = pd.DataFrame(summary_rows)

    output_name = (
        f"low_confidence_two_sided_h3_top{_fraction_token(args.top_fraction)}"
        f"_tp{float(args.take_profit) * 100.0:.3f}pct"
        f"_sl{float(args.stop_loss) * 100.0:.3f}pct"
        f"_{args.same_candle_policy}.png"
    )
    output_path = Path(args.out_dir) / output_name
    _plot_report(
        summary=summary,
        simulations=simulations,
        output_path=output_path,
        title=(
            "5m H3 complement of Long/Short top fractions | "
            f"top={float(args.top_fraction):.0%} | "
            f"TP={float(args.take_profit):.3%} per leg | "
            f"SL={float(args.stop_loss):.3%} per leg | "
            f"cost={float(args.trade_cost_per_leg):.3%} per leg | "
            f"intrabar={args.same_candle_policy}"
        ),
    )
    return summary, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-archive", default=str(DEFAULT_LONG_ARCHIVE))
    parser.add_argument("--short-archive", default=str(DEFAULT_SHORT_ARCHIVE))
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.40)
    parser.add_argument("--take-profit", type=float, default=0.0005)
    parser.add_argument("--stop-loss", type=float, default=0.002)
    parser.add_argument(
        "--trade-cost-per-leg",
        type=float,
        default=0.00016,
    )
    parser.add_argument(
        "--same-candle-policy",
        choices=("worst_case", "high_first", "low_first"),
        default="worst_case",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--val-start", default=config.VAL_START)
    parser.add_argument("--test-start", default=config.TEST_START)
    parser.add_argument("--test-end", default=config.TEST_END)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
    )
    parser.add_argument("--no-model-cache", action="store_true")
    parser.add_argument("--rebuild-model-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary, output_path = run(parse_args())
    print("\n=== Low-confidence two-sided H3 strategy ===")
    print(display_table(summary).to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
