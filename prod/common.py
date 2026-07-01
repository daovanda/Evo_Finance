"""Shared production utilities for prediction and backtesting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mutator.gene import Gene, Individual


@dataclass(frozen=True)
class IndividualSpec:
    archive: str
    rank: int


@dataclass(frozen=True)
class SelectedIndividual:
    model_id: str
    archive_path: Path
    archive_name: str
    rank: int
    individual: Individual


def parse_individual_specs(raw_specs: Iterable[dict | Sequence]) -> list[IndividualSpec]:
    specs: list[IndividualSpec] = []
    for raw in raw_specs:
        if isinstance(raw, dict):
            archive = raw.get("archive")
            rank = raw.get("rank")
        else:
            archive, rank = raw
        if archive is None or rank is None:
            raise ValueError(f"Invalid individual spec: {raw!r}")
        specs.append(IndividualSpec(archive=str(archive), rank=int(rank)))
    if not specs:
        raise ValueError("No production individuals configured.")
    return specs


def resolve_archive_path(archive: str | Path, results_dir: str | Path) -> Path:
    path = Path(archive)
    if path.is_absolute():
        return path
    results_dir = Path(results_dir)
    if len(path.parts) > 0 and path.parts[0].lower() == "results":
        return results_dir.parent / path
    return results_dir / path


def archive_row_to_individual(row: dict) -> Individual:
    formulas = row.get("genes")
    if not isinstance(formulas, list) or not formulas:
        raise ValueError("archive row has no genes list")

    clean_formulas: list[str] = []
    for formula in formulas:
        clean = str(formula).strip()
        if clean and clean not in clean_formulas:
            clean_formulas.append(clean)
    if not clean_formulas:
        raise ValueError("archive row has no valid gene formulas")

    return Individual(
        genes=[Gene(formula=formula) for formula in clean_formulas],
        generation=int(row.get("generation") or 0),
    )


def load_selected_individuals(
    specs: Iterable[IndividualSpec | dict | Sequence],
    results_dir: str | Path,
) -> list[SelectedIndividual]:
    parsed_specs = [
        spec if isinstance(spec, IndividualSpec) else parse_individual_specs([spec])[0]
        for spec in specs
    ]
    selected: list[SelectedIndividual] = []

    for spec in parsed_specs:
        archive_path = resolve_archive_path(spec.archive, results_dir)
        with open(archive_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Archive must be a JSON list: {archive_path}")

        row = next((item for item in rows if int(item.get("rank", -1)) == spec.rank), None)
        if row is None:
            raise ValueError(f"Rank {spec.rank} not found in {archive_path}")

        archive_name = archive_path.name
        selected.append(
            SelectedIndividual(
                model_id=f"{archive_path.stem}_rank{spec.rank:02d}",
                archive_path=archive_path,
                archive_name=archive_name,
                rank=spec.rank,
                individual=archive_row_to_individual(row),
            )
        )

    return selected


def prediction_frame_for_model(
    selected: SelectedIndividual,
    pred: pd.Series,
    split: str,
) -> pd.DataFrame:
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError("Prediction series must have MultiIndex(date, ticker).")

    frame = pred.rename("pred").reset_index()
    frame = frame.rename(columns={"date": "signal_date"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["model_id"] = selected.model_id
    frame["archive"] = selected.archive_name
    frame["archive_rank"] = selected.rank
    frame["split"] = split
    frame["model_rank"] = (
        frame.groupby("signal_date")["pred"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return frame[
        [
            "signal_date",
            "ticker",
            "model_id",
            "archive",
            "archive_rank",
            "split",
            "pred",
            "model_rank",
        ]
    ]


def build_signal_table(
    prediction_frames: Sequence[pd.DataFrame],
    top_k: int,
    min_votes: int | None = None,
    require_all_models: bool = True,
) -> pd.DataFrame:
    if not prediction_frames:
        raise ValueError("No prediction frames provided.")

    combined = pd.concat(prediction_frames, ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    combined["in_top_k"] = combined["model_rank"] <= int(top_k)

    model_count = int(combined["model_id"].nunique())
    if model_count <= 1:
        vote_threshold = 1
    elif require_all_models:
        vote_threshold = model_count
    elif min_votes is not None:
        vote_threshold = max(1, min(int(min_votes), model_count))
    else:
        vote_threshold = 1

    top = combined[combined["in_top_k"]]
    votes = (
        top.groupby(["signal_date", "ticker"])["model_id"]
        .nunique()
        .rename("vote_count")
        .reset_index()
    )
    out = combined.merge(votes, on=["signal_date", "ticker"], how="left")
    out["vote_count"] = out["vote_count"].fillna(0).astype(int)
    out["model_count"] = model_count
    out["vote_threshold"] = vote_threshold
    out["selected"] = out["vote_count"] >= vote_threshold
    return out.sort_values(
        ["signal_date", "selected", "vote_count", "ticker", "model_id"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def selected_tickers_by_date(signal_table: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    if signal_table.empty:
        return {}
    selected = signal_table[signal_table["selected"]].copy()
    selected["signal_date"] = pd.to_datetime(selected["signal_date"])
    unique = selected.drop_duplicates(["signal_date", "ticker"])
    return {
        pd.Timestamp(date): sorted(group["ticker"].astype(str).tolist())
        for date, group in unique.groupby("signal_date")
    }


def build_backtest_trades(
    signal_table: pd.DataFrame,
    price_df: pd.DataFrame,
    holding_horizon: int,
) -> pd.DataFrame:
    if not isinstance(price_df.index, pd.MultiIndex):
        raise ValueError("price_df must have MultiIndex(date, ticker).")

    dates = pd.DatetimeIndex(price_df.index.get_level_values("date").unique()).sort_values()
    date_pos = {pd.Timestamp(date): pos for pos, date in enumerate(dates)}
    signals = selected_tickers_by_date(signal_table)
    rows: list[dict] = []

    for signal_date, tickers in sorted(signals.items()):
        pos = date_pos.get(pd.Timestamp(signal_date))
        if pos is None:
            continue
        entry_pos = pos + 1
        exit_pos = pos + int(holding_horizon)
        if entry_pos >= len(dates) or exit_pos >= len(dates):
            continue

        entry_date = pd.Timestamp(dates[entry_pos])
        exit_date = pd.Timestamp(dates[exit_pos])
        per_ticker: list[tuple[str, float]] = []
        for ticker in tickers:
            try:
                entry = float(price_df.loc[(entry_date, ticker), "open"])
                exit_ = float(price_df.loc[(exit_date, ticker), "close"])
            except KeyError:
                continue
            if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0:
                continue
            per_ticker.append((ticker, (exit_ - entry) / entry))

        if not per_ticker:
            continue

        rows.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "tickers": ";".join(ticker for ticker, _ in per_ticker),
                "n_tickers": len(per_ticker),
                "ret": float(np.mean([ret for _, ret in per_ticker])),
                "per_ticker_ret": ";".join(
                    f"{ticker}:{ret:.8f}" for ticker, ret in per_ticker
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "signal_date",
            "entry_date",
            "exit_date",
            "tickers",
            "n_tickers",
            "ret",
            "per_ticker_ret",
        ],
    )
