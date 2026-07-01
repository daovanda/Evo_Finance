"""Train selected production individuals and export daily prediction signals."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    MIN_SECTOR_MEMBERS_IN_UNIVERSE,
    REQUIRE_SECTOR_MAPPING,
    TEST_END,
    TEST_START,
    VAL_START,
    WF_PURGE_DAYS,
)
from data.loader import load_from_dir
from fitness.fitness import _ic_per_date
from model.data_utils import (
    label_dataframe,
    small_sectors_in_universe,
    split_labeled_by_dates,
    tickers_missing_sector,
    validate_full_universe_panel,
    validate_market_ohlcv,
    validate_ohlcv,
    validate_temporal_splits,
)
from model.trainer import Trainer
from prod.common import (
    build_signal_table,
    load_selected_individuals,
    prediction_frame_for_model,
)

logger = logging.getLogger("evo_finance.prod.predict")


def _mean_ic(pred: pd.Series, label: pd.Series, index: pd.Index) -> tuple[float, int]:
    """Return mean daily Spearman IC and number of IC dates."""
    ic_series = _ic_per_date(pred, label, index)
    if len(ic_series) == 0:
        return float("nan"), 0
    return float(ic_series.mean()), int(len(ic_series))


def _validate_sector_inputs(df: pd.DataFrame) -> None:
    missing = tickers_missing_sector(df)
    if missing and REQUIRE_SECTOR_MAPPING:
        raise ValueError(f"SECTORS mapping is missing tickers: {missing}")
    small = small_sectors_in_universe(df, MIN_SECTOR_MEMBERS_IN_UNIVERSE)
    if small and REQUIRE_SECTOR_MAPPING:
        raise ValueError(f"Some sectors have too few loaded tickers: {small}")


def run(config_module: str = "prod.config") -> pd.DataFrame:
    config = importlib.import_module(config_module)
    data_dir = Path(config.DATA_DIR)
    output_path = Path(config.PREDICTIONS_CSV)

    df = load_from_dir(data_dir, tickers=getattr(config, "TICKERS", None))
    validate_ohlcv(df)
    validate_market_ohlcv(df)
    validate_temporal_splits()
    _validate_sector_inputs(df)

    labeled_df = label_dataframe(df)
    train_df, val_df, test_df = split_labeled_by_dates(
        labeled_df,
        val_start=VAL_START,
        test_start=TEST_START,
        test_end=TEST_END,
        purge_days=WF_PURGE_DAYS,
    )
    expected_tickers = df.index.get_level_values("ticker").unique()
    validate_full_universe_panel(train_df, expected_tickers, "prod train split")
    validate_full_universe_panel(val_df, expected_tickers, "prod val split")
    validate_full_universe_panel(test_df, expected_tickers, "prod test split")

    feature_df = labeled_df.sort_index()
    selected = load_selected_individuals(
        getattr(config, "SELECTED_INDIVIDUALS"),
        getattr(config, "RESULTS_DIR"),
    )

    frames = []
    for item in selected:
        logger.info("Training %s from %s rank %d", item.model_id, item.archive_name, item.rank)
        trainer = Trainer()
        booster, _, val_pred, _, val_labels = trainer.train(
            item.individual,
            train_df,
            val_df,
            feature_df=feature_df,
            mode="final",
        )
        test_pred = trainer.predict(
            booster,
            item.individual,
            test_df,
            feature_df=feature_df,
        )
        test_labels = test_df["label"]
        val_ic, val_ic_days = _mean_ic(val_pred, val_labels, val_df.index)
        test_ic, test_ic_days = _mean_ic(test_pred, test_labels, test_df.index)
        logger.info(
            "Production IC %s: val_IC=%.4f (%d dates) | test_IC=%.4f (%d dates)",
            item.model_id,
            val_ic,
            val_ic_days,
            test_ic,
            test_ic_days,
        )
        frames.append(prediction_frame_for_model(item, val_pred, split="val"))
        frames.append(prediction_frame_for_model(item, test_pred, split="test"))

    signal_table = build_signal_table(
        frames,
        top_k=int(getattr(config, "TOP_K_PER_MODEL", 10)),
        min_votes=getattr(config, "MIN_ENSEMBLE_VOTES", None),
        require_all_models=bool(getattr(config, "ENSEMBLE_REQUIRE_ALL_MODELS", True)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_table.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s (%d rows)", output_path, len(signal_table))
    return signal_table


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prod.config", help="Python config module.")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
