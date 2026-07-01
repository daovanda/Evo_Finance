"""Backtest production prediction signals on val+test signal dates."""

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

from config.settings import HOLDING_HORIZON
from data.loader import load_from_dir
from model.data_utils import validate_market_ohlcv, validate_ohlcv
from prod.common import build_backtest_trades

logger = logging.getLogger("evo_finance.prod.backtest")


def run(config_module: str = "prod.config") -> pd.DataFrame:
    config = importlib.import_module(config_module)
    data_dir = Path(config.DATA_DIR)
    predictions_path = Path(config.PREDICTIONS_CSV)
    output_path = Path(config.BACKTEST_CSV)

    signals = pd.read_csv(predictions_path, parse_dates=["signal_date"])
    df = load_from_dir(data_dir, tickers=getattr(config, "TICKERS", None))
    validate_ohlcv(df)
    validate_market_ohlcv(df)

    trades = build_backtest_trades(
        signals,
        df,
        holding_horizon=HOLDING_HORIZON,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_path, index=False)
    logger.info("Saved backtest to %s (%d trades)", output_path, len(trades))
    return trades


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
