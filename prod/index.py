"""Run the full production pipeline: predict, then backtest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prod.backtest import run as run_backtest
from prod.predict import run as run_predict

logger = logging.getLogger("evo_finance.prod.index")


def run(config_module: str = "prod.config"):
    """Execute prediction and backtest with the same production config."""
    logger.info("Starting production prediction step with config=%s", config_module)
    predictions = run_predict(config_module)

    logger.info("Starting production backtest step with config=%s", config_module)
    trades = run_backtest(config_module)

    logger.info(
        "Production pipeline complete: predictions=%d rows | trades=%d rows",
        len(predictions),
        len(trades),
    )
    return predictions, trades


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
