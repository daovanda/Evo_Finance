"""Local Kronos evaluation config for Evo_Finance BTC 15m data."""

from __future__ import annotations

from pathlib import Path


KRONOS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATA_PATH = PROJECT_ROOT / "data" / "crypto" / "BTCUSDT_15m.csv"
OUTPUT_DIR = KRONOS_ROOT / "my" / "output"

TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"

# Kronos-small/base are documented with max_context=512.
LOOKBACK = 512
MAX_CONTEXT = 512

# Same as current h5 on 15m data: 5 bars = 75 minutes.
PRED_LEN = 5 

# Evaluation range. This matches the current crypto final val/test convention.
EVAL_START = "2026-07-10"
TEST_START = "2026-07-11"
EVAL_END = None

# Sampling config. Higher SAMPLE_COUNT is smoother but slower.
TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1

# Used only for chart reference lines.
TAKE_PROFIT_PCT = 0.003
TRADE_COST = 0.002
