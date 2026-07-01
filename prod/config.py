"""Production configuration.

Edit SELECTED_INDIVIDUALS to choose archive/rank pairs from results/.
If one individual is configured, its daily top-K names are used directly.
If multiple individuals are configured, names are selected only when they are
inside the daily top-K of every model.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "prod" / "output"

# Each archive can be just a filename under results/ or a path.
SELECTED_INDIVIDUALS = [
    {"archive": "archive_final1_seed3.json", "rank": 39},
    {"archive": "archive_final1_seed3.json", "rank": 45},
    {"archive": "archive_final1_seed3.json", "rank": 46},
    {"archive": "archive_final1_seed3.json", "rank": 48},
    {"archive": "archive_final2_seed1.json", "rank": 49},
    {"archive": "archive_final3_seed1.json", "rank": 31},
]

TOP_K_PER_MODEL = 10

# True = intersection of all models' daily top-K lists.
# With TOP_K_PER_MODEL=10, every signal day can have at most 10 selected names.
ENSEMBLE_REQUIRE_ALL_MODELS = True

# Optional only if ENSEMBLE_REQUIRE_ALL_MODELS=False.
MIN_ENSEMBLE_VOTES = None

PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
BACKTEST_CSV = OUTPUT_DIR / "backtest.csv"

# Optional ticker subset passed to data.loader.load_from_dir.
TICKERS = None
