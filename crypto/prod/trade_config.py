"""Runtime config for the crypto spot trading bot.

Defaults are intentionally conservative: dry-run + testnet.  To send real
orders you must edit ALLOW_REAL_TRADING and pass --execute --live explicitly.
"""

from __future__ import annotations

from pathlib import Path


PREDICTION_PATH: Path = Path("crypto/prod/live/latest_prediction.json")
TRADE_STATE_PATH: Path = Path("crypto/prod/live/trade_state.json")
LIVE_NOTIFY_STATE_PATH: Path = Path("crypto/prod/live/live_notify_state.json")

SYMBOL: str = "BTCUSDT"
INTERVAL: str = "15m"
QUOTE_ORDER_QTY: float = 7.0
TAKE_PROFIT_PCT: float = 0.0035
SELL_QTY_SAFETY_FACTOR: float = 0.999

POLL_SECONDS: float = 10.0
MAX_FINAL_SELL_WAIT_SECONDS: float = 5 * 60
RECV_WINDOW: int = 5000

BINANCE_BASE_URL: str = "https://api.binance.com"
BINANCE_TESTNET_BASE_URL: str = "https://testnet.binance.vision"
API_KEY_ENV: str = "BINANCE_API_KEY"
API_SECRET_ENV: str = "BINANCE_API_SECRET"
TELEGRAM_BOT_TOKEN_ENV: str = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV: str = "TELEGRAM_CHAT_ID"
TELEGRAM_NOTIFY_ENV: str = "TELEGRAM_NOTIFY"

# A second switch to prevent accidental real trading.
ALLOW_REAL_TRADING: bool = True
