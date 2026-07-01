"""Download BTC/USDT OHLCV candles for the crypto pipeline.

Default source is Binance Spot public klines:
    https://api.binance.com/api/v3/klines

Example:
    python craw_btc.py
    python craw_btc.py --start "2018-01-01 00:00:00+07:00" --interval 15m
    python craw_btc.py --symbol BTCUSDT --out data/crypto/BTCUSDT_15m.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "15m"
DEFAULT_START = "2018-01-01 00:00:00+07:00"
OUTPUT_TIMEZONE = "Asia/Ho_Chi_Minh"
DEFAULT_OUTPUT_DIR = Path("data/crypto")
DEFAULT_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_PATH = "/api/v3/klines"
BINANCE_LIMIT = 1000
REQUEST_SLEEP_SEC = 0.25
RATE_LIMIT_SLEEP_SEC = 60.0
RATE_LIMIT_STATUS_CODES = {418, 429}
MAX_RETRIES = 8

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _parse_time_ms(value: str | None, *, default_now: bool = False) -> int | None:
    if value is None:
        if default_now:
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        return None

    clean = str(value).strip()
    if not clean:
        return None
    if clean.lower() == "now":
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    ts = pd.Timestamp(clean)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _retry_after_seconds(exc: HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _request_json(url: str, retries: int = MAX_RETRIES) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "Evo_Finance/crypto-crawler"})
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_exc = exc
            if exc.code in RATE_LIMIT_STATUS_CODES:
                sleep_sec = max(
                    _retry_after_seconds(exc) or 0.0,
                    RATE_LIMIT_SLEEP_SEC,
                )
                print(
                    f"Rate limited by Binance HTTP {exc.code} "
                    f"({attempt}/{retries}). Sleep {sleep_sec:.1f}s"
                )
            else:
                sleep_sec = min(2.0 * attempt, 10.0)
                print(
                    f"HTTP error ({attempt}/{retries}): {exc}. "
                    f"Retry in {sleep_sec:.1f}s"
                )
            time.sleep(sleep_sec)
        except (URLError, TimeoutError) as exc:
            last_exc = exc
            sleep_sec = min(2.0 * attempt, 10.0)
            print(f"Request failed ({attempt}/{retries}): {exc}. Retry in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)

    raise RuntimeError(f"Request failed after {retries} retries: {last_exc}") from last_exc


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = BINANCE_LIMIT,
) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    interval = interval.strip()
    if interval not in INTERVAL_TO_MS:
        allowed = ", ".join(sorted(INTERVAL_TO_MS))
        raise ValueError(f"Unsupported interval {interval!r}. Allowed: {allowed}")
    if start_ms <= 0:
        raise ValueError("start_ms must be positive.")
    if end_ms is not None and end_ms <= start_ms:
        raise ValueError("end time must be after start time.")

    rows: list[list[Any]] = []
    cursor = int(start_ms)
    step_ms = INTERVAL_TO_MS[interval]
    base_url = base_url.rstrip("/")

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": int(limit),
            "startTime": cursor,
        }
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        url = f"{base_url}{BINANCE_KLINES_PATH}?{urlencode(params)}"
        batch = _request_json(url)
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + step_ms

        print(
            f"{symbol} {interval}: fetched={len(rows):,} "
            f"last={pd.to_datetime(last_open_time, unit='ms', utc=True)}"
        )

        if len(batch) < limit:
            break
        if end_ms is not None and next_cursor >= end_ms:
            break
        if next_cursor <= cursor:
            raise RuntimeError("Binance pagination did not advance.")

        cursor = next_cursor
        time.sleep(REQUEST_SLEEP_SEC)

    return _klines_to_dataframe(rows)


def _klines_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "is_trading_day",
            ]
        )

    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df["trade_count"] = df["trade_count"].astype(int)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df["date"] = df["date"].dt.tz_convert(OUTPUT_TIMEZONE).dt.tz_localize(None)
    df["is_trading_day"] = 1
    return df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "is_trading_day",
        ]
    ]


def save_ohlcv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if df.empty:
        print(f"Saved empty file: {out_path}")
        return
    print(
        f"Saved {len(df):,} rows to {out_path} | "
        f"{df['date'].min()} -> {df['date'].max()}"
    )


def _default_output_path(symbol: str, interval: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{symbol.upper()}_{interval}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Binance symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Candle interval, default: 15m.")
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date/time. Naive values are treated as UTC. Default: 2018-01-01 00:00:00+07:00.",
    )
    parser.add_argument("--end", default="now", help="End date/time. Naive values are treated as UTC. Default: now.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Binance base URL.")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    interval = args.interval.strip()
    start_ms = _parse_time_ms(args.start)
    end_ms = _parse_time_ms(args.end, default_now=True)
    if start_ms is None:
        raise ValueError("--start is required.")

    out_path = Path(args.out) if args.out else _default_output_path(symbol, interval)
    df = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        base_url=args.base_url,
    )
    save_ohlcv(df, out_path)


if __name__ == "__main__":
    main()
