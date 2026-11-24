"""Download SPY and a diversified 50-stock S&P 500 universe from Alpaca.

The crawler stores one CSV per symbol under ``data/raw/sp500`` and can resume
from the last timestamp already present in each file.

Examples:
    python craw_sp500.py
    python craw_sp500.py --interval 5Min --start 2022-01-01
    python craw_sp500.py --symbols SPY,AAPL,MSFT --interval 15Min

Credentials are read from ``ALPACA_API_KEY`` and ``ALPACA_API_SECRET`` in the
environment or the repository ``.env`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd


MARKET_SYMBOL = "SPY"
SP500_UNIVERSE = [
    "NVDA", "AAPL", "MSFT", "AVGO", "AMD", "ORCL", "CSCO", "AMAT", "TXN", "IBM",
    "GOOGL", "META", "NFLX", "DIS",
    "AMZN", "TSLA", "HD", "MCD", "BKNG",
    "WMT", "COST", "PG", "KO",
    "JPM", "BRK.B", "V", "MA", "BAC", "GS",
    "LLY", "JNJ", "UNH", "ABBV", "MRK", "TMO",
    "GE", "CAT", "RTX", "UNP", "BA",
    "XOM", "CVX", "COP",
    "NEE", "SO",
    "PLD", "AMT",
    "LIN", "SHW", "FCX",
]

DEFAULT_INTERVAL = "15Min"
DEFAULT_START = "2018-01-01"
DEFAULT_OUTPUT_DIR = Path("data/raw/sp500")
DEFAULT_BASE_URL = "https://data.alpaca.markets"
DEFAULT_FEED = "sip"
DEFAULT_ADJUSTMENT = "all"
DEFAULT_LIMIT = 10_000
REQUEST_SLEEP_SECONDS = 0.35
MAX_RETRIES = 8
NEW_YORK = ZoneInfo("America/New_York")

OUTPUT_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _normalize_interval(value: str) -> str:
    clean = value.strip().lower()
    aliases = {
        "1m": "1Min",
        "1min": "1Min",
        "5m": "5Min",
        "5min": "5Min",
        "15m": "15Min",
        "15min": "15Min",
        "30m": "30Min",
        "30min": "30Min",
        "1h": "1Hour",
        "1hour": "1Hour",
        "1d": "1Day",
        "1day": "1Day",
    }
    if clean not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported interval {value!r}. Allowed aliases: {allowed}")
    return aliases[clean]


def _interval_delta(interval: str) -> timedelta:
    values = {
        "1Min": timedelta(minutes=1),
        "5Min": timedelta(minutes=5),
        "15Min": timedelta(minutes=15),
        "30Min": timedelta(minutes=30),
        "1Hour": timedelta(hours=1),
        "1Day": timedelta(days=1),
    }
    return values[interval]


def _utc_iso(value: str | pd.Timestamp | datetime) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _request_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Alpaca HTTP {exc.code}: {detail}")
            if exc.code not in {429, 500, 502, 503, 504}:
                raise last_error from exc
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after else min(2**attempt, 60)
        except (URLError, TimeoutError) as exc:
            last_error = exc
            delay = min(2**attempt, 30)

        delay += random.uniform(0.0, 0.5)
        print(f"Request failed ({attempt}/{MAX_RETRIES}); retry in {delay:.1f}s: {last_error}")
        time.sleep(delay)

    raise RuntimeError(f"Alpaca request failed after {MAX_RETRIES} retries: {last_error}")


def _bars_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows).rename(
        columns={
            "t": "date",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trade_count",
            "vw": "vwap",
        }
    )
    for column in OUTPUT_COLUMNS:
        if column not in df:
            df[column] = pd.NA
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    numeric = [column for column in OUTPUT_COLUMNS if column != "date"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df[OUTPUT_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"])


def _regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    local = df["date"].dt.tz_convert(NEW_YORK)
    minutes = local.dt.hour * 60 + local.dt.minute
    mask = (local.dt.weekday < 5) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return df.loc[mask].copy()


def fetch_symbol_bars(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    api_key: str,
    api_secret: str,
    *,
    feed: str = DEFAULT_FEED,
    adjustment: str = DEFAULT_ADJUSTMENT,
    base_url: str = DEFAULT_BASE_URL,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "User-Agent": "Evo_Finance/sp500-crawler",
    }
    rows: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params = {
            "timeframe": interval,
            "start": start,
            "end": end,
            "limit": DEFAULT_LIMIT,
            "adjustment": adjustment,
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{base_url.rstrip('/')}/v2/stocks/{symbol}/bars?{urlencode(params)}"
        payload = _request_json(url, headers)
        batch = payload.get("bars") or []
        rows.extend(batch)
        page_token = payload.get("next_page_token")

        last = batch[-1].get("t") if batch else "no-data"
        print(f"{symbol} {interval}: fetched={len(rows):,} last={last}")
        if not page_token:
            break
        time.sleep(REQUEST_SLEEP_SECONDS)

    df = _bars_to_dataframe(rows)
    if regular_session_only:
        df = _regular_session_only(df)
    return df


def _existing_resume_start(path: Path, configured_start: str, interval: str) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return _utc_iso(configured_start)
    try:
        dates = pd.read_csv(path, usecols=["date"])["date"]
        last = pd.to_datetime(dates, utc=True, errors="coerce").max()
    except (ValueError, OSError, pd.errors.ParserError):
        return _utc_iso(configured_start)
    if pd.isna(last):
        return _utc_iso(configured_start)
    return _utc_iso(last - _interval_delta(interval))


def _merge_and_save(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    frames = [fresh]
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"], utc=True, errors="coerce")
        frames.insert(0, old)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["date"])
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def _parse_symbols(value: str | None) -> list[str]:
    if value:
        symbols = [item.strip().upper() for item in value.split(",")]
        return list(dict.fromkeys(item for item in symbols if item))
    return [MARKET_SYMBOL, *SP500_UNIVERSE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="1Min, 5Min, 15Min, 30Min, 1Hour or 1Day.")
    parser.add_argument("--start", default=DEFAULT_START, help="UTC start date/time. Default: 2018-01-01.")
    parser.add_argument("--end", default=None, help="UTC end date/time. Default: last safely closed bar.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols; default: SPY plus 50 stocks.")
    parser.add_argument("--feed", choices=["sip", "iex"], default=DEFAULT_FEED)
    parser.add_argument("--adjustment", choices=["raw", "split", "dividend", "all"], default=DEFAULT_ADJUSTMENT)
    parser.add_argument("--all-sessions", action="store_true", help="Keep pre-market and after-hours bars.")
    parser.add_argument("--no-resume", action="store_true", help="Download from --start even when a CSV exists.")
    args = parser.parse_args()

    _load_dotenv()
    api_key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
    api_secret = (
        os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or ""
    ).strip()
    if not api_key or not api_secret:
        raise SystemExit("Missing ALPACA_API_KEY or ALPACA_API_SECRET in .env/environment.")

    interval = _normalize_interval(args.interval)
    safe_end = datetime.now(timezone.utc) - timedelta(minutes=16 if args.feed == "sip" else 1)
    end = _utc_iso(args.end) if args.end else _utc_iso(safe_end)
    symbols = _parse_symbols(args.symbols)
    print(
        f"Downloading {len(symbols)} symbols | interval={interval} | feed={args.feed} | "
        f"regular_session={not args.all_sessions} | output={args.out_dir}"
    )

    failures: list[tuple[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        path = args.out_dir / f"{symbol}_{interval}.csv"
        start = _utc_iso(args.start)
        if not args.no_resume:
            start = _existing_resume_start(path, args.start, interval)
        print(f"[{index}/{len(symbols)}] {symbol}: {start} -> {end}")
        try:
            fresh = fetch_symbol_bars(
                symbol,
                interval,
                start,
                end,
                api_key,
                api_secret,
                feed=args.feed,
                adjustment=args.adjustment,
                regular_session_only=not args.all_sessions,
            )
            merged = _merge_and_save(path, fresh)
            first = merged["date"].iloc[0] if not merged.empty else "empty"
            last = merged["date"].iloc[-1] if not merged.empty else "empty"
            print(f"Saved {path}: rows={len(merged):,} range={first} -> {last}")
        except Exception as exc:  # Keep the other symbols progressing.
            failures.append((symbol, str(exc)))
            print(f"ERROR {symbol}: {exc}")
        time.sleep(REQUEST_SLEEP_SECONDS)

    if failures:
        details = "\n".join(f"  {symbol}: {error}" for symbol, error in failures)
        raise SystemExit(f"Completed with {len(failures)} failed symbols:\n{details}")
    print(f"Completed successfully: {len(symbols)} symbols saved in {args.out_dir}")


if __name__ == "__main__":
    main()
