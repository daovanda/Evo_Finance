"""Download Binance Spot 1s history and aggregate closed 15s or 30s candles."""
import argparse
from pathlib import Path
from tempfile import TemporaryFile
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

import craw_btc
from crypto.config import VAL_START


ARCHIVE = "https://data.binance.vision/data/spot"
AGG = dict(open="first", high="max", low="min", close="last", volume="sum",
           quote_volume="sum", trade_count="sum", taker_buy_base_volume="sum",
           taker_buy_quote_volume="sum")


def archive_frame(symbol, period, key):
    url = f"{ARCHIVE}/{period}/klines/{symbol}/1s/{symbol}-1s-{key}.zip"
    try:
        response = urlopen(url, timeout=120)
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    with response, TemporaryFile() as buffer:
        while chunk := response.read(1024 * 1024):
            buffer.write(chunk)
        buffer.seek(0)
        with ZipFile(buffer) as archive, archive.open(archive.namelist()[0]) as csv:
            frame = pd.read_csv(csv, header=None, usecols=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
    frame.columns = ["date", *AGG]
    stamp = frame.pop("date").astype("int64")
    stamp = stamp.where(stamp >= 10**15, stamp * 1000)
    frame.index = pd.to_datetime(stamp, unit="us", utc=True)
    return frame


def utc_time(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(craw_btc.OUTPUT_TIMEZONE)
    return stamp.tz_convert("UTC")


def aggregate(frame, start, end, seconds):
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    bins = frame.resample(f"{seconds}s")
    candles = bins.agg(AGG)
    counts = bins.size()
    candles = candles.loc[(counts == seconds) & (candles.index >= start)
                          & (candles.index + pd.Timedelta(seconds=seconds) <= end)]
    candles.index = candles.index.tz_convert(craw_btc.OUTPUT_TIMEZONE).tz_localize(None)
    candles.index.name = "date"
    return candles.assign(is_trading_day=1).reset_index()[craw_btc.OUTPUT_COLUMNS]


def api_frame(symbol, start, end):
    craw_btc.INTERVAL_TO_MS["1s"] = 1000
    frame = craw_btc.fetch_klines(symbol, "1s", start.value // 10**6,
                                  end.value // 10**6 - 1, "https://data-api.binance.vision")
    date = pd.to_datetime(frame.pop("date")).dt.tz_localize(craw_btc.OUTPUT_TIMEZONE)
    frame.index = date.dt.tz_convert("UTC")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--seconds", type=int, choices=(15, 30), required=True)
    parser.add_argument("--start", default=VAL_START, help="Naive timestamps use Vietnam time.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force-ipv4", action="store_true")
    args = parser.parse_args()
    if args.force_ipv4:
        craw_btc.force_ipv4_networking()
    symbol = args.symbol.upper()
    start = utc_time(args.start).ceil(f"{args.seconds}s")
    end = pd.Timestamp.now(tz="UTC").floor(f"{args.seconds}s")
    if start >= end:
        parser.error("Start must precede the latest closed bar.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    count = 0
    with partial.open("w", newline="", encoding="utf-8") as target:
        def write(frame, left, right):
            nonlocal count
            candles = aggregate(frame, left, right, args.seconds)
            if not candles.empty:
                candles.to_csv(target, index=False, header=count == 0)
            count += len(candles)
            print(f"{left} -> {right}: {len(candles):,} bars", flush=True)

        month = start.normalize().replace(day=1)
        while month < end:
            next_month = month + pd.offsets.MonthBegin()
            monthly = archive_frame(symbol, "monthly", month.strftime("%Y-%m")) if next_month <= end else None
            if monthly is not None:
                write(monthly, max(start, month), next_month)
            else:
                for day in pd.date_range(month, min(end, next_month), inclusive="left"):
                    left, right = max(start, day), min(end, day + pd.Timedelta(days=1))
                    if left >= right:
                        continue
                    daily = archive_frame(symbol, "daily", day.strftime("%Y-%m-%d"))
                    write(daily if daily is not None else api_frame(symbol, left, right), left, right)
            month = next_month
    if not count:
        raise RuntimeError("No complete candles downloaded.")
    partial.replace(args.out)
    print(f"Saved {count:,} candles to {args.out}")


if __name__ == "__main__":
    main()
