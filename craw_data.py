"""Append missing daily KBS bars to the existing Vietnamese stock CSVs."""

import argparse
import os
from datetime import datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
COLUMNS = ["date", *PRICE_COLUMNS, "volume", "is_trading_day"]
VIETNAM_TZ = timezone(timedelta(hours=7))


def prepare_append(path: Path, fetch, end) -> tuple[pd.DataFrame, str]:
    old = pd.read_csv(path, dtype=str)
    if old.columns.tolist() != COLUMNS:
        raise ValueError("Unexpected CSV column layout")
    old_times = pd.DatetimeIndex(pd.to_datetime(old["date"]))
    old_days = old_times.normalize()
    if old.empty or old_days.hasnans or not old_days.is_unique:
        raise ValueError("Existing CSV is empty or has invalid/duplicate trading dates")
    last = old_days.max()
    if last >= end:
        return old.iloc[:0], str(last.date())
    start = last + pd.Timedelta(days=1)
    fresh = fetch(path.stem, str(start.date()), str(end.date()))
    if fresh.empty:
        return old.iloc[:0], str(last.date())
    fresh = fresh.rename(columns={"time": "date"}).copy()
    fresh["date"] = pd.to_datetime(fresh["date"])
    fresh = fresh.dropna(subset=["date", *PRICE_COLUMNS, "volume"])
    fresh = fresh[fresh["volume"].gt(0) & fresh[PRICE_COLUMNS].gt(0).all(axis=1)]
    fresh_days = fresh["date"].dt.normalize()
    new = fresh.loc[fresh_days.between(start, end)].assign(day=fresh_days)
    new = new.sort_values("date", kind="stable").drop_duplicates("day", keep="last").set_index("day")
    # The stock loader joins on the full timestamp, not only the calendar date.
    time_of_day = pd.Series(old_times - old_days).mode().iloc[0]
    new["date"] = (new.index + time_of_day).strftime("%Y-%m-%d %H:%M:%S")
    new["is_trading_day"] = 1
    return new[COLUMNS], str(last.date())


def append_atomic(path: Path, rows: pd.DataFrame) -> None:
    original = path.read_bytes()
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            handle.write(original)
            if not original.endswith(b"\n"):
                handle.write(b"\n")
            handle.write(rows.to_csv(index=False, header=False).encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true", help="Fetch and validate without writing")
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("*.csv"))
    if not paths:
        parser.error(f"No stock CSVs in {args.data_dir}")

    os.environ["VNAI_DISABLE_PROMO"] = "1"
    from vnstock import Vnstock

    vn = Vnstock(source="KBS")
    now = datetime.now(VIETNAM_TZ)
    end = pd.Timestamp(now.date() - timedelta(days=now.time() < clock_time(16)))

    def fetch(symbol, start_date, end_date):
        return vn.stock(symbol=symbol).quote.history(start=start_date, end=end_date, interval="1D")

    pending = []
    failures = []
    for number, path in enumerate(paths, start=1):
        print(f"[{number}/{len(paths)}] {path.stem}: checking through {end.date()}", flush=True)
        try:
            rows, last = prepare_append(path, fetch, end)
            latest = rows["date"].iloc[-1][:10] if not rows.empty else last
            print(f"  {last} -> {latest}: +{len(rows)} sessions", flush=True)
            if not rows.empty:
                pending.append((path, rows))
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            failures.append(path.stem)
        if number < len(paths):
            sleep(5)

    if failures:
        print(f"No CSVs changed. Failed: {', '.join(failures)}", flush=True)
        return 1
    if not args.dry_run:
        for path, rows in pending:
            append_atomic(path, rows)
    print(f"{'Validated' if args.dry_run else 'Updated'} {len(pending)} CSVs.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
