"""
Evo_Finance — Data Loader
──────────────────────────
Reads per-ticker CSV files produced by craw_data.py and assembles them into
a single MultiIndex (date, ticker) DataFrame ready for the pipeline.

Expected input (per file at <data_dir>/<TICKER>.csv):
    date, open, high, low, close, volume, is_trading_day

Key behaviours
  - Chỉ giữ ngày giao dịch thực sự (volume > 0).
  - Align tất cả ticker về phần giao nhau của ngày giao dịch.
  - Không forward-fill giá: mỗi hàng đều là dữ liệu giao dịch thật.
  - Trả về DataFrame với MultiIndex (date, ticker) sorted ascending.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config.settings import MARKET_INDEX_TICKER

logger = logging.getLogger(__name__)

# Deprecated: current loader uses strict date intersection and does not
# forward-fill/drop by missing-day ratio. Kept to avoid breaking older imports.
MISSING_DAY_THRESHOLD: float = 0.90

# Cột OHLCV bắt buộc (không tính is_trading_day)
_REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
_KNOWN_INDEX_TICKERS = {"VNINDEX", "VN30", "VN30INDEX", "HNXINDEX", "UPCOMINDEX"}


# ─── Public API ───────────────────────────────────────────────────────────────

def load_from_dir(
    data_dir:  str | Path,
    tickers:   Optional[List[str]] = None,
    min_rows:  int = 100,
    ffill_prices: bool = True,
    include_vnindex: bool = True,
    market_index_ticker: str | None = None,
) -> pd.DataFrame:
    """
    Load và merge tất cả file CSV trong data_dir thành MultiIndex DataFrame.

    Parameters
    ----------
    data_dir      : Thư mục chứa <TICKER>.csv (output của craw_data.py).
    tickers       : Danh sách ticker muốn load; None = load tất cả file .csv
                    (trừ VNINDEX).
    min_rows      : Ticker có ít hơn min_rows dòng hợp lệ thì bị bỏ qua.
    ffill_prices  : Backward-compatible no-op. Loader now uses strict date
                    intersection and does not forward-fill synthetic rows.

    Returns
    -------
    pd.DataFrame với MultiIndex (date, ticker), columns = [open, high, low,
    close, volume, is_trading_day], sorted ascending.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir không tồn tại: {data_dir}")

    # ── Tìm files ─────────────────────────────────────────────────────────────
    market_ticker = _normalize_market_index_ticker(market_index_ticker)
    index_tickers = _index_tickers_to_exclude(market_ticker)

    if tickers is not None:
        stock_tickers = [
            t for t in tickers
            if t.upper() not in index_tickers
        ]
        csv_files = [data_dir / f"{t}.csv" for t in stock_tickers]
        missing   = [f for f in csv_files if not f.exists()]
        if missing:
            logger.warning("Không tìm thấy file cho: %s", [f.stem for f in missing])
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))
        # Loại VNINDEX khỏi universe cổ phiếu (dùng riêng nếu cần)
        csv_files = [f for f in csv_files if f.stem.upper() not in index_tickers]

    if not csv_files:
        raise ValueError(f"Không tìm thấy file CSV nào trong {data_dir}")

    logger.info("Tìm thấy %d file CSV, bắt đầu load...", len(csv_files))

    # ── Load từng ticker ──────────────────────────────────────────────────────
    frames: dict[str, pd.DataFrame] = {}
    for fpath in csv_files:
        ticker = fpath.stem.upper()
        df = _load_single(fpath, ticker, min_rows)
        if df is not None:
            frames[ticker] = df

    if not frames:
        raise ValueError("Không có ticker nào load thành công.")

    logger.info("Load thành công %d / %d ticker.", len(frames), len(csv_files))

    market_index_df: Optional[pd.DataFrame] = None
    if include_vnindex and market_ticker is not None:
        market_index_df = load_market_index(data_dir, market_ticker)
        if market_index_df is not None:
            logger.info(
                "Load market index %s: %d rows, %s -> %s",
                market_ticker,
                len(market_index_df),
                market_index_df.index.min().date(),
                market_index_df.index.max().date(),
            )

    # ── INTERSECTION: chỉ giữ ngày TẤT CẢ ticker đều có giao dịch thực ──────
    # UNION sẽ tạo ffill noise ở ngày 1 ticker bị suspend/halted
    # INTERSECTION đảm bảo mỗi hàng là dữ liệu thực, không có giá giả
    date_sets = [set(df.index) for df in frames.values()]
    if market_index_df is not None:
        date_sets.append(set(market_index_df.index))
    all_dates = sorted(set.intersection(*date_sets))
    all_dates = pd.DatetimeIndex(all_dates)

    if len(all_dates) == 0:
        raise ValueError(
            "Không có ngày nào chung giữa tất cả các ticker. "
            "Kiểm tra lại dữ liệu hoặc giảm số ticker."
        )

    logger.info(
        "Intersection: %d ngày chung (từ %s đến %s)",
        len(all_dates), all_dates.min().date(), all_dates.max().date(),
    )

    # Log ticker nào làm mất nhiều ngày nhất (hữu ích để debug suspend)
    if len(frames) > 1:
        full_union = set.union(*date_sets)
        for ticker, df in sorted(
            frames.items(),
            key=lambda kv: len(full_union - set(kv[1].index)),
            reverse=True,
        )[:5]:
            missing = len(full_union - set(df.index))
            if missing > 0:
                logger.info(
                    "  %s thiếu %d ngày so với union (suspend/halt/IPO muộn)",
                    ticker, missing,
                )

    # ── Reindex về intersection — không cần ffill vì chỉ giữ ngày thực ───────
    merged: dict[str, pd.DataFrame] = {}

    for ticker, df in frames.items():
        df_intersect = df.reindex(all_dates)

        # Sau intersection vẫn có thể còn NaN nếu ticker có gap lạ
        nan_ratio = df_intersect["close"].isna().mean()
        if nan_ratio > 0:
            logger.warning(
                "Drop %s: còn %.1f%% NaN sau intersection.",
                ticker, nan_ratio * 100,
            )
            continue

        merged[ticker] = df_intersect

    if not merged:
        raise ValueError("Tất cả ticker đều bị drop sau khi intersection.")

    # ── Gộp thành MultiIndex ──────────────────────────────────────────────────
    combined = pd.concat(merged, names=["ticker"])          # (ticker, date)
    combined = combined.swaplevel().sort_index()             # (date, ticker)
    combined.index.names = ["date", "ticker"]

    if market_index_df is not None:
        market_df = market_index_df.reindex(all_dates).add_prefix("market_")
        if market_df["market_close"].isna().any():
            raise ValueError(f"{market_ticker} con NaN sau khi intersection theo ngay.")
        combined = combined.join(market_df, on="date")

    # Drop các dòng vẫn còn NaN ở close (đầu chuỗi trước khi ticker có data)
    before = len(combined)
    combined = combined.dropna(subset=["close"])
    after = len(combined)
    if before != after:
        logger.info("Dropped %d dòng NaN đầu chuỗi.", before - after)

    logger.info(
        "Dataset cuối: %d dòng | %d ticker | %d ngày",
        len(combined),
        combined.index.get_level_values("ticker").nunique(),
        combined.index.get_level_values("date").nunique(),
    )
    return combined


# ─── Internal ─────────────────────────────────────────────────────────────────

def _load_single(
    fpath: Path,
    ticker: str,
    min_rows: int,
) -> Optional[pd.DataFrame]:
    """Load và validate 1 file CSV."""
    try:
        df = pd.read_csv(fpath, parse_dates=["date"])
    except Exception as exc:
        logger.warning("Không đọc được %s: %s", fpath, exc)
        return None

    # Kiểm tra cột bắt buộc
    missing_cols = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        logger.warning("%s thiếu cột: %s — bỏ qua.", ticker, missing_cols)
        return None

    # Thêm is_trading_day nếu chưa có
    if "is_trading_day" not in df.columns:
        df["is_trading_day"] = (df["volume"] > 0).astype(int)

    # Chỉ giữ ngày giao dịch thực tế
    df = df[df["volume"] > 0].copy()

    if len(df) < min_rows:
        logger.warning("%s chỉ có %d dòng hợp lệ (< %d) — bỏ qua.", ticker, len(df), min_rows)
        return None

    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()

    # Chỉ giữ các cột cần thiết
    keep = _REQUIRED_COLS + ["is_trading_day"]
    df = df[[c for c in keep if c in df.columns]]

    # Ép kiểu numeric
    for col in _REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.debug("Loaded %s: %d rows, %s → %s", ticker, len(df),
                 df.index.min().date(), df.index.max().date())
    return df


# ─── Convenience: load VNINDEX riêng ─────────────────────────────────────────

def _normalize_market_index_ticker(ticker: str | None = None) -> str | None:
    ticker = MARKET_INDEX_TICKER if ticker is None else ticker
    if ticker is None:
        return None
    ticker = str(ticker).strip().upper()
    return ticker or None


def _index_tickers_to_exclude(market_ticker: str | None) -> set[str]:
    tickers = set(_KNOWN_INDEX_TICKERS)
    if market_ticker is not None:
        tickers.add(market_ticker)
    return tickers


def load_market_index(
    data_dir: str | Path,
    ticker: str | None = None,
) -> Optional[pd.DataFrame]:
    """Load the configured market benchmark as a flat date-indexed DataFrame."""
    market_ticker = _normalize_market_index_ticker(ticker)
    if market_ticker is None:
        return None

    fpath = Path(data_dir) / f"{market_ticker}.csv"
    if not fpath.exists():
        logger.warning("%s.csv khong ton tai tai %s", market_ticker, data_dir)
        return None
    return _load_single(fpath, market_ticker, min_rows=10)


def load_vnindex(data_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Load VNINDEX.csv riêng nếu cần dùng làm market benchmark.
    Trả về DataFrame flat với index = date.
    """
    fpath = Path(data_dir) / "VNINDEX.csv"
    if not fpath.exists():
        logger.warning("VNINDEX.csv không tồn tại tại %s", data_dir)
        return None
    df = _load_single(fpath, "VNINDEX", min_rows=10)
    return df
