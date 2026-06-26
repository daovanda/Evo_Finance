import pandas as pd


import os
import sys
os.environ["VNAI_DISABLE_PROMO"] = "1"
sys.stdout.reconfigure(encoding="utf-8")
from vnstock import Vnstock
from datetime import datetime
import os
import numpy as np
import time
from config.settings import MARKET_INDEX_TICKER

# =========================
# CONFIG
# =========================
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = [
    "ACB", "BID", "BVH", "CTG", "FPT",
    "GAS", "GVR", "HDB", "HPG", "KDH",
    "MBB", "MSN", "MWG", "NVL", "PDR",
    "PLX", "PNJ", "POW", "SAB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM",
    "VIC", "VJC", "VNM", "VPB", "VRE"
]

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

vn = Vnstock(source="KBS")

# =========================
# STOCK DATA - FIXED
# =========================
for ticker in TICKERS:
    print(f"📥 Loading {ticker}")

    try:
        stock = vn.stock(symbol=ticker)
        df = stock.quote.history(
            start=START_DATE,
            end=END_DATE,
            interval="1D"
        )
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]]
        
        # FIXED: Chỉ giữ lại ngày giao dịch thực tế, không ffill
        df = df[df["volume"] > 0].copy()
        
        # FIX duplicate last date: giữ dòng cuối cùng (data mới nhất)
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        
        # Đánh dấu là ngày giao dịch
        df["is_trading_day"] = 1
        
    except Exception as e:
        print(f"⚠️ {ticker} error:", e)
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "is_trading_day"])

    df.to_csv(f"{OUTPUT_DIR}/{ticker}.csv", index=False)

    time.sleep(5)  # Giữ khoảng cách giữa các request để tránh bị rate limit

# =========================
# MARKET INDEX - FIXED
# =========================
market_ticker = str(MARKET_INDEX_TICKER).strip().upper() if MARKET_INDEX_TICKER else ""

if market_ticker:
    print(f"📥 Loading {market_ticker}")

    stock = vn.stock(symbol=market_ticker)
    df = stock.quote.history(
        start=START_DATE,
        end=END_DATE,
        interval="1D"
    )
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # FIXED: Chỉ giữ ngày giao dịch
    df = df[df["volume"] > 0].copy()

    # FIX duplicate last date
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    df["is_trading_day"] = 1

    df.to_csv(f"{OUTPUT_DIR}/{market_ticker}.csv", index=False)

print("✅ DONE - Only trading days saved")
