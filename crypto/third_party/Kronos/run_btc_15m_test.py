from pathlib import Path
import sys

import pandas as pd
import torch

from model import Kronos, KronosTokenizer, KronosPredictor


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "crypto" / "BTCUSDT_15m.csv"
OUT_PATH = ROOT / "crypto" / "third_party" / "Kronos" / "btc_15m_prediction.csv"

LOOKBACK = 512
PRED_LEN = 5


def main():
    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["date"])
    df = df.sort_values("timestamps").drop_duplicates("timestamps")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["amount"] = df["close"] * df["volume"]
    df = df.dropna(subset=["timestamps", "open", "high", "low", "close", "volume", "amount"])

    if len(df) < LOOKBACK + PRED_LEN:
        raise ValueError(f"Not enough rows: {len(df)}")

    context = df.iloc[-(LOOKBACK + PRED_LEN):-PRED_LEN].copy()
    future_ts = df.iloc[-PRED_LEN:]["timestamps"].reset_index(drop=True)

    x_df = context[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = context["timestamps"].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Input last candle: {x_timestamp.iloc[-1]}")
    print(f"Predict timestamps: {future_ts.iloc[0]} -> {future_ts.iloc[-1]}")

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=future_ts,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    )

    pred_df.to_csv(OUT_PATH, index=True)
    print(pred_df)
    print(f"Saved: {OUT_PATH}")

    entry_open = float(df.iloc[-PRED_LEN]["open"])
    pred_mfe = float(pred_df["high"].max() / entry_open - 1.0)
    pred_close_ret = float(pred_df["close"].iloc[-1] / entry_open - 1.0)
    print(f"entry_open={entry_open:.2f}")
    print(f"pred_mfe={pred_mfe:.4%}")
    print(f"pred_close_ret_h{PRED_LEN}={pred_close_ret:.4%}")


if __name__ == "__main__":
    main()