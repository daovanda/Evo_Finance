# Crypto Meta Learner

Meta learner nay la mot lop loc tin hieu don gian:

1. Load mot individual tu archive theo `rank`.
2. Train lai base model theo tung walk-forward fold va tung horizon.
3. Predict tren fold validation de tao OOF prediction.
4. Chi lay cac dong base model co signal dong thuan giua cac horizon.
5. Train mot LightGBM nho de du doan trade rule co net profit duong hay khong.
6. Train lai base model tren final train, predict final val/test, cho qua meta model
   da train va ve chart tom tat de so sanh truoc/sau meta filter.

Mac dinh meta target la:

```text
meta_label = payoff_future_return > TRADE_COST
```

voi `payoff_future_return` lay tu `crypto/config.py`:

```text
neu MFE >= PAYOFF_TP: future_return = PAYOFF_TP
nguoc lai: future_return = close_return_h
```

## Chay thu

```bash
python -m crypto.meta_learner.train \
  --archive crypto/results/crypto_btc_mfe_seed1_12h.json \
  --rank 1 \
  --base-label-mode mfe \
  --meta-label-mode payoff
```

Neu archive cu duoc train voi `close_exit`:

```bash
python -m crypto.meta_learner.train \
  --archive crypto/results/crypto_btc_close_exit_seed1_12h.json \
  --rank 1 \
  --base-label-mode close_exit \
  --base-label-threshold 0.001 \
  --meta-label-mode payoff
```

## Output

Mac dinh luu trong:

```text
crypto/meta_learner/output/<run_name>/
```

Bao gom:

```text
meta_dataset.csv
meta_model.txt
meta_holdout_predictions.csv
final_val_predictions.csv
final_test_predictions.csv
meta_summary.png
manifest.json
```

Y nghia nhanh:

```text
meta_dataset.csv              OOF dataset tu cac walk-forward validation fold
meta_holdout_predictions.csv  phan cuoi OOF dung de holdout cho meta learner
final_val_predictions.csv     final train -> predict val -> meta predict
final_test_predictions.csv    final train -> predict test -> meta predict
meta_summary.png              bang + bieu do truc quan truoc/sau meta filter
```

Trong phan final eval, meta threshold duoc fit tren `final_val_predictions.csv`
theo `TRADE_TOP_FRACTION`, sau do ap dung nguyen threshold do sang
`final_test_predictions.csv`. Nhu vay test khong tu chon threshold rieng.

`meta_dataset.csv` co cac feature toi thieu:

```text
pred_h*
threshold_h*
margin_h*
pred_mean
pred_std
margin_min
margin_mean
signal_count
vol_regime_z_50
range_pct_mean_14
realized_vol_z_120
buy_pressure_mean_10
signed_volume_sum_ratio_50
taker_buy_base_volume_ratio_14
quote_volume_proxy_log_delta_40
ret_close_20
```

Neu muon train tren tat ca OOF predictions thay vi chi signal rows:

```bash
python -m crypto.meta_learner.train \
  --archive crypto/results/crypto_btc_mfe_seed1_12h.json \
  --rank 1 \
  --base-label-mode mfe \
  --meta-label-mode payoff \
  --all-predictions
```

Co the override split final neu can:

```bash
python -m crypto.meta_learner.train \
  --archive crypto/results/crypto_btc_mfe_seed1_12h.json \
  --rank 1 \
  --base-label-mode mfe \
  --meta-label-mode payoff \
  --val-start 2024-01-01 \
  --test-start 2025-01-01
```
