# Guild Live Crypto

File nay huong dan phan tich archive, train model production, chay live prediction, chay giao dien, va chay bot trade Binance Spot.

Thu muc lien quan:

- Evolution archive: `crypto/results/*.json`
- Chart analyze: `crypto/results/chart/`
- Model production: `crypto/prod/model/<run_name>/`
- Manifest model: `crypto/prod/model/<run_name>/manifest.json`
- Live prediction: `crypto/prod/live/latest_prediction.json`
- Trade state: `crypto/prod/live/trade_state.json`
- Runtime config trade: `crypto/prod/trade_config.py`
- Secret local: `.env`

## 1. Config production

### Crypto model config

Dung file:

```text
crypto/config.py
```

Quan trong nhat:

```python
HOLDING_HORIZONS = [3, 5]
LABEL_THRESHOLD = 0.001
VAL_START = "2024-01-01"
TEST_START = "2025-01-01"
TRADE_TOP_FRACTION = 0.20
LGBM_PARAMS = {...}
```

`HOLDING_HORIZONS` quyet dinh so model duoc train moi individual. Vi du `[3, 5]` se co file:

```text
rank_01_h3.txt
rank_01_h5.txt
```

### Trading config

Dung file:

```text
crypto/prod/trade_config.py
```

Vi du hien tai:

```python
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
QUOTE_ORDER_QTY = 5.5
TAKE_PROFIT_PCT = 0.0035
SELL_QTY_SAFETY_FACTOR = 0.999
POLL_SECONDS = 10.0
MAX_FINAL_SELL_WAIT_SECONDS = 5 * 60
ALLOW_REAL_TRADING = True
```

Y nghia:

- `QUOTE_ORDER_QTY`: so USDT dung cho moi lenh BUY.
- `TAKE_PROFIT_PCT`: muc limit sell TP. `0.0035` = 0.35%.
- `SELL_QTY_SAFETY_FACTOR`: ban it hon so BTC mua mot chut de tranh loi fee/rounding.
- `ALLOW_REAL_TRADING`: cong tac chan lenh live. Muon trade that can `True` va CLI phai co `--execute --live`.

## 2. Tao file `.env`

Tao file `.env` o root project:

```powershell
cd D:\Evo_Finance
Copy-Item .env.example .env
```

Noi dung:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_NOTIFY=1
```

Khong commit `.env` len Git.

## 3. Phan tich archive

Analyze se train lai model tren final split va ve chart cho tung individual.

Chay rank 1:

```powershell
python -m crypto.analyze `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --rank 1
```

Chay top 5:

```powershell
python -m crypto.analyze `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --top 5
```

Chay nhieu rank cu the:

```powershell
python -m crypto.analyze `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --rank 1 3 7
```

Output chart:

```text
crypto/results/chart/
```

Trong chart:

- AUC: kha nang phan biet label 0/1.
- Precision: ty le label 1 trong nhom duoc model chon.
- PE: precision excess so voi base rate.
- MFE: max favorable excursion, tuc muc high lon nhat trong horizon so voi entry.
- Base MFE: ty le MFE baseline cua toan bo ngay.
- Ensemble: chi trade khi cac horizon trong cung individual deu dong y.

## 4. Train model production

Train model tu archive va luu vao `crypto/prod/model`.

Train rank 1:

```powershell
python -m crypto.prod.train_model `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --rank 1 `
  --run-name crypto_btc_seed1_12h
```

Train top 3:

```powershell
python -m crypto.prod.train_model `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --top 3 `
  --run-name crypto_btc_seed1_12h_top3
```

Train rank 1 va 5:

```powershell
python -m crypto.prod.train_model `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --rank 1 5 `
  --run-name crypto_btc_seed1_12h_r1_r5
```

Output:

```text
crypto/prod/model/crypto_btc_seed1_12h/
  manifest.json
  rank_01_h3.txt
  rank_01_h5.txt
```

`manifest.json` chua:

- feature cua individual
- danh sach horizon
- duong dan model
- threshold lay tu top prediction tren val
- config snapshot

## 5. Chay live prediction local

Backend se:

1. Doc local CSV.
2. Crawl them data Binance tu `last_local_candle - crawl_lookback_days`.
3. Merge va ghi de `data/crypto/BTCUSDT_15m.csv`.
4. Tinh feature tren tail gan nhat.
5. Load model tu `manifest.json`.
6. Xuat `crypto/prod/live/latest_prediction.json`.
7. Gui Telegram prediction neu `.env` da cau hinh.

Chay mot lan:

```powershell
python -m crypto.prod.live_backend `
  --model-dir crypto/prod/model/crypto_btc_seed1_12h `
  --data data/crypto/BTCUSDT_15m.csv `
  --out crypto/prod/live/latest_prediction.json
```

Chay loop moi nen:

```powershell
python -m crypto.prod.live_backend `
  --model-dir crypto/prod/model/crypto_btc_seed1_12h `
  --data data/crypto/BTCUSDT_15m.csv `
  --out crypto/prod/live/latest_prediction.json `
  --loop `
  --sleep-after-open 5 `
  --feature-lookback-bars 5000
```

`--sleep-after-open 5` nghia la cho them 5 giay sau khi nen moi mo roi moi crawl/predict.

`--feature-lookback-bars 5000` nghia la chi lay 5000 nen gan nhat de tinh feature live. Code da kiem tra voi model hien tai: prediction tu tail 5000 khop voi reference tail dai.

## 6. Chay giao dien local

Terminal rieng:

```powershell
python -m crypto.prod.live.app `
  --host 127.0.0.1 `
  --port 8765 `
  --prediction crypto/prod/live/latest_prediction.json `
  --trade-state crypto/prod/live/trade_state.json `
  --model-dir crypto/prod/model/crypto_btc_seed1_12h
```

Mo trinh duyet:

```text
http://127.0.0.1:8765
```

Giao dien tu refresh moi 10 giay.

## 7. Test Binance account

Truoc khi trade that, kiem tra API:

```powershell
python -m crypto.prod.trader --account-test --live
```

Ket qua can thay:

- `canTrade: true`
- `symbol_status: TRADING`
- balance USDT du cho `QUOTE_ORDER_QTY`

Gui test Telegram:

```powershell
python -m crypto.prod.trader --telegram-test
```

## 8. Chay bot trade dry-run

Dry-run khong gui order that.

```powershell
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --once
```

Dry-run loop:

```powershell
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --loop
```

Test guard BUY khong filled:

```powershell
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --simulate-buy-status NEW `
  --once
```

## 9. Chay live trading that

Can thoa man:

1. `.env` co `BINANCE_API_KEY` va `BINANCE_API_SECRET`.
2. Binance API da bat Spot Trading.
3. `crypto/prod/trade_config.py` co `ALLOW_REAL_TRADING = True`.
4. Tai khoan spot co du USDT.
5. Khong co open order BTCUSDT thu cong dang treo.
6. Da test `--account-test --live`.

Xoa state cu neu chac chan khong co vi the bot dang mo:

```powershell
Remove-Item crypto/prod/live/trade_state.json -ErrorAction SilentlyContinue
```

Chay trader live:

```powershell
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --execute `
  --live `
  --loop
```

Neu muon override so tien lenh cho lan chay nay:

```powershell
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --execute `
  --live `
  --loop `
  --quote-order-qty 6
```

CLI override khong sua file `trade_config.py`; no chi ap dung cho process dang chay.

## 10. Chay 3 terminal local

Terminal 1: live backend

```powershell
cd D:\Evo_Finance
python -m crypto.prod.live_backend `
  --model-dir crypto/prod/model/crypto_btc_seed1_12h `
  --data data/crypto/BTCUSDT_15m.csv `
  --out crypto/prod/live/latest_prediction.json `
  --loop `
  --sleep-after-open 5
```

Terminal 2: UI

```powershell
cd D:\Evo_Finance
python -m crypto.prod.live.app `
  --host 127.0.0.1 `
  --port 8765 `
  --prediction crypto/prod/live/latest_prediction.json `
  --trade-state crypto/prod/live/trade_state.json `
  --model-dir crypto/prod/model/crypto_btc_seed1_12h
```

Terminal 3: trader

Dry-run:

```powershell
cd D:\Evo_Finance
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --loop
```

Live:

```powershell
cd D:\Evo_Finance
python -m crypto.prod.trader `
  --prediction crypto/prod/live/latest_prediction.json `
  --state crypto/prod/live/trade_state.json `
  --execute `
  --live `
  --loop
```

Thu tu khuyen nghi:

1. Backend.
2. UI.
3. Trader dry-run.
4. Trader live khi da chac chan.

## 11. Logic bot trade hien tai

Moi loop trader:

1. Doc `trade_state.json`.
2. Neu dang co state blocking thi monitor TP/final sell, khong vao lenh moi.
3. Doc `latest_prediction.json`.
4. Neu khong co ensemble signal -> `NO_SIGNAL`.
5. Neu co signal nhung chua dung phut entry -> `WAITING_ENTRY`.
6. Neu da qua phut entry -> `ENTRY_TIME_PASSED`.
7. Neu dung phut entry:
   - Kiem tra symbol trading.
   - Kiem tra khong co open order tren Binance.
   - Kiem tra quote qty >= min notional.
   - Kiem tra du USDT.
   - Gui MARKET BUY bang `quoteOrderQty`.
8. Neu BUY khong `FILLED` -> khong dat TP, bao loi.
9. Neu BUY `FILLED`:
   - Tinh `avg_entry`.
   - Dat LIMIT SELL TP tai `avg_entry * (1 + TAKE_PROFIT_PCT)`.
10. Neu TP filled -> dong vi the.
11. Neu toi deadline `close(t+h)`:
   - Kiem tra TP co filled chua.
   - Huy TP neu con treo.
   - Market sell phan con lai.
12. Neu final sell qua 5 phut chua confirm -> bao loi can check thu cong.

## 12. Chay live tren VM

Nen dung VM rieng cho live, cau hinh nho cung du:

- `e2-small` hoac `e2-medium` cho live.
- Neu vua live vua train thi dung may lon hon.

SSH:

```powershell
gcloud compute ssh evo-finance-crypto-live --zone asia-southeast1-a
```

Cai project:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip tmux
git clone https://github.com/daovanda/Evo_Finance.git
cd ~/Evo_Finance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tao `.env` tren VM:

```bash
cd ~/Evo_Finance
nano .env
```

Noi dung:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_NOTIFY=1
```

Upload model/archive/data neu can bang `gcloud compute scp`, hoac `git pull` neu da commit model.

Chay backend trong tmux:

```bash
tmux new -s live_backend
cd ~/Evo_Finance
source .venv/bin/activate
python -m crypto.prod.live_backend \
  --model-dir crypto/prod/model/crypto_btc_seed1_12h \
  --data data/crypto/BTCUSDT_15m.csv \
  --out crypto/prod/live/latest_prediction.json \
  --loop \
  --sleep-after-open 5 \
  2>&1 | tee crypto/prod/live/live_backend.log
```

Detach: `Ctrl+B`, roi `D`.

Chay trader trong tmux khac:

```bash
tmux new -s live_trader
cd ~/Evo_Finance
source .venv/bin/activate
python -m crypto.prod.trader \
  --prediction crypto/prod/live/latest_prediction.json \
  --state crypto/prod/live/trade_state.json \
  --execute \
  --live \
  --loop \
  2>&1 | tee crypto/prod/live/trader.log
```

Chay UI neu muon xem tren VM:

```bash
tmux new -s live_ui
cd ~/Evo_Finance
source .venv/bin/activate
python -m crypto.prod.live.app \
  --host 0.0.0.0 \
  --port 8765 \
  --prediction crypto/prod/live/latest_prediction.json \
  --trade-state crypto/prod/live/trade_state.json \
  --model-dir crypto/prod/model/crypto_btc_seed1_12h
```

Neu mo UI public, can cau hinh firewall va nen han che IP. Cach an toan hon la dung SSH tunnel:

```powershell
gcloud compute ssh evo-finance-crypto-live --zone asia-southeast1-a -- -L 8765:127.0.0.1:8765
```

Sau do mo local:

```text
http://127.0.0.1:8765
```

## 13. Tai file live tu VM ve local

Prediction:

```powershell
gcloud compute scp `
  evo-finance-crypto-live:/home/daovanda2405/Evo_Finance/crypto/prod/live/latest_prediction.json `
  D:\Evo_Finance\crypto\prod\live\latest_prediction.json `
  --zone asia-southeast1-a
```

Trade state:

```powershell
gcloud compute scp `
  evo-finance-crypto-live:/home/daovanda2405/Evo_Finance/crypto/prod/live/trade_state.json `
  D:\Evo_Finance\crypto\prod\live\trade_state.json `
  --zone asia-southeast1-a
```

Log:

```powershell
gcloud compute scp `
  evo-finance-crypto-live:/home/daovanda2405/Evo_Finance/crypto/prod/live/trader.log `
  D:\Evo_Finance\crypto\prod\live\trader.log `
  --zone asia-southeast1-a
```

## 14. Tao API key Binance

Tren Binance:

1. Account -> API Management.
2. Create API key.
3. Dat ten vi du `evo-crypto-live`.
4. Bat `Enable Reading`.
5. Bat `Enable Spot & Margin Trading`.
6. Khong bat withdraw.
7. Neu co static IP thi restrict IP cho an toan.
8. Luu API key va secret vao `.env`.

Neu khong tick duoc Spot Trading:

- Kiem tra da xac minh bao mat/2FA.
- Kiem tra API key type co cho trading khong.
- Kiem tra tai khoan co bi restriction khu vuc/san pham khong.
- Tao lai key moi neu key cu bi gioi han.

## 15. Safe checklist truoc khi live

Chay cac lenh nay:

```powershell
python -m crypto.prod.trader --account-test --live
python -m crypto.prod.trader --telegram-test
python -m crypto.prod.live_backend --model-dir crypto/prod/model/crypto_btc_seed1_12h
python -m crypto.prod.trader --once
```

Chi chay live khi:

- `latest_prediction.json` co `entry_candle_time > signal_time`.
- UI hien dung prediction.
- Telegram nhan duoc prediction/status.
- Binance account test OK.
- Khong co open order BTCUSDT thu cong.
- USDT free >= `QUOTE_ORDER_QTY`.
- Ban chap nhan rui ro that tien khi `--execute --live`.

## 16. Loi thuong gap

### Khong co tin hieu 18 gio

Binh thuong neu ensemble can h3 va h5 cung vuot threshold. Kiem tra prediction:

```powershell
Get-Content crypto/prod/live/latest_prediction.json
```

Neu prediction gan threshold nhung khong vuot, bot se `NO TRADE`.

### `Timestamp outside recvWindow`

Code da tu sync server time va retry. Neu van loi, sync gio may:

```powershell
w32tm /resync
```

Tren Linux:

```bash
timedatectl
```

### `Insufficient free USDT balance`

Nap them USDT vao Spot wallet hoac giam:

```powershell
--quote-order-qty 5.5
```

### Co open order nen bot khong vao lenh

Bot se chan de tranh mo lenh moi khi tai khoan co lenh treo ngoai state. Kiem tra tren Binance Spot -> Open Orders, huy lenh thu cong neu can.

### State bi block

Doc:

```powershell
Get-Content crypto/prod/live/trade_state.json
```

Chi xoa state khi chac chan khong con vi the/lenh treo:

```powershell
Remove-Item crypto/prod/live/trade_state.json -ErrorAction SilentlyContinue
```

