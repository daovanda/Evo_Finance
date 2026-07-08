# Guild Evolution Crypto

File nay huong dan chay tien hoa feature BTC/USDT trong thu muc `crypto/`.

Mac dinh pipeline dung:

- Data: `data/crypto/BTCUSDT_15m.csv`
- Entry point: `python -m crypto.main`
- Config chinh: `crypto/config.py`
- Archive output: `crypto/results/*.json`
- Checkpoint: neu save la `name.json` thi checkpoint la `name.checkpoint.json`
- Timezone trong CSV: gio Viet Nam `Asia/Ho_Chi_Minh`
- Nen mac dinh: `15m`

## 1. Chuan bi local

Tu thu muc project:

```powershell
cd D:\Evo_Finance
```

Kiem tra Python co import duoc module:

```powershell
python -m crypto.main --help
```

Neu chua co data BTC:

```powershell
python craw_btc.py --symbol BTCUSDT --interval 15m --start "2018-01-01 00:00:00+07:00" --out data/crypto/BTCUSDT_15m.csv
```

Data se co cac cot chinh:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `trade_count`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`
- `is_trading_day`

## 2. Config quan trong

Sua trong `crypto/config.py`.

### Horizon va label

```python
HOLDING_HORIZONS = [3, 5]
LABEL_THRESHOLD = 0.001
LABEL_MODE = "close_exit"  # "close_exit" hoac "mfe"
```

Nghia la tao label cho `h3` va `h5`; label = 1 khi future return vuot `0.1%`.

`LABEL_MODE="close_exit"`:

```text
future_return(t, h) = (close(t+h) - open(t+1)) / open(t+1)
```

`LABEL_MODE="mfe"`:

```text
future_return(t, h) = (max(high(t+1)..high(t+h)) - open(t+1)) / open(t+1)
```

Mode `mfe` phu hop hon voi chien luoc dat TP, vi chi can gia cham muc loi trong horizon la label duong.

Co the override label mode bang CLI, khong can sua file config:

```powershell
python -m crypto.main --label-mode mfe --help
```

### Final split

```python
VAL_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = None
```

Dung khi het budget de danh gia final tren val/test.

### Walk-forward evolution

```python
WF_END = TEST_START
WF_MIN_TRAIN_MONTHS = 36
WF_VAL_MONTHS = 6
WF_STEP_MONTHS = 6
WF_PURGE_BARS = None
```

`WF_PURGE_BARS=None` tu dong dung `max(HOLDING_HORIZONS) + 1`.

### Feature va corr

```python
WINDOWS = [3, 5, 7, 10, 14, 20, 30, 40, 50, 60, 80, 120, 160, 240, 320, 400, 480]
FEATURE_CORR_THRESHOLD = 0.70
```

Tat ca window tinh theo so nen, khong phai so ngay. Voi data `15m`, `480` nen xap xi 5 ngay.

### Checkpoint

```python
CHECKPOINT_EVERY_SECONDS = 12 * 60 * 60
```

Co the override bang CLI `--checkpoint-every 3600`.

## 3. Chay tien hoa tren local

### Chay thu 30 phut

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 1800 `
  --seed 1 `
  --label-mode mfe `
  --save crypto/results/crypto_btc_seed1_30m.json `
  --checkpoint-every 600
```

### Chay 5 gio

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 18000 `
  --seed 1 `
  --label-mode mfe `
  --save crypto/results/crypto_btc_seed1_5h.json `
  --checkpoint-every 3600
```

### Chay 12 gio

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 43200 `
  --seed 1 `
  --label-mode mfe `
  --save crypto/results/crypto_btc_seed1_12h.json `
  --checkpoint-every 3600
```

### Resume tu checkpoint

Ghi de vao cung file output:

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 43200 `
  --seed 1 `
  --resume crypto/results/crypto_btc_seed1_12h.checkpoint.json `
  --label-mode mfe `
  --save crypto/results/crypto_btc_seed1_12h.json `
  --checkpoint-every 3600
```

### Chay voi horizon khac

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 18000 `
  --seed 2 `
  --horizons 3,5,10 `
  --label-mode mfe `
  --label-threshold 0.001 `
  --save crypto/results/crypto_btc_seed2_h3_h5_h10.json
```

## 4. Doc log khi chay local

Neu muon ghi log ra file:

```powershell
python -m crypto.main `
  --data data/crypto/BTCUSDT_15m.csv `
  --budget 18000 `
  --seed 1 `
  --save crypto/results/crypto_btc_seed1_5h.json `
  --checkpoint-every 3600 `
  2>&1 | Tee-Object crypto/results/run_crypto_btc_seed1_5h.log
```

## 5. Tao VM tren Google Cloud

Goi y cau hinh:

- Machine type: `e2-standard-4` hoac cao hon
- Zone: `asia-southeast1-a`
- Boot disk: Ubuntu 24.04 LTS, 50 GB
- Firewall: khong can mo port neu chi chay tien hoa

Tao VM bang giao dien:

1. Google Cloud Console -> Compute Engine -> VM instances.
2. Create instance.
3. Name vi du: `evo-finance-crypto-seed1`.
4. Region/zone: `asia-southeast1-a`.
5. Machine: `e2-standard-4`.
6. Boot disk: Ubuntu 24.04 LTS, 50 GB.
7. Create.

Hoac tao bang CLI:

```powershell
gcloud compute instances create evo-finance-crypto-seed1 `
  --zone asia-southeast1-a `
  --machine-type e2-standard-4 `
  --image-family ubuntu-2404-lts-amd64 `
  --image-project ubuntu-os-cloud `
  --boot-disk-size 50GB
```

## 6. Cai project tren VM

SSH vao VM:

```powershell
gcloud compute ssh evo-finance-crypto-seed1 --zone asia-southeast1-a
```

Trong VM:

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

Neu repo da co san:

```bash
cd ~/Evo_Finance
git pull
source .venv/bin/activate
pip install -r requirements.txt
```

Kiem tra:

```bash
python -m crypto.main --help
```

## 7. Chay tien hoa tren VM bang tmux

Tao session:

```bash
tmux new -s crypto_seed1
```

Trong tmux:

```bash
cd ~/Evo_Finance
source .venv/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p crypto/results

python -m crypto.main \
  --data data/crypto/BTCUSDT_15m.csv \
  --budget 43200 \
  --seed 1 \
  --save crypto/results/crypto_btc_seed1_12h.json \
  --checkpoint-every 3600 \
  2>&1 | tee crypto/results/run_crypto_btc_seed1_12h.log
```

Thoat tmux nhung van giu process chay:

```text
Ctrl+B, sau do bam D
```

Vao lai tmux:

```bash
tmux attach -t crypto_seed1
```

Xem log:

```bash
tail -f ~/Evo_Finance/crypto/results/run_crypto_btc_seed1_12h.log
```

Kiem tra process:

```bash
pgrep -af 'python.*crypto.main'
```

## 8. Chay nhieu seed tren nhieu VM

VM seed 2:

```bash
tmux new -s crypto_seed2
cd ~/Evo_Finance
source .venv/bin/activate
mkdir -p crypto/results
python -m crypto.main \
  --data data/crypto/BTCUSDT_15m.csv \
  --budget 43200 \
  --seed 2 \
  --save crypto/results/crypto_btc_seed2_12h.json \
  --checkpoint-every 3600 \
  2>&1 | tee crypto/results/run_crypto_btc_seed2_12h.log
```

VM seed 3:

```bash
tmux new -s crypto_seed3
cd ~/Evo_Finance
source .venv/bin/activate
mkdir -p crypto/results
python -m crypto.main \
  --data data/crypto/BTCUSDT_15m.csv \
  --budget 43200 \
  --seed 3 \
  --save crypto/results/crypto_btc_seed3_12h.json \
  --checkpoint-every 3600 \
  2>&1 | tee crypto/results/run_crypto_btc_seed3_12h.log
```

## 9. Resume tren VM

```bash
cd ~/Evo_Finance
source .venv/bin/activate

python -m crypto.main \
  --data data/crypto/BTCUSDT_15m.csv \
  --budget 43200 \
  --seed 1 \
  --resume crypto/results/crypto_btc_seed1_12h.checkpoint.json \
  --save crypto/results/crypto_btc_seed1_12h.json \
  --checkpoint-every 3600 \
  2>&1 | tee -a crypto/results/run_crypto_btc_seed1_12h.log
```

## 10. Tai archive/log tu VM ve local

Chay tren may Windows local, khong chay trong VM.

Tai archive:

```powershell
gcloud compute scp `
  evo-finance-crypto-seed1:/home/daovanda2405/Evo_Finance/crypto/results/crypto_btc_seed1_12h.json `
  D:\Evo_Finance\crypto\results\crypto_btc_seed1_12h.json `
  --zone asia-southeast1-a
```

Tai checkpoint:

```powershell
gcloud compute scp `
  evo-finance-crypto-seed1:/home/daovanda2405/Evo_Finance/crypto/results/crypto_btc_seed1_12h.checkpoint.json `
  D:\Evo_Finance\crypto\results\crypto_btc_seed1_12h.checkpoint.json `
  --zone asia-southeast1-a
```

Tai log:

```powershell
gcloud compute scp `
  evo-finance-crypto-seed1:/home/daovanda2405/Evo_Finance/crypto/results/run_crypto_btc_seed1_12h.log `
  D:\Evo_Finance\crypto\results\run_crypto_btc_seed1_12h.log `
  --zone asia-southeast1-a
```

Neu bi `permission denied`, sua owner tren VM:

```powershell
gcloud compute ssh evo-finance-crypto-seed1 --zone asia-southeast1-a --command "sudo chown -R `$USER:`$USER /home/daovanda2405/Evo_Finance/crypto/results && chmod -R u+rwX /home/daovanda2405/Evo_Finance/crypto/results"
```

Sau do chay lai `gcloud compute scp`.

## 11. Kiem tra archive sau khi tai ve

```powershell
python -m crypto.analyze `
  --archive crypto/results/crypto_btc_seed1_12h.json `
  --rank 1
```

Bieu do se luu trong:

```text
crypto/results/chart/
```

## 12. Loi thuong gap

### `python: command not found` tren VM

Dung:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m crypto.main --help
```

Sau khi activate `.venv`, lenh `python` se dung duoc.

### tmux session bien mat

Kiem tra process:

```bash
pgrep -af 'python.*crypto.main'
```

Kiem tra log cuoi:

```bash
tail -80 ~/Evo_Finance/crypto/results/run_crypto_btc_seed1_12h.log
```

Neu co checkpoint thi resume tu checkpoint.

### Archive khong co final metrics

Final metrics chi duoc tinh sau khi het budget. Neu process crash truoc khi het budget, checkpoint co the chua co final val/test metrics. Resume tiep hoac chay budget ngan de ket thuc va ghi final.
