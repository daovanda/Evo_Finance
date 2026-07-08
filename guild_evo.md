# Guild Evo Finance - Stock VN

Tai lieu nay huong dan chay pipeline tien hoa feature cho phan co phieu Viet Nam
trong repo `Evo_Finance`. Phan crypto co tai lieu rieng trong `crypto/`.

## 1. Cau truc nhanh

Pipeline stock:

```text
data/raw/*.csv
  -> load OHLCV + market index
  -> label theo tung ticker
  -> tao walk-forward folds
  -> mutate individual
  -> train LightGBM lambdarank
  -> tinh fitness tren WF validation
  -> luu archive
  -> analyze archive va ve chart
```

Mot `individual` la mot tap feature. Moi feature trong individual la mot `gene`.
Archive luu nhung individual co fitness tot nhat.

## 2. Cau hinh quan trong

Tat ca knob chinh nam trong `config/settings.py`.

### 2.1. Final split

Hien tai:

```python
VAL_START  = "2023-05-12"
TEST_START = "2025-01-01"
TEST_END   = None
```

Y nghia:

```text
final train: dau data -> truoc VAL_START
final val:   VAL_START -> truoc TEST_START
final test:  TEST_START -> het data neu TEST_END=None
```

Final split chi dung sau khi het budget de danh gia lai cuoi cung. Test khong
duoc dung trong vong lap tien hoa.

### 2.2. Label

Hien tai:

```python
HOLDING_HORIZON = 10
label(t) = (close(t+10) - open(t+1)) / open(t+1)
```

Tuc la tai ngay `t`, model chi dung du lieu den close `t`; entry gia dinh la
open ngay `t+1`, exit la close ngay `t+10`.

### 2.3. Walk-forward trong tien hoa

Hien tai:

```python
WF_END = TEST_START
WF_MIN_TRAIN_MONTHS = 48
WF_VAL_MONTHS = 6
WF_STEP_MONTHS = 6
WF_PURGE_DAYS = HOLDING_HORIZON
```

Y nghia:

```text
WF chi dung data truoc TEST_START.
Moi fold co train dang expanding tu dau data.
Moi fold val dai 6 thang.
Fold tiep theo dich them 6 thang.
Truoc val co purge 10 ngay giao dich de tranh label leak.
```

Trong moi WF fold, stock dang chia them trong `fold train`:

```python
WF_EARLY_STOP_VALID_FRACTION = 0.20
WF_EARLY_STOP_MIN_VALID_DATES = 20
```

Y nghia:

```text
fit_train      = phan dau cua fold train
early_stop_set = phan cuoi cua fold train
fold_val       = chi dung de tinh fitness
```

So ngay early stop:

```text
n_stop = max(20% so ngay trong fold train, 20 ngay)
n_stop khong vuot qua 50% fold train
```

Vi du:

```text
fold train co 500 ngay -> early-stop lay 100 ngay cuoi
fold train co 60 ngay  -> early-stop lay 20 ngay cuoi
fold train co 30 ngay  -> early-stop lay 15 ngay cuoi do bi chan 50%
```

### 2.4. Market index va sector

Market index:

```python
MARKET_INDEX_TICKER = "VNINDEX"
```

Loader se doc file:

```text
data/raw/VNINDEX.csv
```

Sau do tao cac cot `market_*` cho tung dong co phieu. Neu muon dung chi so khac,
them file CSV tuong ung vao `data/raw/` va doi `MARKET_INDEX_TICKER`.

Sector nam trong:

```python
SECTORS = {...}
REQUIRE_SECTOR_MAPPING = True
MIN_SECTOR_MEMBERS_IN_UNIVERSE = 2
```

Khi them ticker moi, can cap nhat `SECTORS` neu van bat
`REQUIRE_SECTOR_MAPPING=True`.

### 2.5. Feature va mutator

Gioi han so feature trong individual:

```python
FEATURE_MIN = 3
FEATURE_MAX = 30
```

Window whitelist:

```python
WINDOWS = [3, 5, 10, 14, 20, 30, 60, 120]
```

Correlation guard:

```python
CORR_THRESHOLD = 0.70
CORR_CHECK_MAX_ROWS = 0
DOMAIN_CORR_MAX_CHECKS = 0
DOMAIN_PRECOMPUTE_ON_START = True
```

`CORR_THRESHOLD` dung de loc feature qua giong nhau trong domain va trong
individual. Hien tai dang loc ca hai dau: tuong quan duong qua cao va am qua cao
deu bi loai theo tri tuyet doi.

Co 2 lop corr-check:

```text
individual corr-check:
  Gene moi van check voi toan bo feature con lai trong individual.
  Day la lop quan trong nhat de tranh individual bi trung feature.

domain corr-check:
  Neu gene moi chua co trong domain, he thu them gene do vao domain.
  Luc do gene moi duoc check voi domain de tranh domain phinh ra boi cac ban sao.
```

Y nghia cac bien gioi han:

```text
CORR_CHECK_MAX_ROWS = 0
  0  = dung toan bo valid rows de tinh corr.
  >0 = lay sample co dinh toi da N rows de tinh nhanh hon.

DOMAIN_CORR_MAX_CHECKS = 0
  0  = check voi toan bo domain.
  >0 = chi check toi da N formula trong domain.

DOMAIN_PRECOMPUTE_ON_START = True
  Tinh san feature domain luc bat dau. Khoi dong lau hon va ton RAM hon,
  nhung sau do domain corr-check do bi giat/khoi tao lazy hon.
```

Neu muon chay nhanh hon, co the dung vi du:

```python
CORR_CHECK_MAX_ROWS = 12000
DOMAIN_CORR_MAX_CHECKS = 40
DOMAIN_PRECOMPUTE_ON_START = False
```

Neu muon corr full chat nhat nhu cau hinh hien tai, giu:

```python
CORR_CHECK_MAX_ROWS = 0
DOMAIN_CORR_MAX_CHECKS = 0
DOMAIN_PRECOMPUTE_ON_START = True
```

### 2.6. LightGBM

Walk-forward va final dang co hai bo config rieng:

```python
LGBM_WF_PARAMS
LGBM_WF_NUM_BOOST_ROUND = 250
LGBM_WF_EARLY_STOPPING = 20

LGBM_FINAL_PARAMS
LGBM_FINAL_NUM_BOOST_ROUND = 250
LGBM_FINAL_EARLY_STOPPING = 20
```

Hien tai ca hai bo tham so dang gan giong nhau:

```python
objective = "lambdarank"
metric = "ndcg"
eval_at = [10]
learning_rate = 0.03
num_leaves = 15
max_depth = 4
feature_fraction = 0.6
bagging_fraction = 0.7
bagging_freq = 1
min_data_in_leaf = 300
lambda_l1 = 5.0
lambda_l2 = 20.0
```

## 3. Chay tren local Windows

### 3.1. Cai moi moi truong

Chay trong PowerShell tai thu muc repo:

```powershell
cd D:\Evo_Finance
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu da co `.venv`, chi can:

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1
```

### 3.2. Chay thu nhanh 10 phut

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1

$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

python main.py `
  --data-dir data/raw `
  --budget 600 `
  --seed 1 `
  --save results/archive_stock_smoke_seed1.json `
  --checkpoint-every 300
```

File tao ra:

```text
results/archive_stock_smoke_seed1.json
results/archive_stock_smoke_seed1.checkpoint.json
```

### 3.3. Chay 12 gio tren local

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1

$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

python main.py `
  --data-dir data/raw `
  --budget 43200 `
  --seed 1 `
  --save results/archive_stock_seed1_12h.json `
  --checkpoint-every 3600
```

`--checkpoint-every 3600` nghia la moi 1 gio luu checkpoint mot lan.

### 3.4. Chay tiep tu checkpoint hoac archive

Lenh resume se doc archive cu, danh gia lai theo WF hien tai, roi chay tiep.

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1

python main.py `
  --data-dir data/raw `
  --budget 43200 `
  --seed 1 `
  --resume results/archive_stock_seed1_12h.checkpoint.json `
  --save results/archive_stock_seed1_12h.json `
  --checkpoint-every 3600
```

Neu muon ghi sang file moi:

```powershell
python main.py `
  --data-dir data/raw `
  --budget 43200 `
  --seed 1 `
  --resume results/archive_stock_seed1_12h.checkpoint.json `
  --save results/archive_stock_seed1_12h_resume.json `
  --checkpoint-every 3600
```

### 3.5. Chay nhieu seed

Chay moi seed ra mot file rieng:

```powershell
python main.py --data-dir data/raw --budget 43200 --seed 1 --save results/archive_stock_seed1_12h.json --checkpoint-every 3600
python main.py --data-dir data/raw --budget 43200 --seed 2 --save results/archive_stock_seed2_12h.json --checkpoint-every 3600
python main.py --data-dir data/raw --budget 43200 --seed 3 --save results/archive_stock_seed3_12h.json --checkpoint-every 3600
```

Tren mot may ca nhan, khong nen chay qua nhieu seed song song neu CPU/RAM yeu.

## 4. Analyze archive tren local

`analyze.py` se load archive, train lai cac individual duoc chon, tinh lai metric
va ve chart vao `results/chart/`.

### 4.1. Analyze top 10

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1

python analyze.py `
  --data-dir data/raw `
  --archive results/archive_stock_seed1_12h.json `
  --top 10 `
  --out-dir results/chart/stock_seed1_12h_top10
```

### 4.2. Analyze mot vai rank cu the

```powershell
python analyze.py `
  --data-dir data/raw `
  --archive results/archive_stock_seed1_12h.json `
  --rank 1 5 12 `
  --out-dir results/chart/stock_seed1_selected
```

`--rank` uu tien hon `--top`.

### 4.3. Analyze voi split tuy chinh

Neu luc chay evolution co override split, analyze cung phai dung lai dung split do:

```powershell
python analyze.py `
  --data-dir data/raw `
  --archive results/archive_custom_split.json `
  --val-start 2023-05-12 `
  --test-start 2025-01-01 `
  --wf-end 2025-01-01 `
  --wf-min-train-months 48 `
  --wf-val-months 6 `
  --wf-step-months 6 `
  --wf-purge-days 10 `
  --top 10 `
  --out-dir results/chart/custom_split_top10
```

## 5. Chay tren Google Cloud VM

Vi du duoi day dung Ubuntu/Debian VM, may `e2-standard-4` la muc hop ly de chay
stock 1 seed. Co the chay nhieu VM cho nhieu seed.

### 5.1. SSH vao VM

Tu may local:

```powershell
gcloud config set project inance-499913
gcloud config set compute/zone asia-southeast1-a
gcloud compute ssh evo-finance-stock-seed1 --zone asia-southeast1-a
```

Thay `evo-finance-stock-seed1` bang ten VM thuc te.

### 5.2. Cai moi truong tren VM

Neu VM moi tao, doi cloud-init/apt xong truoc:

```bash
sudo cloud-init status --wait
sudo apt update
sudo apt install -y git python3-venv python3-pip python-is-python3 tmux
```

Clone repo va cai package:

```bash
cd ~
git clone https://github.com/daovanda/Evo_Finance.git
cd ~/Evo_Finance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p results
```

Neu repo da ton tai:

```bash
cd ~/Evo_Finance
git pull
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.3. Chay trong tmux

Tao session:

```bash
tmux new -s stock_seed1
```

Trong tmux:

```bash
cd ~/Evo_Finance
source .venv/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results

python main.py \
  --data-dir data/raw \
  --budget 259200 \
  --seed 1 \
  --save results/archive_stock_gcp_e2s4_seed1_3days.json \
  --checkpoint-every 3600 \
  2>&1 | tee results/run_stock_gcp_e2s4_seed1_3days.log
```

`259200` giay = 3 ngay. Neu muon 12 gio:

```bash
python main.py \
  --data-dir data/raw \
  --budget 43200 \
  --seed 1 \
  --save results/archive_stock_gcp_e2s4_seed1_12h.json \
  --checkpoint-every 3600 \
  2>&1 | tee results/run_stock_gcp_e2s4_seed1_12h.log
```

Detach tmux:

```text
Ctrl+B roi bam D
```

Attach lai:

```bash
tmux attach -t stock_seed1
```

Xem log khi dang chay:

```bash
tail -f ~/Evo_Finance/results/run_stock_gcp_e2s4_seed1_3days.log
```

Kiem tra process:

```bash
pgrep -af 'python.*main.py'
tmux ls
```

### 5.4. Resume tren VM

Neu co checkpoint:

```bash
cd ~/Evo_Finance
source .venv/bin/activate

python main.py \
  --data-dir data/raw \
  --budget 43200 \
  --seed 1 \
  --resume results/archive_stock_gcp_e2s4_seed1_3days.checkpoint.json \
  --save results/archive_stock_gcp_e2s4_seed1_3days.json \
  --checkpoint-every 3600 \
  2>&1 | tee -a results/run_stock_gcp_e2s4_seed1_3days.log
```

Muon ghi de file archive cu thi giu cung `--save`. Muon tao file moi thi doi ten
file trong `--save`.

### 5.5. Chay seed khac tren VM khac

Tren VM seed 2:

```bash
tmux new -s stock_seed2

cd ~/Evo_Finance
source .venv/bin/activate
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python main.py \
  --data-dir data/raw \
  --budget 259200 \
  --seed 2 \
  --save results/archive_stock_gcp_e2s4_seed2_3days.json \
  --checkpoint-every 3600 \
  2>&1 | tee results/run_stock_gcp_e2s4_seed2_3days.log
```

Tren VM seed 3 chi can doi `--seed 3` va ten file thanh `seed3`.

## 6. Tai file tu VM ve local

Chay tren PowerShell local, khong phai trong SSH.

### 6.1. Tai archive cuoi

```powershell
gcloud compute scp `
  evo-finance-stock-seed1:/home/daovanda2405/Evo_Finance/results/archive_stock_gcp_e2s4_seed1_3days.json `
  D:\Evo_Finance\results\archive_stock_gcp_e2s4_seed1_3days_from_vm.json `
  --zone asia-southeast1-a
```

### 6.2. Tai checkpoint

```powershell
gcloud compute scp `
  evo-finance-stock-seed1:/home/daovanda2405/Evo_Finance/results/archive_stock_gcp_e2s4_seed1_3days.checkpoint.json `
  D:\Evo_Finance\results\archive_stock_gcp_e2s4_seed1_3days.checkpoint.from_vm.json `
  --zone asia-southeast1-a
```

### 6.3. Tai log

```powershell
gcloud compute scp `
  evo-finance-stock-seed1:/home/daovanda2405/Evo_Finance/results/run_stock_gcp_e2s4_seed1_3days.log `
  D:\Evo_Finance\results\run_stock_gcp_e2s4_seed1_3days_from_vm.log `
  --zone asia-southeast1-a
```

### 6.4. Neu bi permission denied

Chay lenh nay tren PowerShell local:

```powershell
gcloud compute ssh evo-finance-stock-seed1 `
  --zone asia-southeast1-a `
  --command "sudo chown -R daovanda2405:daovanda2405 /home/daovanda2405/Evo_Finance/results && chmod -R u+rwX /home/daovanda2405/Evo_Finance/results"
```

Sau do chay lai `gcloud compute scp`.

Neu user VM khong phai `daovanda2405`, thay bang user dung tren VM. Co the xem user
bang:

```bash
whoami
```

## 7. Checklist truoc khi chay dai

1. `data/raw/` co du CSV ticker va `VNINDEX.csv`.
2. `MARKET_INDEX_TICKER` dung voi file market index muon dung.
3. `SECTORS` da cap nhat neu them ticker moi.
4. `VAL_START`, `TEST_START`, `WF_END` dung voi muc tieu test.
5. `WF_PURGE_DAYS >= HOLDING_HORIZON`.
6. Chay smoke 10 phut truoc khi chay 12 gio/3 ngay.
7. Luon dat `--save` de co archive va checkpoint.
8. Dung ten file khac nhau cho moi seed.
9. Khi resume, archive se duoc danh gia lai theo code/config hien tai.
10. Sau khi lay archive ve, chay `analyze.py` de xem chart truoc khi chon individual.

## 8. Lenh mau day du

### Local 12 gio + analyze top 10

```powershell
cd D:\Evo_Finance
.\.venv\Scripts\Activate.ps1

$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

python main.py `
  --data-dir data/raw `
  --budget 43200 `
  --seed 1 `
  --save results/archive_stock_seed1_12h.json `
  --checkpoint-every 3600

python analyze.py `
  --data-dir data/raw `
  --archive results/archive_stock_seed1_12h.json `
  --top 10 `
  --out-dir results/chart/stock_seed1_12h_top10
```

### VM 3 ngay seed 1

```bash
cd ~/Evo_Finance
source .venv/bin/activate

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python main.py \
  --data-dir data/raw \
  --budget 259200 \
  --seed 1 \
  --save results/archive_stock_gcp_e2s4_seed1_3days.json \
  --checkpoint-every 3600 \
  2>&1 | tee results/run_stock_gcp_e2s4_seed1_3days.log
```

### Local tai file sau khi VM chay xong

```powershell
gcloud compute scp `
  evo-finance-stock-seed1:/home/daovanda2405/Evo_Finance/results/archive_stock_gcp_e2s4_seed1_3days.json `
  D:\Evo_Finance\results\archive_stock_gcp_e2s4_seed1_3days_from_vm.json `
  --zone asia-southeast1-a

gcloud compute scp `
  evo-finance-stock-seed1:/home/daovanda2405/Evo_Finance/results/run_stock_gcp_e2s4_seed1_3days.log `
  D:\Evo_Finance\results\run_stock_gcp_e2s4_seed1_3days_from_vm.log `
  --zone asia-southeast1-a
```
