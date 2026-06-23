#!/usr/bin/env bash
# 对 best checkpoint 跑 test_v2，结果追加到 CSV 并保存独立 log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /mnt/data/lvyongli/envs/miniconda3/bin/activate lunwen

EXPERIMENT=${1:?experiment_name}
CKPT=${2:?checkpoint_path}
CONFIG_TAG=${3:-unknown}
GPU=${GPU:-4}

RECORD_DIR="$ROOT/log/baselines/test_records"
mkdir -p "$RECORD_DIR"
CSV="$RECORD_DIR/test_results.csv"
TS=$(date '+%F %T')
LOG="$RECORD_DIR/${EXPERIMENT}_${CONFIG_TAG}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" | tee -a "$LOG"
  exit 1
fi

# test.py 用 pth_dirs 第一段作 dataset 名，且 pth_dir = save_log + pth_dirs
# baseline checkpoint 在 log/baselines/DenseSIRST/...，需对齐 save_log 与相对路径
if [[ "$CKPT" == "$ROOT/log/baselines/"* ]]; then
  REL=${CKPT#"$ROOT/log/baselines/"}
  SAVE_LOG=./log/baselines/
elif [[ "$CKPT" == "$ROOT/log/"* ]]; then
  REL=${CKPT#"$ROOT/log/"}
  SAVE_LOG=./log/
else
  echo "ERROR: checkpoint must be under $ROOT/log/" | tee -a "$LOG"
  exit 1
fi

echo "[$TS] TEST $EXPERIMENT config=$CONFIG_TAG ckpt=$CKPT" | tee "$LOG"
echo "pth_dirs=$REL save_log=$SAVE_LOG" | tee -a "$LOG"

export CUDA_VISIBLE_DEVICES="$GPU"
python -u test.py \
  --dataset_dir ./datasets \
  --dataset_names DenseSIRST \
  --test_split test_v2 \
  --save_log "$SAVE_LOG" \
  --pth_dirs "$REL" \
  --threshold 0.5 \
  2>&1 | tee -a "$LOG"

METRIC_LINE=$(grep -E '^pixAcc:' "$LOG" | tail -1 || true)
if [[ -z "$METRIC_LINE" ]]; then
  echo "ERROR: no metrics line in test output" | tee -a "$LOG"
  exit 1
fi

read -r pixAcc mIoU nIoU PD FA F1 <<< "$(python3 - <<PY
import re
line = '''$METRIC_LINE'''
m = re.search(
    r'pixAcc:\s*([\d.]+)\|\s*mIoU:\s*([\d.]+)\s*\|\s*nIoU:\s*([\d.]+)\s*\|\s*Pd:\s*([\d.]+)\|\s*Fa:\s*([\d.]+)\s*\|F1:\s*([\d.]+)',
    line)
if not m:
    raise SystemExit('parse failed')
print(*m.groups())
PY
)"

BEST_EPOCH=$(basename "$CKPT" | sed -n 's/SCTransNet_\([0-9]*\)_best\.pth\.tar/\1/p')

if [[ ! -f "$CSV" ]]; then
  echo "timestamp,experiment,config_tag,checkpoint,best_epoch,pixAcc,mIoU,nIoU,PD,FA,F1,log_path" > "$CSV"
fi
echo "$TS,$EXPERIMENT,$CONFIG_TAG,$CKPT,$BEST_EPOCH,$pixAcc,$mIoU,$nIoU,$PD,$FA,$F1,$LOG" >> "$CSV"
echo "RECORDED -> $CSV" | tee -a "$LOG"
