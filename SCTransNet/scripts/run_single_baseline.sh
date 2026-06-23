#!/usr/bin/env bash
# 训练单个 baseline（v2 策略）并在结束后 test + 记录
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/data/lvyongli/envs/miniconda3/bin/activate lunwen

NAME=${1:?experiment_name}
shift

DATASET_DIR=./datasets
GPU=${GPU:-4}
EPOCHS=${EPOCHS:-1000}
LOG_ROOT=./log/baselines
CONFIG_TAG=${CONFIG_TAG:-v2_begin500_val10_test50}

VAL_ARGS=(
  --begin_val 500
  --val_interval 10
  --val_max_samples 0
  --test_interval 50
)

find_best_ckpt() {
  local exp_dir="$LOG_ROOT/DenseSIRST/$1"
  ls -t "$exp_dir"/SCTransNet_*_best.pth.tar 2>/dev/null | head -1
}

echo "========== START $NAME epochs=$EPOCHS config=$CONFIG_TAG $(date '+%F %T') =========="
python -u train.py \
  --dataset_dir "$DATASET_DIR" \
  --dataset_names DenseSIRST \
  --gpu "$GPU" \
  --epochs "$EPOCHS" \
  --save "$LOG_ROOT" \
  --experiment_name "$NAME" \
  --batchSize 16 \
  "${VAL_ARGS[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG_ROOT/${NAME}.console.log"

CKPT=$(find_best_ckpt "$NAME")
if [[ -z "${CKPT:-}" ]]; then
  echo "ERROR: no best checkpoint for $NAME" >&2
  exit 1
fi
bash scripts/run_test_record.sh "$NAME" "$CKPT" "$CONFIG_TAG"
echo "========== DONE $NAME $(date '+%F %T') =========="
