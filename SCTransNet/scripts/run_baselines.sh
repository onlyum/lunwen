#!/usr/bin/env bash
# P1: DenseSIRST baseline 实验（GPU 4，快速堆 epoch 模式）
set -euo pipefail
cd "$(dirname "$0")/.."
source /mnt/data/lvyongli/envs/miniconda3/bin/activate lunwen

DATASET_DIR=./datasets
GPU=4
EPOCHS=${EPOCHS:-200}
LOG_ROOT=./log/baselines

# 默认验证策略（与 train.py 一致，可省略）：
# begin_val=10, val_interval=10, val_max_samples=64
# full_val_interval=50, test_interval=50, threshold_sweep_interval=0

run_one() {
  local name=$1
  shift
  echo "========== START $name $(date '+%F %T') =========="
  python train.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_names DenseSIRST \
    --gpu "$GPU" \
    --epochs "$EPOCHS" \
    --save "$LOG_ROOT" \
    --experiment_name "$name" \
    --batchSize 16 \
    --early_stop_patience 80 \
    --min_epochs 50 \
    "$@" \
    2>&1 | tee -a "$LOG_ROOT/${name}.console.log"
  echo "========== DONE $name $(date '+%F %T') =========="
}

mkdir -p "$LOG_ROOT"

run_one baseline_lr1e4 --loss_mode bce --lr 0.0001
run_one baseline_weighted_bce --loss_mode weighted_bce --lr 0.0001 --pos_weight 80
run_one baseline_bce_dice --loss_mode bce_dice --lr 0.0001 --bce_weight 1.0 --dice_weight 1.0
run_one baseline_tversky --loss_mode tversky --lr 0.0001 --tversky_alpha 0.7 --tversky_beta 0.3
run_one baseline_focal_tversky --loss_mode focal_tversky --lr 0.0001 --focal_gamma 0.75

echo "All baselines finished."
