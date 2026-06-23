#!/usr/bin/env bash
# P1: DenseSIRST 五个 baseline（GPU 4，v2 策略，顺序执行）
set -euo pipefail
cd "$(dirname "$0")/.."
export CONFIG_TAG=${CONFIG_TAG:-v2_begin500_val10_test50}

bash scripts/run_single_baseline.sh baseline_original_bce --loss_mode bce --lr 0.001
bash scripts/run_single_baseline.sh baseline_lr1e4 --loss_mode bce --lr 0.0001
bash scripts/run_single_baseline.sh baseline_bce_dice --loss_mode bce_dice --lr 0.0001 --bce_weight 1.0 --dice_weight 1.0
bash scripts/run_single_baseline.sh baseline_tversky --loss_mode tversky --lr 0.0001 --tversky_alpha 0.7 --tversky_beta 0.3
bash scripts/run_single_baseline.sh baseline_focal_tversky --loss_mode focal_tversky --lr 0.0001 --focal_gamma 0.75

echo "All 5 baselines finished."
