#!/usr/bin/env bash
# v1 训练（begin_val=500,val_interval=1,test_interval=0）结束后：
# 1) test best 并记录  2) 归档 v1  3) 按新策略重跑前两个 baseline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source /mnt/data/lvyongli/envs/miniconda3/bin/activate lunwen

STATE_DIR="$ROOT/log/baselines/orchestrator"
mkdir -p "$STATE_DIR"
STATE="$STATE_DIR/pipeline_state.txt"
LOG="$STATE_DIR/pipeline.log"
LOCK="$STATE_DIR/pipeline.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "orchestrator already running" >&2
  exit 0
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

mark() {
  echo "$1" > "$STATE"
  log "STATE -> $1"
}

wait_pid() {
  local pid=$1
  local label=$2
  log "waiting for $label pid=$pid ..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "$label finished (pid=$pid)"
}

find_best_ckpt() {
  local exp=$1
  ls -t "$ROOT/log/baselines/DenseSIRST/$exp"/SCTransNet_*_best.pth.tar 2>/dev/null | head -1
}

archive_v1() {
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  local src="$ROOT/log/baselines/DenseSIRST/baseline_original_bce"
  local dst="$ROOT/log/baselines/DenseSIRST/baseline_original_bce_v1_val1_${ts}"
  if [[ -d "$src" ]]; then
    mv "$src" "$dst"
    log "archived v1 checkpoints -> $dst"
  fi
}

# --- main ---
V1_PID=${V1_PID:-3941551}
V1_EXP=baseline_original_bce
V1_CONFIG=v1_begin500_val1_test0

if [[ -f "$STATE" ]] && grep -q '^v2_done$' "$STATE" 2>/dev/null; then
  log "pipeline already complete"
  exit 0
fi

if [[ ! -f "$STATE" ]] || ! grep -q '^v1_test_done$' "$STATE" 2>/dev/null; then
  if kill -0 "$V1_PID" 2>/dev/null; then
    wait_pid "$V1_PID" "v1 baseline_original_bce"
  else
    log "v1 pid $V1_PID not running, proceed to test/archive"
  fi

  CKPT=$(find_best_ckpt "$V1_EXP")
  if [[ -z "${CKPT:-}" ]]; then
    log "ERROR: v1 best checkpoint not found"
    exit 1
  fi
  log "v1 best checkpoint: $CKPT"
  bash "$ROOT/scripts/run_test_record.sh" "$V1_EXP" "$CKPT" "$V1_CONFIG"
  archive_v1
  mark v1_test_done
fi

if [[ ! -f "$STATE" ]] || ! grep -q '^v2_done$' "$STATE" 2>/dev/null; then
  mark v2_running
  export CONFIG_TAG=v2_begin500_val10_test50
  bash "$ROOT/scripts/run_baselines.sh" 2>&1 | tee -a "$LOG"
  mark v2_done
fi

log "pipeline complete"
