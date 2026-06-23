#!/usr/bin/env bash
# GPU4 双槽调度：保持 2 个 train.py 同时运行，依次完成 5 个 v2 baseline（训练+test）
# 用法: INTERVAL=3600 bash scripts/gpu4_baseline_scheduler.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU=${GPU:-4}
MAX_SLOTS=${MAX_SLOTS:-2}
INTERVAL=${INTERVAL:-3600}
V1_PID=${V1_PID:-3941551}
CONFIG_V2=v2_begin500_val10_test50
CONFIG_V1=v1_begin500_val1_test0

SCHED_DIR="$ROOT/log/baselines/scheduler"
STATE="$SCHED_DIR/state.tsv"
JOBS_DIR="$SCHED_DIR/jobs"
LOG="$SCHED_DIR/scheduler.log"
LOCK="$SCHED_DIR/scheduler.lock"

mkdir -p "$SCHED_DIR" "$JOBS_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

gpu4_train_pids() {
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk -F',' '{gsub(/ /,"",$1); if($1!="") print $1}' \
    | while read -r pid; do
        if ps -p "$pid" -o args= 2>/dev/null | grep -q '[p]ython.*train\.py'; then
          echo "$pid"
        fi
      done
}

gpu4_train_count() {
  gpu4_train_pids | wc -l
}

init_state() {
  if [[ -f "$STATE" ]]; then
    return
  fi
  cat > "$STATE" <<'EOF'
# experiment|status|wrapper_pid|config_tag|extra_args
baseline_original_bce|pending||v2_begin500_val10_test50|--loss_mode bce --lr 0.001
baseline_lr1e4|pending||v2_begin500_val10_test50|--loss_mode bce --lr 0.0001
baseline_bce_dice|pending||v2_begin500_val10_test50|--loss_mode bce_dice --lr 0.0001 --bce_weight 1.0 --dice_weight 1.0
baseline_tversky|pending||v2_begin500_val10_test50|--loss_mode tversky --lr 0.0001 --tversky_alpha 0.7 --tversky_beta 0.3
baseline_focal_tversky|pending||v2_begin500_val10_test50|--loss_mode focal_tversky --lr 0.0001 --focal_gamma 0.75
EOF
  echo "V1_HANDLED=0" > "$SCHED_DIR/flags.env"
  log "initialized queue (5 v2 baselines)"
}

load_flag() {
  # shellcheck disable=SC1090
  source "$SCHED_DIR/flags.env"
}

save_flag() {
  echo "V1_HANDLED=${V1_HANDLED:-0}" > "$SCHED_DIR/flags.env"
}

find_best_ckpt() {
  local exp=$1
  ls -t "$ROOT/log/baselines/DenseSIRST/$exp"/SCTransNet_*_best.pth.tar 2>/dev/null | head -1
}

handle_v1_if_needed() {
  load_flag
  if [[ "${V1_HANDLED:-0}" == "1" ]]; then
    return
  fi
  if kill -0 "$V1_PID" 2>/dev/null; then
    log "v1 baseline_original_bce still running pid=$V1_PID (old config)"
    return
  fi
  log "v1 finished pid=$V1_PID -> test + archive"
  CKPT=$(find_best_ckpt baseline_original_bce)
  if [[ -n "${CKPT:-}" ]]; then
    bash "$ROOT/scripts/run_test_record.sh" baseline_original_bce "$CKPT" "$CONFIG_V1" \
      >> "$LOG" 2>&1 || log "WARN: v1 test failed"
  else
    log "WARN: v1 best checkpoint not found"
  fi
  TS=$(date +%Y%m%d_%H%M%S)
  SRC="$ROOT/log/baselines/DenseSIRST/baseline_original_bce"
  if [[ -d "$SRC" ]]; then
    mv "$SRC" "${SRC}_v1_val1_${TS}"
    log "archived v1 -> ${SRC}_v1_val1_${TS}"
  fi
  V1_HANDLED=1
  save_flag
}

read_queue_line() {
  local line=$1
  IFS='|' read -r exp status wp cfg extra <<< "$line"
}

write_queue_line() {
  printf '%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5"
}

update_running_jobs() {
  local tmp line exp status wp cfg extra latest_log
  tmp=$(mktemp)
  while IFS= read -r line; do
    [[ "$line" == "#"* || -z "$line" ]] && continue
    read_queue_line "$line"
    if [[ "$status" == "running" && -n "$wp" ]]; then
      if kill -0 "$wp" 2>/dev/null; then
        write_queue_line "$exp" "$status" "$wp" "$cfg" "$extra" >> "$tmp"
      else
        latest_log=$(ls -t "$JOBS_DIR/${exp}_"*.log 2>/dev/null | head -1 || true)
        if [[ -n "$latest_log" ]] && grep -q "DONE $exp" "$latest_log"; then
          log "job done $exp wrapper=$wp"
          write_queue_line "$exp" "done" "" "$cfg" "$extra" >> "$tmp"
        else
          log "ERROR $exp failed wrapper=$wp log=${latest_log:-none}"
          write_queue_line "$exp" "pending" "" "$cfg" "$extra" >> "$tmp"
        fi
      fi
    else
      write_queue_line "$exp" "$status" "$wp" "$cfg" "$extra" >> "$tmp"
    fi
  done < "$STATE"
  mv "$tmp" "$STATE"
}

all_done() {
  local pending running
  pending=$(awk -F'|' '$2=="pending"{c++} END{print c+0}' "$STATE")
  running=$(awk -F'|' '$2=="running"{c++} END{print c+0}' "$STATE")
  load_flag
  if [[ "${V1_HANDLED:-0}" != "1" ]] && kill -0 "$V1_PID" 2>/dev/null; then
    return 1
  fi
  [[ "$pending" -eq 0 && "$running" -eq 0 ]]
}

start_next_jobs() {
  local count free tmp line exp status wp cfg extra started=0 JOBLOG
  count=$(gpu4_train_count)
  free=$((MAX_SLOTS - count))
  log "GPU${GPU} train processes=$count free_slots=$free"
  if [[ "$free" -le 0 ]]; then
    return
  fi

  load_flag
  if [[ "${V1_HANDLED:-0}" != "1" ]] && kill -0 "$V1_PID" 2>/dev/null; then
    log "waiting for v1 to finish before starting v2 queue"
    return
  fi

  tmp=$(mktemp)
  while IFS= read -r line; do
    [[ "$line" == "#"* || -z "$line" ]] && continue
    read_queue_line "$line"
    if [[ "$status" == "pending" && "$started" -lt "$free" ]]; then
      JOBLOG="$JOBS_DIR/${exp}_$(date +%Y%m%d_%H%M%S).log"
      log "starting $exp args=[$extra]"
      (
        export CONFIG_TAG="$cfg" GPU="$GPU"
        # shellcheck disable=SC2086
        bash "$ROOT/scripts/run_single_baseline.sh" "$exp" $extra
      ) >> "$JOBLOG" 2>&1 &
      wp=$!
      write_queue_line "$exp" "running" "$wp" "$cfg" "$extra" >> "$tmp"
      started=$((started + 1))
      log "started $exp wrapper_pid=$wp log=$JOBLOG"
    else
      write_queue_line "$exp" "$status" "$wp" "$cfg" "$extra" >> "$tmp"
    fi
  done < "$STATE"
  mv "$tmp" "$STATE"
}

print_status() {
  local line exp status wp cfg extra
  log "--- status ---"
  gpu4_train_pids | while read -r pid; do
    log "  gpu${GPU} train pid=$pid $(ps -p "$pid" -o args= 2>/dev/null | cut -c1-120)"
  done
  while IFS= read -r line; do
    [[ "$line" == "#"* || -z "$line" ]] && continue
    read_queue_line "$line"
    log "  queue $exp status=$status wrapper=${wp:-none}"
  done < "$STATE"
  if [[ -f "$ROOT/log/baselines/test_records/test_results.csv" ]]; then
    log "  test_records: $(wc -l < "$ROOT/log/baselines/test_records/test_results.csv") lines"
  fi
}

tick() {
  exec 9>"$LOCK"
  if ! flock -n 9; then
    log "another scheduler tick running, skip"
    return
  fi
  init_state
  handle_v1_if_needed
  update_running_jobs
  start_next_jobs
  print_status
  if all_done; then
    log "ALL 5 v2 baselines done (+ v1 handled)"
    return 0
  fi
  return 1
}

log "scheduler started GPU=$GPU MAX_SLOTS=$MAX_SLOTS INTERVAL=${INTERVAL}s V1_PID=$V1_PID"
tick || true

while true; do
  sleep "$INTERVAL"
  if tick; then
    log "scheduler exit (queue complete)"
    exit 0
  fi
done
