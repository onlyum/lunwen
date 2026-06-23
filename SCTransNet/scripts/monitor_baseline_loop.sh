#!/usr/bin/env bash
# 每 30 分钟检查 v1 训练状态；结束后触发 orchestrate_baseline_pipeline.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STATE_DIR="$ROOT/log/baselines/orchestrator"
mkdir -p "$STATE_DIR"
MON_LOG="$STATE_DIR/monitor.log"
V1_PID=${V1_PID:-3941551}
INTERVAL=${INTERVAL:-1800}

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MON_LOG"
}

status_line() {
  local epoch info
  info=$(ls -t "$ROOT/log/baselines"/DenseSIRST_baseline_original_bce_*.txt 2>/dev/null | head -1)
  if [[ -n "${info:-}" ]]; then
    epoch=$(grep -E '^Jun .* Epoch---' "$info" | tail -1 || true)
    log "v1 log: $(basename "$info") | ${epoch:-no epoch yet}"
  fi
  if kill -0 "$V1_PID" 2>/dev/null; then
    log "v1 train RUNNING pid=$V1_PID"
  else
    log "v1 train STOPPED pid=$V1_PID"
  fi
  pgrep -af "train.py.*baseline" | grep python | tee -a "$MON_LOG" || log "no baseline train.py"
}

log "monitor started interval=${INTERVAL}s v1_pid=$V1_PID"
status_line

while true; do
  sleep "$INTERVAL"
  status_line
  if [[ -f "$STATE_DIR/pipeline_state.txt" ]] && grep -q '^v2_done$' "$STATE_DIR/pipeline_state.txt"; then
    log "pipeline complete, monitor exit"
    echo 'AGENT_LOOP_WAKE_BASELINE {"prompt":"baseline pipeline complete; summarize final status"}'
    exit 0
  fi
  if ! kill -0 "$V1_PID" 2>/dev/null; then
    if [[ ! -f "$STATE_DIR/pipeline_state.txt" ]] || ! grep -q '^v1_test_done$' "$STATE_DIR/pipeline_state.txt" 2>/dev/null; then
      log "v1 stopped, launching orchestrate_baseline_pipeline.sh"
      bash "$ROOT/scripts/orchestrate_baseline_pipeline.sh" >> "$STATE_DIR/pipeline.log" 2>&1 &
      wait $!
      log "orchestrator finished with exit=$?"
      echo 'AGENT_LOOP_WAKE_BASELINE {"prompt":"baseline v2 pipeline finished; summarize test records and training status"}'
      exit 0
    fi
  fi
done
