#!/usr/bin/env bash
# 命令行查看训练进度
ROOT="${1:-./log/baselines/DenseSIRST}"
echo "=== 运行中的 train.py 进程 ==="
ps aux | grep '[p]ython train.py' || true
echo
echo "=== 最新 console 日志 (tail) ==="
latest=$(ls -t "$ROOT"/*.console.log 2>/dev/null | head -1)
if [[ -n "${latest:-}" ]]; then
  echo "file: $latest"
  tail -n 20 "$latest"
else
  find ./log -name '*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2- | while read -r f; do
    echo "--- $f ---"; tail -n 8 "$f"; echo
  done
fi
echo
echo "=== epoch_metrics.csv (最新实验) ==="
latest_csv=$(find "$ROOT" -name epoch_metrics.csv -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [[ -n "${latest_csv:-}" ]]; then
  echo "file: $latest_csv"
  column -t -s, "$latest_csv" 2>/dev/null | tail -n 15 || tail -n 15 "$latest_csv"
else
  echo "(暂无)"
fi
echo
echo "=== GPU 状态 ==="
gpustat 2>/dev/null || nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
