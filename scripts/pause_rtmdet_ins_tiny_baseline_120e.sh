#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big.py"
WORKDIR="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big"
RESUME_FLAG="$WORKDIR/.resume_on_boot_enabled"
CONTROL_LOG="$WORKDIR/scheduled_control.log"
RUNNER_LOG="$WORKDIR/auto_restart_runner.log"

mkdir -p "$WORKDIR"
touch "$RESUME_FLAG"

echo "[$(date '+%F %T')] scheduled pause requested before power outage" | tee -a "$CONTROL_LOG" "$RUNNER_LOG"

mapfile -t target_pgids < <(
  ps -eo pgid=,cmd= | awk -v config="$CONFIG_PATH" -v workdir="$WORKDIR" '
    index($0, config) && index($0, workdir) && index($0, "tools/train.py") {print $1}
    index($0, config) && index($0, workdir) && index($0, "bash -lc set -o pipefail") {print $1}
  ' | sort -u
)

if [[ ${#target_pgids[@]} -eq 0 ]]; then
  echo "[$(date '+%F %T')] no matching training process found; resume flag kept for reboot" | tee -a "$CONTROL_LOG" "$RUNNER_LOG"
  exit 0
fi

for pgid in "${target_pgids[@]}"; do
  echo "[$(date '+%F %T')] sending SIGTERM to process group $pgid" | tee -a "$CONTROL_LOG" "$RUNNER_LOG"
  kill -TERM -- "-$pgid" 2>/dev/null || true
done

for _ in 1 2 3 4; do
  mapfile -t remaining_pgids < <(
    ps -eo pgid=,cmd= | awk -v config="$CONFIG_PATH" -v workdir="$WORKDIR" '
      index($0, config) && index($0, workdir) && index($0, "tools/train.py") {print $1}
      index($0, config) && index($0, workdir) && index($0, "bash -lc set -o pipefail") {print $1}
    ' | sort -u
  )
  if [[ ${#remaining_pgids[@]} -eq 0 ]]; then
    break
  fi
  sleep 5
done

if [[ ${#remaining_pgids[@]:-0} -gt 0 ]]; then
  for pgid in "${remaining_pgids[@]}"; do
    echo "[$(date '+%F %T')] process group $pgid still alive; sending SIGKILL" | tee -a "$CONTROL_LOG" "$RUNNER_LOG"
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
fi

echo "[$(date '+%F %T')] scheduled pause finished" | tee -a "$CONTROL_LOG" "$RUNNER_LOG"