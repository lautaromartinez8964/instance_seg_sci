#!/usr/bin/env bash
set -u
set -o pipefail

CONFIG="${1:?config is required}"
WORKDIR="${2:?workdir is required}"
INITIAL_CKPT="${3:-}"
SESSION_NAME="${4:-}"

cd ~/projects/mmdetection || exit 1
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || exit 1
set -u

mkdir -p "$WORKDIR/auto_logs"
RUNNER_LOG="$WORKDIR/auto_restart_runner.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

pick_resume_ckpt() {
  if [ -f "$WORKDIR/last_checkpoint" ]; then
    local last_ckpt
    last_ckpt=$(cat "$WORKDIR/last_checkpoint")
    if [ -n "${last_ckpt:-}" ] && [ -f "$last_ckpt" ]; then
      echo "$last_ckpt"
      return 0
    fi
  fi

  if [ -n "${INITIAL_CKPT:-}" ] && [ -f "$INITIAL_CKPT" ]; then
    echo "$INITIAL_CKPT"
    return 0
  fi

  local latest_epoch
  latest_epoch=$(ls "$WORKDIR"/epoch_*.pth 2>/dev/null | \
    sed -E 's/.*epoch_([0-9]+)\.pth/\1 &/' | \
    sort -n | tail -1 | awk '{print $2}')
  if [ -n "${latest_epoch:-}" ] && [ -f "$latest_epoch" ]; then
    echo "$latest_epoch"
    return 0
  fi

  echo ""
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  ts=$(date +%Y%m%d_%H%M%S)
  run_log="$WORKDIR/auto_logs/run_$ts.log"
  resume_ckpt="$(pick_resume_ckpt)"

  echo "[$(date '+%F %T')] attempt=$attempt session=${SESSION_NAME:-<none>} resume=${resume_ckpt:-<none>}" | tee -a "$RUNNER_LOG" "$run_log"

  if [ -n "$resume_ckpt" ]; then
    set +e
    python tools/train.py "$CONFIG" --work-dir "$WORKDIR" --resume "$resume_ckpt" 2>&1 | tee -a "$RUNNER_LOG" "$run_log"
    exit_code=${PIPESTATUS[0]}
    set -e
  else
    set +e
    python tools/train.py "$CONFIG" --work-dir "$WORKDIR" 2>&1 | tee -a "$RUNNER_LOG" "$run_log"
    exit_code=${PIPESTATUS[0]}
    set -e
  fi

  echo "[$(date '+%F %T')] exit_code=$exit_code" | tee -a "$RUNNER_LOG" "$run_log"
  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date '+%F %T')] training finished normally" | tee -a "$RUNNER_LOG" "$run_log"
    break
  fi

  echo "[$(date '+%F %T')] training crashed; retry in 20s" | tee -a "$RUNNER_LOG" "$run_log"
  sleep 20
done