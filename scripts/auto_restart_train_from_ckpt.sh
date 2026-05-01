#!/usr/bin/env bash
set -u
set -o pipefail

CONFIG="${1:?config is required}"
WORKDIR="${2:?workdir is required}"
INITIAL_CKPT="${3:-}"
SESSION_NAME="${4:-}"
shift $(( $# >= 4 ? 4 : $# ))
EXTRA_TRAIN_ARGS=("$@")

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
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

# If no new log output for this long, treat as hang and restart.
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-120}"
STALL_CHECK_INTERVAL_SECONDS="${STALL_CHECK_INTERVAL_SECONDS:-10}"

detect_target_epoch() {
  local config_path="$1"
  local epoch
  local base_rel
  local base_path

  epoch=$(grep -Eo 'max_epochs[[:space:]]*=[[:space:]]*[0-9]+' "$config_path" 2>/dev/null | tail -n 1 | grep -Eo '[0-9]+' | tail -n 1)
  if [ -n "${epoch:-}" ]; then
    echo "$epoch"
    return 0
  fi

  epoch=$(grep -Eo 'max_epochs[[:space:]]*:[[:space:]]*[0-9]+|max_epochs=[0-9]+' "$config_path" 2>/dev/null | tail -n 1 | grep -Eo '[0-9]+' | tail -n 1)
  if [ -n "${epoch:-}" ]; then
    echo "$epoch"
    return 0
  fi

  base_rel=$(sed -n "s/^_base_[[:space:]]*=[[:space:]]*\['\([^']*\)'\].*/\1/p" "$config_path" | head -n 1)
  if [ -n "${base_rel:-}" ]; then
    base_path=$(python - <<'PY' "$config_path" "$base_rel"
import os
import sys

config_path = sys.argv[1]
base_rel = sys.argv[2]
print(os.path.normpath(os.path.join(os.path.dirname(config_path), base_rel)))
PY
)
    if [ -n "${base_path:-}" ] && [ -f "$base_path" ]; then
      epoch=$(detect_target_epoch "$base_path")
      if [ -n "${epoch:-}" ]; then
        echo "$epoch"
        return 0
      fi
    fi
  fi

  echo ""
}

TARGET_EPOCH="${TARGET_EPOCH:-$(detect_target_epoch "$CONFIG")}" 

list_candidate_logs() {
  find "$WORKDIR" \
    \( -path "$WORKDIR/auto_logs/run_*.log" -o -path "$WORKDIR/*/*.log" -o -path "$WORKDIR/*.log" \) \
    -type f -print0 2>/dev/null
}

detect_iters_per_epoch() {
  local log_file
  local iters_per_epoch

  while IFS= read -r -d '' log_file; do
    iters_per_epoch=$(grep -Eo 'Epoch\(train\)[[:space:]]*\[[0-9]+\]\[[0-9]+/[0-9]+\]' "$log_file" 2>/dev/null | \
      sed -E 's/.*\[[0-9]+\/([0-9]+)\]/\1/' | tail -n 1)
    if [ -n "${iters_per_epoch:-}" ]; then
      echo "$iters_per_epoch"
      return 0
    fi
  done < <(list_candidate_logs)

  echo ""
}

pick_latest_iter_ckpt() {
  local latest_iter

  latest_iter=$(ls "$WORKDIR"/iter_*.pth 2>/dev/null | \
    sed -E 's/.*iter_([0-9]+)\.pth/\1 &/' | \
    sort -n | tail -1 | awk '{print $2}')
  if [ -n "${latest_iter:-}" ] && [ -f "$latest_iter" ]; then
    echo "$latest_iter"
    return 0
  fi

  echo ""
}

is_training_complete() {
  local target_epoch="$1"
  local log_file
  local iters_per_epoch
  local target_iters
  local latest_iter_ckpt
  local latest_iter

  if [ -z "${target_epoch:-}" ]; then
    return 1
  fi

  if [ -f "$WORKDIR/epoch_${target_epoch}.pth" ]; then
    return 0
  fi

  while IFS= read -r -d '' log_file; do
      if grep -qE "Epoch\\(val\\) \[$target_epoch\]\\[[0-9]+/[0-9]+\\].*coco/segm_mAP" "$log_file" 2>/dev/null; then
        return 0
      fi
  done < <(list_candidate_logs)

  iters_per_epoch=$(detect_iters_per_epoch)
  latest_iter_ckpt=$(pick_latest_iter_ckpt)
  if [ -n "${iters_per_epoch:-}" ] && [ -n "${latest_iter_ckpt:-}" ]; then
    latest_iter=$(basename "$latest_iter_ckpt" | sed -E 's/^iter_([0-9]+)\.pth$/\1/')
    target_iters=$((target_epoch * iters_per_epoch))
    if [ -n "${latest_iter:-}" ] && [ "$latest_iter" -ge "$target_iters" ]; then
      return 0
    fi
  fi

  return 1
}

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

run_with_watchdog() {
  local run_log="$1"
  shift
  local train_cmd=("$@")
  local cmd_str
  local child_pid
  local child_pgid
  local now_ts
  local last_progress_ts
  local log_mtime

  printf -v cmd_str '%q ' "${train_cmd[@]}"

  # Run train command as a separate process group so we can kill the whole
  # pipeline (python + tee) when no log progress is observed. Keep pipefail
  # enabled inside the child shell so python failures are not masked by tee.
  set +e
  setsid bash -lc "set -o pipefail; $cmd_str 2>&1 | tee -a '$RUNNER_LOG' '$run_log'" &
  child_pid=$!
  child_pgid=$child_pid
  set -e

  if [ -f "$run_log" ]; then
    last_progress_ts=$(stat -c %Y "$run_log" 2>/dev/null || date +%s)
  else
    last_progress_ts=$(date +%s)
  fi

  while kill -0 "$child_pid" 2>/dev/null; do
    now_ts=$(date +%s)
    log_mtime=$(stat -c %Y "$run_log" 2>/dev/null || echo "$last_progress_ts")
    if [ "$log_mtime" -gt "$last_progress_ts" ]; then
      last_progress_ts="$log_mtime"
    fi

    if [ $((now_ts - last_progress_ts)) -ge "$STALL_TIMEOUT_SECONDS" ]; then
      echo "[$(date '+%F %T')] stall detected: no log update for ${STALL_TIMEOUT_SECONDS}s; terminating process group $child_pgid" \
        | tee -a "$RUNNER_LOG" "$run_log"
      kill -TERM -- "-$child_pgid" 2>/dev/null || true

      for _ in 1 2 3 4 5 6; do
        if ! kill -0 "$child_pid" 2>/dev/null; then
          break
        fi
        sleep 5
      done

      if kill -0 "$child_pid" 2>/dev/null; then
        echo "[$(date '+%F %T')] process group $child_pgid still alive; force killing" \
          | tee -a "$RUNNER_LOG" "$run_log"
        kill -KILL -- "-$child_pgid" 2>/dev/null || true
      fi

      wait "$child_pid" 2>/dev/null || true
      set +e
      return 124
    fi

    sleep "$STALL_CHECK_INTERVAL_SECONDS"
  done

  set +e
  wait "$child_pid"
  local child_exit_code=$?
  return "$child_exit_code"
}

attempt=0
while true; do
  if is_training_complete "$TARGET_EPOCH"; then
    echo "[$(date '+%F %T')] detected completed training before launch (target_epoch=${TARGET_EPOCH:-unknown}); exiting" | tee -a "$RUNNER_LOG"
    break
  fi

  attempt=$((attempt + 1))
  ts=$(date +%Y%m%d_%H%M%S)
  run_log="$WORKDIR/auto_logs/run_$ts.log"
  resume_ckpt="$(pick_resume_ckpt)"

  extra_args_str="<none>"
  if [ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]; then
    printf -v extra_args_str '%q ' "${EXTRA_TRAIN_ARGS[@]}"
  fi

  echo "[$(date '+%F %T')] attempt=$attempt session=${SESSION_NAME:-<none>} resume=${resume_ckpt:-<none>} stall_timeout=${STALL_TIMEOUT_SECONDS}s stall_check_interval=${STALL_CHECK_INTERVAL_SECONDS}s extra_args=${extra_args_str}" | tee -a "$RUNNER_LOG" "$run_log"

  train_cmd=(python -X faulthandler tools/train.py "$CONFIG" --work-dir "$WORKDIR")
  if [ -n "$resume_ckpt" ]; then
    train_cmd+=(--resume "$resume_ckpt")
  fi
  if [ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]; then
    train_cmd+=("${EXTRA_TRAIN_ARGS[@]}")
  fi

  set +e
  run_with_watchdog "$run_log" "${train_cmd[@]}"
  exit_code=$?
  set -e

  echo "[$(date '+%F %T')] exit_code=$exit_code" | tee -a "$RUNNER_LOG" "$run_log"
  if [ "$exit_code" -eq 0 ]; then
    echo "[$(date '+%F %T')] training finished normally" | tee -a "$RUNNER_LOG" "$run_log"
    break
  fi

  if is_training_complete "$TARGET_EPOCH"; then
    echo "[$(date '+%F %T')] detected completed training after non-zero exit (target_epoch=${TARGET_EPOCH:-unknown}); exiting without restart" | tee -a "$RUNNER_LOG" "$run_log"
    break
  fi

  echo "[$(date '+%F %T')] training crashed; retry in 20s" | tee -a "$RUNNER_LOG" "$run_log"
  sleep 20
done