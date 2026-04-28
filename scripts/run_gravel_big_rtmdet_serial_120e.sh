#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"
SUMMARY_FILE="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_serial_120e_results.md"
MASTER_LOG="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_serial_120e_runner.log"

mkdir -p "$ROOT_DIR/work_dirs_gravel_big"

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmdec
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CONFIGS=(
  "configs/gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_nwd_assign_120e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_vmamba2242_120e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big.py"
)

WORKDIRS=(
  "work_dirs_gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_nwd_assign_120e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_vmamba2242_120e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big"
)

NAMES=(
  "RTMDet-Ins Tiny Baseline 120e"
  "RTMDet-Ins Tiny NWD Assign 120e"
  "RTMDet-Ins Tiny Full VMamba2242 120e"
  "RTMDet-Ins Tiny Shared DT v3 120e"
)

COMMON_TRAIN_ARGS=()

if [[ "${RTMDET_SERIAL_USE_AMP:-1}" == "1" ]]; then
  COMMON_TRAIN_ARGS+=(--amp)
fi

COMMON_TRAIN_ARGS+=(
  --cfg-options
  "train_dataloader.batch_size=${RTMDET_SERIAL_BATCH_SIZE:-8}"
  "optim_wrapper.accumulative_counts=${RTMDET_SERIAL_ACCUMULATIVE_COUNTS:-2}"
  "train_dataloader.num_workers=${RTMDET_SERIAL_NUM_WORKERS:-0}"
  "train_dataloader.persistent_workers=False"
  "val_dataloader.num_workers=${RTMDET_SERIAL_NUM_WORKERS:-0}"
  "val_dataloader.persistent_workers=False"
  "test_dataloader.num_workers=${RTMDET_SERIAL_NUM_WORKERS:-0}"
  "test_dataloader.persistent_workers=False"
  "default_hooks.checkpoint.by_epoch=False"
  "default_hooks.checkpoint.interval=${RTMDET_SERIAL_CKPT_INTERVAL_ITERS:-200}"
  "default_hooks.checkpoint.max_keep_ckpts=${RTMDET_SERIAL_MAX_KEEP_CKPTS:-20}"
  "train_cfg.val_interval=${RTMDET_SERIAL_VAL_INTERVAL:-10}"
)

append_summary_header() {
  if [[ -s "$SUMMARY_FILE" ]]; then
    return 0
  fi

  cat > "$SUMMARY_FILE" <<'EOF'
# Gravel Big RTMDet-Ins Serial 120e Results

该文件由串行训练脚本自动维护。

  训练顺序：Baseline -> NWD Assign -> Full VMamba2242 -> Shared DT v3

EOF
}

detect_target_epoch() {
  local config_path="$1"
  local epoch
  local base_rel
  local base_path

  epoch=$(grep -Eo 'max_epochs[[:space:]]*=[[:space:]]*[0-9]+' "$ROOT_DIR/$config_path" 2>/dev/null | tail -n 1 | grep -Eo '[0-9]+' | tail -n 1)
  if [[ -n "${epoch:-}" ]]; then
    echo "$epoch"
    return 0
  fi

  epoch=$(grep -Eo 'max_epochs[[:space:]]*:[[:space:]]*[0-9]+|max_epochs=[0-9]+' "$ROOT_DIR/$config_path" 2>/dev/null | tail -n 1 | grep -Eo '[0-9]+' | tail -n 1)
  if [[ -n "${epoch:-}" ]]; then
    echo "$epoch"
    return 0
  fi

  base_rel=$(sed -n "s/^_base_[[:space:]]*=[[:space:]]*\['\([^']*\)'\].*/\1/p" "$ROOT_DIR/$config_path" | head -n 1)
  if [[ -n "${base_rel:-}" ]]; then
    base_path=$(python - <<'PY' "$ROOT_DIR/$config_path" "$base_rel"
import os
import sys

config_path = sys.argv[1]
base_rel = sys.argv[2]
print(os.path.normpath(os.path.join(os.path.dirname(config_path), base_rel)))
PY
)
    if [[ -n "${base_path:-}" && -f "$base_path" ]]; then
      epoch=$(detect_target_epoch "${base_path#"$ROOT_DIR/"}")
      if [[ -n "${epoch:-}" ]]; then
        echo "$epoch"
        return 0
      fi
    fi
  fi

  echo ""
}

is_training_complete() {
  local config="$1"
  local workdir="$2"
  local target_epoch
  local log_file

  target_epoch="$(detect_target_epoch "$config")"
  if [[ -z "${target_epoch:-}" ]]; then
    return 1
  fi

  if [[ -f "$ROOT_DIR/$workdir/epoch_${target_epoch}.pth" ]]; then
    return 0
  fi

  if [[ -d "$ROOT_DIR/$workdir/auto_logs" ]]; then
    while IFS= read -r -d '' log_file; do
      if grep -qE "Epoch\\(val\\) \[$target_epoch\]\\[[0-9]+/[0-9]+\\].*coco/segm_mAP|Epoch\\(train\\) \[$target_epoch\]\\[[0-9]+/[0-9]+\\]" "$log_file" 2>/dev/null; then
        return 0
      fi
    done < <(find "$ROOT_DIR/$workdir/auto_logs" -maxdepth 1 -type f -name 'run_*.log' -print0 2>/dev/null)
  fi

  return 1
}

append_model_summary() {
  local name="$1"
  local config="$2"
  local workdir="$3"
  local best_ckpt="$4"
  local eval_log="$5"

  {
    echo "## $name"
    echo
    echo "- config: $config"
    echo "- work_dir: $workdir"
    echo "- best_ckpt: ${best_ckpt:-not-found}"
    echo "- eval_log: ${eval_log:-not-run}"
    if [[ -n "$eval_log" && -f "$eval_log" ]]; then
      echo
      echo '```text'
      grep -E 'Average Precision|Average Recall|coco/' "$eval_log" || true
      echo '```'
    fi
    echo
  } >> "$SUMMARY_FILE"
}

run_test_eval() {
  local config="$1"
  local workdir="$2"
  local best_ckpt="$3"
  local eval_log="$ROOT_DIR/$workdir/test_eval.log"
  local eval_exit_code=0

  if [[ -z "$best_ckpt" || ! -f "$best_ckpt" ]]; then
    echo "[$(date '+%F %T')] skip test eval for $workdir: best checkpoint not found" | tee -a "$MASTER_LOG" >&2
    echo ""
    return 0
  fi

  echo "[$(date '+%F %T')] test eval: $config with $best_ckpt" | tee -a "$MASTER_LOG" >&2
  set +e
  python "$ROOT_DIR/tools/test.py" "$ROOT_DIR/$config" "$best_ckpt" \
    --work-dir "$ROOT_DIR/$workdir/test_eval" \
    --cfg-options val_dataloader.num_workers=2 test_dataloader.num_workers=2 \
    2>&1 | tee "$eval_log" >&2
  eval_exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "$eval_exit_code" -ne 0 ]]; then
    echo "[$(date '+%F %T')] test eval failed for $workdir: exit_code=$eval_exit_code" | tee -a "$MASTER_LOG" >&2
  fi

  echo "$eval_log"
}

append_summary_header

START_INDEX="${START_INDEX:-0}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"
export STALL_TIMEOUT_SECONDS

for idx in "${!CONFIGS[@]}"; do
  if [[ "$idx" -lt "$START_INDEX" ]]; then
    continue
  fi

  config="${CONFIGS[$idx]}"
  workdir="${WORKDIRS[$idx]}"
  name="${NAMES[$idx]}"

  if is_training_complete "$config" "$workdir"; then
    echo "[$(date '+%F %T')] skip training (completed): $name" | tee -a "$MASTER_LOG"
  else
    echo "[$(date '+%F %T')] start training: $name" | tee -a "$MASTER_LOG"
    if ! "$AUTO_SCRIPT" "$ROOT_DIR/$config" "$ROOT_DIR/$workdir" "" "$name" "${COMMON_TRAIN_ARGS[@]}"; then
      if is_training_complete "$config" "$workdir"; then
        echo "[$(date '+%F %T')] training wrapper exited non-zero but final checkpoint exists: $name" | tee -a "$MASTER_LOG"
      else
        echo "[$(date '+%F %T')] training failed before completion: $name" | tee -a "$MASTER_LOG"
        exit 1
      fi
    fi
  fi

  best_ckpt="$(find "$ROOT_DIR/$workdir" -maxdepth 1 -type f -name 'best_coco_segm_mAP*.pth' | sort | tail -n 1 || true)"
  eval_log="$(run_test_eval "$config" "$workdir" "$best_ckpt")"
  append_model_summary "$name" "$config" "$workdir" "$best_ckpt" "$eval_log"
  echo "[$(date '+%F %T')] finished: $name" | tee -a "$MASTER_LOG"
done

echo "[$(date '+%F %T')] all serial jobs finished" | tee -a "$MASTER_LOG"