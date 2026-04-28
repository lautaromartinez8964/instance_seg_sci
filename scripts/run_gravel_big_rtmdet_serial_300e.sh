#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"
SUMMARY_FILE="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_serial_300e_results.md"
MASTER_LOG="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_serial_300e_runner.log"

mkdir -p "$ROOT_DIR/work_dirs_gravel_big"

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmdec
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CONFIGS=(
  "configs/gravel_big/rtmdet_ins_tiny_baseline_300e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_nwd_assign_300e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_300e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_hybrid_vmamba2292_300e_gravel_big.py"
  "configs/gravel_big/rtmdet_ins_tiny_vmamba2292_300e_gravel_big.py"
)

WORKDIRS=(
  "work_dirs_gravel_big/rtmdet_ins_tiny_baseline_300e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_nwd_assign_300e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_300e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_hybrid_vmamba2292_300e_gravel_big"
  "work_dirs_gravel_big/rtmdet_ins_tiny_vmamba2292_300e_gravel_big"
)

NAMES=(
  "RTMDet-Ins Tiny Baseline"
  "RTMDet-Ins Tiny NWD Assign"
  "RTMDet-Ins Tiny Shared DT v3"
  "RTMDet-Ins Tiny Hybrid VMamba2292"
  "RTMDet-Ins Tiny Full VMamba2292"
)

COMMON_TRAIN_ARGS=(
  --cfg-options
  train_dataloader.num_workers=2
  val_dataloader.num_workers=2
  test_dataloader.num_workers=2
)

append_summary_header() {
  if [[ -s "$SUMMARY_FILE" ]]; then
    return 0
  fi

  cat > "$SUMMARY_FILE" <<'EOF'
# Gravel Big RTMDet-Ins Serial 300e Results

该文件由串行训练脚本自动维护。

  训练顺序：Baseline -> NWD Assign -> Shared DT v3 -> Hybrid VMamba2292 -> Full VMamba2292

EOF
}

is_training_complete() {
  local workdir="$1"
  [[ -f "$ROOT_DIR/$workdir/epoch_300.pth" ]]
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

  if is_training_complete "$workdir"; then
    echo "[$(date '+%F %T')] skip training (completed): $name" | tee -a "$MASTER_LOG"
  else
    echo "[$(date '+%F %T')] start training: $name" | tee -a "$MASTER_LOG"
    if ! "$AUTO_SCRIPT" "$ROOT_DIR/$config" "$ROOT_DIR/$workdir" "" "$name" "${COMMON_TRAIN_ARGS[@]}"; then
      if is_training_complete "$workdir"; then
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