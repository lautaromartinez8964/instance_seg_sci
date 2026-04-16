#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"
SUMMARY_FILE="$ROOT_DIR/work_dirs_gravel/gravel_vmamba_serial_36e_augment_results.md"
MASTER_LOG="$ROOT_DIR/work_dirs_gravel/gravel_vmamba_serial_36e_augment_runner.log"

mkdir -p "$ROOT_DIR/work_dirs_gravel"

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mmdec
set -u

CONFIGS=(
  "configs/gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414_augment.py"
  "configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414_augment.py"
  "configs/gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414_augment.py"
)

WORKDIRS=(
  "work_dirs_gravel/mask_rcnn_vmamba_official_2292_fpn_36e_gravel_414_augment"
  "work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_fpn_36e_gravel_414_augment"
  "work_dirs_gravel/mask_rcnn_rs_lightmamba_s4_global_attn_hf_fpn_36e_gravel_414_augment"
)

NAMES=(
  "Official VMamba 2292"
  "RS-LightMamba S4 GlobalAttn"
  "RS-LightMamba S4 GlobalAttn HF-FPN"
)

append_summary_header() {
  cat > "$SUMMARY_FILE" <<'EOF'
# Gravel VMamba Serial 36e Augment Results

该文件由串行训练脚本自动维护。

EOF
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
      grep -E 'Average Precision|Average Recall' "$eval_log" || true
      echo '```'
    fi
    echo
  } >> "$SUMMARY_FILE"
}

run_test_eval() {
  local config="$1"
  local workdir="$2"
  local best_ckpt="$3"
  local eval_log="$workdir/test_eval.log"

  if [[ -z "$best_ckpt" || ! -f "$best_ckpt" ]]; then
    echo "[$(date '+%F %T')] skip test eval for $workdir: best checkpoint not found" | tee -a "$MASTER_LOG"
    echo ""
    return 0
  fi

  echo "[$(date '+%F %T')] test eval: $config with $best_ckpt" | tee -a "$MASTER_LOG"
  python "$ROOT_DIR/tools/test.py" "$ROOT_DIR/$config" "$best_ckpt" --work-dir "$ROOT_DIR/$workdir/test_eval" \
    2>&1 | tee "$eval_log"
  echo "$eval_log"
}

append_summary_header

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  workdir="${WORKDIRS[$idx]}"
  name="${NAMES[$idx]}"

  echo "[$(date '+%F %T')] start training: $name" | tee -a "$MASTER_LOG"
  STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-240}" \
    "$AUTO_SCRIPT" "$ROOT_DIR/$config" "$ROOT_DIR/$workdir"

  best_ckpt="$(find "$ROOT_DIR/$workdir" -maxdepth 1 -type f -name 'best_coco_segm_mAP*.pth' | sort | tail -n 1 || true)"
  eval_log="$(run_test_eval "$config" "$workdir" "$best_ckpt")"
  append_model_summary "$name" "$config" "$workdir" "$best_ckpt" "$eval_log"
  echo "[$(date '+%F %T')] finished: $name" | tee -a "$MASTER_LOG"
done

echo "[$(date '+%F %T')] all serial jobs finished" | tee -a "$MASTER_LOG"