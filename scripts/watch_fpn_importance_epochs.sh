#!/usr/bin/env bash
set -u
set -o pipefail

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_1x_isaid.py}"
WORKDIR="${2:-work_dirs/mask_rcnn_rs_lightmamba_s4_global_attn_ig_fpn_1x_isaid}"
OUTROOT="${3:-$WORKDIR/ig_fpn_vis}"

cd ~/projects/mmdetection || exit 1
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || exit 1
set -u

mkdir -p "$OUTROOT"

DEVICE="${DEVICE:-cuda:0}"
GRID_SIZE="${GRID_SIZE:-4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
FORCE="${FORCE:-0}"
EPOCHS=(1 4 8 12)
SAMPLES=(
  P0003_0_0_800_800
  P0003_0_223_800_800
  P0003_347_0_800_800
  P0003_347_223_800_800
  P0004_0_0_800_800
  P0004_0_600_800_800
  P0004_0_664_800_800
  P0004_24_0_800_800
  P0004_24_600_800_800
  P0004_24_664_800_800
  P0007_0_0_800_800
  P0007_0_1200_800_800
  P0007_0_1337_800_800
  P0007_0_600_800_800
  P0007_600_0_800_800
  P0007_600_1200_800_800
  P0007_600_1337_800_800
  P0007_600_600_800_800
  P0007_690_0_800_800
  P0007_690_1200_800_800
)

run_epoch_vis() {
  local epoch="$1"
  local ckpt="$WORKDIR/epoch_${epoch}.pth"
  local epoch_dir="$OUTROOT/epoch_${epoch}"
  local done_flag="$OUTROOT/.epoch_${epoch}.done"
  local cmd=(python tools/analysis_tools/compare_fpn_importance.py "$CONFIG" "$ckpt" --out-dir "$epoch_dir" --device "$DEVICE" --grid-size "$GRID_SIZE")
  local sample

  for sample in "${SAMPLES[@]}"; do
    cmd+=(--image "$sample")
  done

  if [[ "$FORCE" != "1" && -f "$done_flag" ]]; then
    echo "[$(date '+%F %T')] epoch ${epoch} already rendered, skip"
    return 0
  fi

  echo "[$(date '+%F %T')] render epoch ${epoch} from $ckpt"
  "${cmd[@]}"
  touch "$done_flag"
}

wait_for_epoch() {
  local epoch="$1"
  local ckpt="$WORKDIR/epoch_${epoch}.pth"
  while [[ ! -f "$ckpt" ]]; do
    echo "[$(date '+%F %T')] waiting for $ckpt"
    sleep "$POLL_SECONDS"
  done
}

for epoch in "${EPOCHS[@]}"; do
  wait_for_epoch "$epoch"
  run_epoch_vis "$epoch"
done

echo "[$(date '+%F %T')] all requested epochs rendered under $OUTROOT"