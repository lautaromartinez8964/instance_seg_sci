#!/usr/bin/env bash
set -e
set -o pipefail

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_rs_lightmamba_ig_scan_v2_fpn_1x_isaid_2241.py}"
WORKDIR="${2:-work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v2_fpn_1x_isaid_2241}"
IMAGE_DIR="${3:-data/iSAID/val/images}"
NUM_IMAGES="${4:-6}"
OUT_ROOT="${5:-$WORKDIR/ig_scan_v2_vis}"
EPOCHS="${6:-1 3}"

cd ~/projects/mmdetection || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || true

for EPOCH in $EPOCHS; do
  CKPT="$WORKDIR/epoch_${EPOCH}.pth"
  echo "[$(date)] Waiting for $CKPT"
  while [ ! -f "$CKPT" ]; do
    sleep 30
  done

  echo "[$(date)] Found $CKPT, running visualization"
  bash scripts/visualize_ig_scan_v2_latest.sh \
    "$CONFIG" \
    "$WORKDIR" \
    "$IMAGE_DIR" \
    "$NUM_IMAGES" \
    "$OUT_ROOT" \
    "$CKPT"
done

echo "[$(date)] Epoch watcher finished."