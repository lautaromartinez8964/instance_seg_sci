#!/usr/bin/env bash
set -e
set -o pipefail

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_rs_lightmamba_ig_scan_v2_fpn_1x_isaid_2241.py}"
WORKDIR="${2:-work_dirs/mask_rcnn_rs_lightmamba_ig_scan_v2_fpn_1x_isaid_2241}"
IMAGE_DIR="${3:-data/iSAID/val/images}"
NUM_IMAGES="${4:-6}"
OUT_ROOT="${5:-$WORKDIR/ig_scan_v2_vis}"
CKPT_OVERRIDE="${6:-}"

cd ~/projects/mmdetection || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || true
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

find_latest_ckpt() {
  if [ -f "$WORKDIR/latest.pth" ]; then
    echo "$WORKDIR/latest.pth"
    return 0
  fi

  local f
  f=$(ls "$WORKDIR"/epoch_*.pth 2>/dev/null | \
      sed -E 's/.*epoch_([0-9]+)\.pth/\1 &/' | \
      sort -n | tail -1 | awk '{print $2}')
  if [ -n "${f:-}" ] && [ -f "$f" ]; then
    echo "$f"
    return 0
  fi

  echo ""
  return 0
}

if [ -n "$CKPT_OVERRIDE" ]; then
  CKPT="$CKPT_OVERRIDE"
else
  CKPT="$(find_latest_ckpt)"
fi
if [ -z "$CKPT" ]; then
  echo "No checkpoint found in $WORKDIR" >&2
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 1
fi

CKPT_BASENAME="$(basename "$CKPT" .pth)"
OUT_DIR="$OUT_ROOT/$CKPT_BASENAME"
mkdir -p "$OUT_DIR"

mapfile -t IMAGES < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) \
  | grep -Ev '(_instance_color_RGB|_instance_id_RGB)' \
  | sort \
  | head -n "$NUM_IMAGES")

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "No usable images found in $IMAGE_DIR" >&2
  exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Saving visualizations to: $OUT_DIR"

for IMG in "${IMAGES[@]}"; do
  STEM="$(basename "$IMG")"
  STEM="${STEM%.*}"
  TARGET_DIR="$OUT_DIR/$STEM"
  mkdir -p "$TARGET_DIR"
  echo "Visualizing $IMG -> $TARGET_DIR"
  python tools/analysis_tools/visualize_ig_scan_v2.py \
    "$CONFIG" \
    "$CKPT" \
    "$IMG" \
    --out-dir "$TARGET_DIR"
done

echo "Visualization complete."