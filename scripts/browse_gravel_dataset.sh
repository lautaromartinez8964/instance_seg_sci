#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yxy18034962/projects/mmdetection}"
DATASET_SRC="${DATASET_SRC:-/home/yxy18034962/datasets/gravel_roboflow}"
DATA_LINK="${DATA_LINK:-$REPO_ROOT/data/gravel_roboflow}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/_base_/datasets/gravel_instance.py}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/work_dirs_gravel}"

cd "$REPO_ROOT"

if [[ ! -d "$DATASET_SRC" ]]; then
  echo "[error] dataset source not found: $DATASET_SRC" >&2
  echo "Upload merged_dataset to $DATASET_SRC first." >&2
  exit 2
fi

for required in \
  "$DATASET_SRC/train" \
  "$DATASET_SRC/val" \
  "$DATASET_SRC/test" \
  "$DATASET_SRC/annotations/instances_train.json" \
  "$DATASET_SRC/annotations/instances_val.json" \
  "$DATASET_SRC/annotations/instances_test.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing dataset component: $required" >&2
    exit 3
  fi
done

mkdir -p "$REPO_ROOT/data" "$OUT_ROOT"
ln -sfn "$DATASET_SRC" "$DATA_LINK"

run_browse() {
  local split="$1"
  local ann_file="$2"
  local img_prefix="$3"
  local out_dir="$OUT_ROOT/browse_${split}"

  mkdir -p "$out_dir"
  python tools/analysis_tools/browse_dataset.py "$CONFIG" \
    --output-dir "$out_dir" \
    --not-show \
    --cfg-options \
    "train_dataloader.dataset.ann_file=$ann_file" \
    "train_dataloader.dataset.data_prefix.img=$img_prefix" \
    "train_dataloader.dataset.test_mode=True"
}

run_browse train annotations/instances_train.json train/
run_browse val annotations/instances_val.json val/
run_browse test annotations/instances_test.json test/

echo "[done] browse outputs written to $OUT_ROOT"#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/yxy18034962/projects/mmdetection}"
DATASET_SRC="${DATASET_SRC:-/home/yxy18034962/datasets/gravel_roboflow}"
DATA_LINK="${DATA_LINK:-$REPO_ROOT/data/gravel_roboflow}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/_base_/datasets/gravel_instance.py}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/work_dirs_gravel}"

cd "$REPO_ROOT"

if [[ ! -d "$DATASET_SRC" ]]; then
  echo "[error] dataset source not found: $DATASET_SRC" >&2
  echo "Upload merged_dataset to $DATASET_SRC first." >&2
  exit 2
fi

for required in \
  "$DATASET_SRC/train" \
  "$DATASET_SRC/val" \
  "$DATASET_SRC/test" \
  "$DATASET_SRC/annotations/instances_train.json" \
  "$DATASET_SRC/annotations/instances_val.json" \
  "$DATASET_SRC/annotations/instances_test.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] missing dataset component: $required" >&2
    exit 3
  fi
done

mkdir -p "$REPO_ROOT/data" "$OUT_ROOT"
ln -sfn "$DATASET_SRC" "$DATA_LINK"

run_browse() {
  local split="$1"
  local ann_file="$2"
  local img_prefix="$3"
  local out_dir="$OUT_ROOT/browse_${split}"

  mkdir -p "$out_dir"
  python tools/analysis_tools/browse_dataset.py "$CONFIG" \
    --output-dir "$out_dir" \
    --not-show \
    --cfg-options \
    "train_dataloader.dataset.ann_file=$ann_file" \
    "train_dataloader.dataset.data_prefix.img=$img_prefix" \
    "train_dataloader.dataset.test_mode=True"
}

run_browse train annotations/instances_train.json train/
run_browse val annotations/instances_val.json val/
run_browse test annotations/instances_test.json test/

echo "[done] browse outputs written to $OUT_ROOT"