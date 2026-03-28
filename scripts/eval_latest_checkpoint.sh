#!/usr/bin/env bash
set -e
set -o pipefail

CONFIG="$1"
WORKDIR="$2"
OUT_PKL="${3:-}"

if [ -z "$CONFIG" ] || [ -z "$WORKDIR" ]; then
  echo "Usage: bash scripts/eval_latest_checkpoint.sh <config> <workdir> [out_pkl]"
  exit 1
fi

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
}

CKPT="$(find_latest_ckpt)"
if [ -z "$CKPT" ]; then
  echo "No checkpoint found under $WORKDIR"
  exit 2
fi

if [ -z "$OUT_PKL" ]; then
  base_name="$(basename "$CKPT" .pth)"
  OUT_PKL="$WORKDIR/results_${base_name}.pkl"
fi

echo "Using checkpoint: $CKPT"
bash scripts/eval_and_extract_metrics.sh "$CONFIG" "$CKPT" "$WORKDIR" "$OUT_PKL"
