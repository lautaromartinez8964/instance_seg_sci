#!/usr/bin/env bash
set -e
set -o pipefail

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py}"
WORKDIR="${2:-work_dirs/mask_rcnn_r50_fpn_1x_isaid}"
EXTRA_CFG="${3:-train_dataloader.num_workers=0 train_dataloader.persistent_workers=False val_dataloader.num_workers=0 val_dataloader.persistent_workers=False test_dataloader.num_workers=0 test_dataloader.persistent_workers=False train_cfg.val_interval=9999}"
TRAIN_ARGS="${4:-}"

cd ~/projects/mmdetection || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || true

mkdir -p "$WORKDIR/auto_logs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

find_latest_ckpt() {
  # 优先 latest.pth
  if [ -f "$WORKDIR/latest.pth" ]; then
    echo "$WORKDIR/latest.pth"
    return 0
  fi

  # 否则找 epoch_*.pth 中数字最大的
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

while true; do
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$WORKDIR/auto_logs/run_$TS.log"
  CKPT="$(find_latest_ckpt)"

  echo "[$(date)] CONFIG=$CONFIG" | tee -a "$LOG"
  echo "[$(date)] WORKDIR=$WORKDIR" | tee -a "$LOG"
  echo "[$(date)] CKPT=${CKPT:-<none>}" | tee -a "$LOG"
  echo "[$(date)] TRAIN_ARGS=${TRAIN_ARGS:-<none>}" | tee -a "$LOG"

  if [ -n "$CKPT" ]; then
    set +e
    python tools/train.py "$CONFIG" \
      --work-dir "$WORKDIR" \
      --resume "$CKPT" \
      $TRAIN_ARGS \
      --cfg-options $EXTRA_CFG \
      2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    set -e
  else
    set +e
    python tools/train.py "$CONFIG" \
      --work-dir "$WORKDIR" \
      $TRAIN_ARGS \
      --cfg-options $EXTRA_CFG \
      2>&1 | tee -a "$LOG"
    code=${PIPESTATUS[0]}
    set -e
  fi
  echo "[$(date)] exit code: $code" | tee -a "$LOG"

  if [ "$code" -eq 0 ]; then
    echo "[$(date)] training finished normally." | tee -a "$LOG"
    break
  fi

  echo "[$(date)] crashed, sleep 20s then restart..." | tee -a "$LOG"
  sleep 20
done