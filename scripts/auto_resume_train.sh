#!/usr/bin/env bash
set -u

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_r50_fpn_1x_isaid.py}"
WORKDIR="${2:-work_dirs/mask_rcnn_r50_fpn_1x_isaid}"
EXTRA_CFG="${3:-train_dataloader.num_workers=2 val_dataloader.num_workers=2}"

cd ~/projects/mmdetection || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || exit 1

mkdir -p "$WORKDIR/auto_logs"

export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while true; do
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$WORKDIR/auto_logs/run_$TS.log"
  echo "[$(date)] start/resume: $CONFIG" | tee -a "$LOG"

  python tools/train.py "$CONFIG" --work-dir "$WORKDIR" --resume \
    --cfg-options $EXTRA_CFG \
    2>&1 | tee -a "$LOG"

  code=${PIPESTATUS[0]}
  echo "[$(date)] exit code: $code" | tee -a "$LOG"

  if [ "$code" -eq 0 ]; then
    echo "[$(date)] training finished." | tee -a "$LOG"
    break
  fi

  echo "[$(date)] crashed, restart in 20s..." | tee -a "$LOG"
  sleep 20
done
