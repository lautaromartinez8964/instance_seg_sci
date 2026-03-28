#!/usr/bin/env bash
set -e
set -o pipefail

CONFIG="${1:-projects/iSAID/configs/mask_rcnn_vmamba_official_fpn_1x_isaid.py}"
CKPT="${2:-work_dirs/mask_rcnn_vmamba_official_fpn_1x_isaid_2292/epoch_12.pth}"
WORKDIR="${3:-work_dirs/mask_rcnn_vmamba_official_fpn_1x_isaid_2292}"
OUT_PKL="${4:-$WORKDIR/results_epoch12.pkl}"

cd /home/yxy18034962/projects/mmdetection || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmdec || true

mkdir -p "$WORKDIR/eval_logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$WORKDIR/eval_logs/eval_$TS.log"
SUMMARY="$WORKDIR/eval_logs/eval_$TS.metrics.txt"

echo "[$(date)] CONFIG=$CONFIG" | tee -a "$LOG"
echo "[$(date)] CKPT=$CKPT" | tee -a "$LOG"
echo "[$(date)] WORKDIR=$WORKDIR" | tee -a "$LOG"
echo "[$(date)] OUT_PKL=$OUT_PKL" | tee -a "$LOG"

python tools/test.py "$CONFIG" "$CKPT" \
  --work-dir "$WORKDIR" \
  --out "$OUT_PKL" \
  --cfg-options \
    test_dataloader.num_workers=0 \
    test_dataloader.persistent_workers=False \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=False \
  2>&1 | tee -a "$LOG"

python - <<'PY' "$LOG" "$SUMMARY"
import re
import sys

log_path = sys.argv[1]
summary_path = sys.argv[2]
pattern = re.compile(r'coco/bbox_mAP:|coco/segm_mAP:')
target_line = None
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if pattern.search(line):
            target_line = line.strip()

with open(summary_path, 'w', encoding='utf-8') as f:
    if target_line is None:
        f.write('METRICS_NOT_FOUND\n')
    else:
        f.write(target_line + '\n')

if target_line is None:
    print('METRICS_NOT_FOUND')
else:
    print(target_line)
PY

echo "[$(date)] metrics summary saved to $SUMMARY" | tee -a "$LOG"