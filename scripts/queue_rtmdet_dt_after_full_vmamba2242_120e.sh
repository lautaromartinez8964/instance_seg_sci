#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERIAL_SCRIPT="$ROOT_DIR/scripts/run_gravel_big_rtmdet_serial_120e.sh"
FULL_WORKDIR="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_vmamba2242_120e_gravel_big"
QUEUE_LOG="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_post_full_vmamba2242_dt_v3_queue.log"
FULL_LOG="$FULL_WORKDIR/auto_restart_runner.log"

mkdir -p "$ROOT_DIR/work_dirs_gravel_big"
touch "$QUEUE_LOG"
touch "$FULL_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

is_full_training_complete() {
  if [[ -f "$FULL_WORKDIR/iter_29640.pth" ]]; then
    return 0
  fi

  if grep -qE 'Epoch\(val\) \[120\]\[[0-9]+/[0-9]+\].*coco/segm_mAP' "$FULL_LOG" 2>/dev/null; then
    return 0
  fi

  return 1
}

launch_dt_v3() {
  log "launch training: RTMDet-Ins Tiny Shared DT v3 120e"
  START_INDEX=3 bash "$SERIAL_SCRIPT"
}

if is_full_training_complete; then
  log "full vmamba2242 already complete, launching DT v3 immediately"
  launch_dt_v3
  exit 0
fi

log "waiting for full vmamba2242 completion before launching DT v3"
tail -Fn0 "$FULL_LOG" | while IFS= read -r line; do
  if [[ "$line" =~ Epoch\(val\)\ \[120\]\[[0-9]+/[0-9]+\].*coco/segm_mAP ]] || \
     [[ "$line" =~ Saving\ checkpoint\ at\ 29640\ iterations ]]; then
    log "detected full vmamba2242 completion signal"
    launch_dt_v3
    break
  fi
done