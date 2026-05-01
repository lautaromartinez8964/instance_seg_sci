#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"

CURRENT_CONFIG="configs/gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big.py"
CURRENT_WORKDIR="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big"
NEXT_CONFIG="configs/gravel_big/rtmdet_ins_tiny_nwd_assign_120e_gravel_big.py"
NEXT_WORKDIR="${NEXT_WORKDIR:-$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_nwd_assign_after_dt_v3_120e_gravel_big}"
NEXT_SESSION_NAME="${NEXT_SESSION_NAME:-RTMDet-Ins Tiny NWD Assign After DT v3 120e}"
QUEUE_LOG="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_post_dt_v3_nwd_assign_queue.log"

POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-120}"
STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"
export STALL_TIMEOUT_SECONDS

mkdir -p "$ROOT_DIR/work_dirs_gravel_big"
touch "$QUEUE_LOG"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

is_target_complete() {
  local workdir="$1"
  local target_epoch="${2:-120}"
  local log_file

  if [[ -f "$workdir/epoch_${target_epoch}.pth" ]]; then
    return 0
  fi

  if [[ -d "$workdir/auto_logs" ]]; then
    while IFS= read -r -d '' log_file; do
      if grep -qE "Epoch\\(val\\) \[$target_epoch\]\[[0-9]+/[0-9]+\].*coco/segm_mAP|Epoch\\(train\\) \[$target_epoch\]\[[0-9]+/[0-9]+\]" "$log_file" 2>/dev/null; then
        return 0
      fi
    done < <(find "$workdir/auto_logs" -maxdepth 1 -type f -name 'run_*.log' -print0 2>/dev/null)
  fi

  return 1
}

launch_next_training() {
  if is_target_complete "$NEXT_WORKDIR" 120; then
    log "skip next training (already complete): $NEXT_WORKDIR"
    return 0
  fi

  log "launch next training: $NEXT_SESSION_NAME"
  log "next config: $NEXT_CONFIG"
  log "next workdir: $NEXT_WORKDIR"

  bash "$AUTO_SCRIPT" \
    "$ROOT_DIR/$NEXT_CONFIG" \
    "$NEXT_WORKDIR" \
    "" \
    "$NEXT_SESSION_NAME" \
    --amp \
    --cfg-options \
    train_dataloader.batch_size=8 \
    optim_wrapper.accumulative_counts=2 \
    train_dataloader.num_workers=0 \
    train_dataloader.persistent_workers=False \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=False \
    test_dataloader.num_workers=0 \
    test_dataloader.persistent_workers=False \
    default_hooks.checkpoint.by_epoch=False \
    default_hooks.checkpoint.interval=200 \
    default_hooks.checkpoint.max_keep_ckpts=20 \
    train_cfg.val_interval=10
}

if is_target_complete "$CURRENT_WORKDIR" 120; then
  log "current DT v3 training already complete, launching NWD assign immediately"
  launch_next_training
  exit $?
fi

log "waiting for current DT v3 training to complete"
log "current config: $CURRENT_CONFIG"
log "current workdir: $CURRENT_WORKDIR"
log "next config: $NEXT_CONFIG"
log "next workdir: $NEXT_WORKDIR"

while true; do
  if is_target_complete "$CURRENT_WORKDIR" 120; then
    log "detected DT v3 completion signal"
    launch_next_training
    exit $?
  fi
  sleep "$POLL_INTERVAL_SECONDS"
done