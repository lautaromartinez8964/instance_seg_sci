#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"
CONFIG_PATH="$ROOT_DIR/configs/gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big.py"
WORKDIR="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_baseline_120e_gravel_big"
RESUME_FLAG="$WORKDIR/.resume_on_boot_enabled"
CONTROL_LOG="$WORKDIR/scheduled_control.log"
SESSION_NAME="RTMDet-Ins Tiny Baseline 120e power-resume"

mkdir -p "$WORKDIR"

if [[ ! -f "$RESUME_FLAG" ]]; then
  echo "[$(date '+%F %T')] resume flag missing; skip auto-resume" >> "$CONTROL_LOG"
  exit 0
fi

if ps -eo cmd= | awk -v config="$CONFIG_PATH" -v workdir="$WORKDIR" 'index($0, config) && index($0, workdir) && index($0, "tools/train.py") {found=1} END {exit !found}'; then
  echo "[$(date '+%F %T')] matching training process already running; skip auto-resume" >> "$CONTROL_LOG"
  exit 0
fi

echo "[$(date '+%F %T')] launching auto-resume" | tee -a "$CONTROL_LOG"
nohup bash "$AUTO_SCRIPT" \
  "$CONFIG_PATH" \
  "$WORKDIR" \
  "" \
  "$SESSION_NAME" \
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
  train_cfg.val_interval=10 \
  >> "$CONTROL_LOG" 2>&1 &

echo "[$(date '+%F %T')] auto-resume launched with pid=$!" | tee -a "$CONTROL_LOG"