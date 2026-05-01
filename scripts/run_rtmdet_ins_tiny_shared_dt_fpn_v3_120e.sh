#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTO_SCRIPT="$ROOT_DIR/scripts/auto_restart_train_from_ckpt.sh"
CONFIG_PATH="$ROOT_DIR/configs/gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big.py"
WORKDIR="$ROOT_DIR/work_dirs_gravel_big/rtmdet_ins_tiny_shared_dt_fpn_v3_120e_gravel_big"
SESSION_NAME="RTMDet-Ins Tiny Shared DT v3 120e"

mkdir -p "$WORKDIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export STALL_TIMEOUT_SECONDS="${STALL_TIMEOUT_SECONDS:-1800}"

exec bash "$AUTO_SCRIPT" \
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
  default_hooks.checkpoint.by_epoch=True \
  default_hooks.checkpoint.interval=30 \
  default_hooks.checkpoint.max_keep_ckpts=20 \
  train_cfg.val_interval=10