#!/usr/bin/env bash
# Fast single-GPU end-to-end prototype: a few hundred iterations on a (subset of
# the) LMDB to validate the full DINOv3 12-band pipeline before the GCP run.
#
# Usage:
#   bash train_prototype.sh <LMDB_PATH> <OUTPUT_DIR> [BATCH_PER_GPU]
set -euo pipefail

LMDB_PATH="${1:?need LMDB path}"
OUTPUT_DIR="${2:?need output dir}"
BATCH_PER_GPU="${3:-16}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
PRETRAIN_SSL_DIR="$(dirname "$PKG_DIR")"

export PYTHONPATH="$PRETRAIN_SSL_DIR:${PYTHONPATH:-}"

# Resolve torchrun from the uv venv created by gcp_setup.sh, falling back to PATH.
DATA_DIR="${DATA_DIR:-/data}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/venv}"
TORCHRUN="$([[ -x "$VENV_DIR/bin/torchrun" ]] && echo "$VENV_DIR/bin/torchrun" || echo "torchrun")"

# Tiny schedule: 2 "epochs" of 50 iters each = 100 steps. compile off for speed
# of first iteration and clearer errors during prototyping.
"$TORCHRUN" --nproc_per_node=1 --master_port="${MASTER_PORT:-29501}" \
  "$PKG_DIR/train_ssl4eo.py" \
  --config-file "$PKG_DIR/configs/ssl4eo_s2_vits16.yaml" \
  --output-dir "$OUTPUT_DIR" \
  train.dataset_path="SSL4EOS2:root=${LMDB_PATH}" \
  train.batch_size_per_gpu="$BATCH_PER_GPU" \
  train.OFFICIAL_EPOCH_LENGTH=50 \
  train.num_workers=4 \
  train.compile=false \
  optim.epochs=2 \
  optim.warmup_epochs=1 \
  teacher.warmup_teacher_temp_epochs=1 \
  dino.head_n_prototypes=4096 \
  ibot.head_n_prototypes=4096 \
  checkpointing.period=50
