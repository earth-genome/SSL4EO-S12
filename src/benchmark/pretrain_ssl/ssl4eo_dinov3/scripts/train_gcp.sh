#!/usr/bin/env bash
# Launch DINOv3 12-band Sentinel-2 pretraining on a single multi-GPU node
# (e.g. GCE a2-ultragpu-8g / A100-80GB or a3-highgpu-8g / H100) via torchrun.
#
# Usage:
#   bash train_gcp.sh <LMDB_PATH> <OUTPUT_DIR> [CONFIG] [BATCH_PER_GPU] [EXTRA opts...]
#
# Example:
#   bash train_gcp.sh /data/ssl4eo_s2c_uint8.lmdb gs_or_local/out \
#       configs/ssl4eo_s2_vits16.yaml 64
set -euo pipefail

LMDB_PATH="${1:?need LMDB path}"
OUTPUT_DIR="${2:?need output dir}"
CONFIG="${3:-configs/ssl4eo_s2_vits16.yaml}"
BATCH_PER_GPU="${4:-64}"
shift $(( $# < 4 ? $# : 4 )) || true
EXTRA_OPTS=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"             # ssl4eo_dinov3/
PRETRAIN_SSL_DIR="$(dirname "$PKG_DIR")" # pretrain_ssl/
CONFIG_PATH="$([[ "$CONFIG" = /* ]] && echo "$CONFIG" || echo "$PKG_DIR/$CONFIG")"

# Resolve torchrun from the uv venv created by gcp_setup.sh, falling back to PATH.
DATA_DIR="${DATA_DIR:-/data}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/venv}"
TORCHRUN="$([[ -x "$VENV_DIR/bin/torchrun" ]] && echo "$VENV_DIR/bin/torchrun" || echo "torchrun")"

# Resolve dataset size and GPU count to size one "epoch".
DATASET_SIZE="${DATASET_SIZE:-250000}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
GLOBAL_BATCH=$(( BATCH_PER_GPU * NUM_GPUS ))
OEL=$(( (DATASET_SIZE + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))  # ceil

echo "[train_gcp] GPUs=$NUM_GPUS  global_batch=$GLOBAL_BATCH  OFFICIAL_EPOCH_LENGTH=$OEL"
echo "[train_gcp] config=$CONFIG_PATH  lmdb=$LMDB_PATH  out=$OUTPUT_DIR"

# Performance / correctness environment.
export PYTHONPATH="$PRETRAIN_SSL_DIR:${PYTHONPATH:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# TF32 is also set inside dinov3/train/train.py; harmless to reaffirm here.
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"

"$TORCHRUN" \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="${MASTER_PORT:-29500}" \
  "$PKG_DIR/train_ssl4eo.py" \
  --config-file "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  train.dataset_path="SSL4EOS2:root=${LMDB_PATH}" \
  train.batch_size_per_gpu="$BATCH_PER_GPU" \
  train.OFFICIAL_EPOCH_LENGTH="$OEL" \
  "${EXTRA_OPTS[@]}"
