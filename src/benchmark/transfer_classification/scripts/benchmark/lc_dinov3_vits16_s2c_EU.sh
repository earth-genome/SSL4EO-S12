#!/usr/bin/env bash
set -euo pipefail

# Linear classification eval on EuroSAT using a DINOv3 ViT-S/16 checkpoint.
# Uses torchrun for local multi-GPU execution.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"

DATA_DIR="${DATA_DIR:-/p/scratch/hai_ssl4eo/data/eurosat/tif}"
PRETRAINED="${PRETRAINED:-./checkpoints/dino_v3_limited_vits16/checkpoint.pth}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints/dinov3_lc/EU_vits16}"

mkdir -p "${CHECKPOINTS_DIR}"

cd "${EVAL_DIR}"
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  linear_EU_dino.py \
  --data_dir "${DATA_DIR}" \
  --bands B13 \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --arch vit_small \
  --patch_size 16 \
  --train_frac 1.0 \
  --batch_size_per_gpu 64 \
  --lr 0.01 \
  --epochs 100 \
  --num_workers 10 \
  --seed 42 \
  --checkpoint_key teacher_backbone \
  --use_rope \
  --pretrained "${PRETRAINED}"
