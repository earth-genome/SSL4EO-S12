#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run to validate DINOv3 objective wiring.
python -u pretrain_dino_s2c.py \
  --data /path/to/ssl4eo_s2c.lmdb \
  --checkpoints_dir /tmp/ssl4eo_dinov3_smoke \
  --bands B13 \
  --lmdb \
  --arch vit_small \
  --patch_size 16 \
  --num_workers 2 \
  --batch_size_per_gpu 2 \
  --epochs 1 \
  --warmup_epochs 0 \
  --lr 1.5e-4 \
  --optimizer adamw \
  --objective dino_v3 \
  --koleo_weight 0.1 \
  --use_koleo true \
  --mode s2c \
  --dtype uint8 \
  --season augment \
  --in_size 224
