#!/usr/bin/env bash
# Certify a trained DINOv3 12-band model against the original DINO ViT-S/16
# SSL4EO baselines. Extracts frozen embeddings for each benchmark's train/val
# tiles, runs a linear probe, and gates on the published numbers.
#
# Prereqs: per-benchmark tiles exported as .npy (12- or 13-band; B10 auto-dropped)
# plus label arrays. Point the *_DIR / *_Y variables at your prepared splits.
#
# Usage:
#   CKPT=/out/eval/<it>/teacher_checkpoint.pth \
#   CONFIG=../configs/ssl4eo_s2_vits16.yaml \
#   bash run_certification.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
CONFIG="${CONFIG:-$PKG_DIR/configs/ssl4eo_s2_vits16.yaml}"
CKPT="${CKPT:?set CKPT to the teacher_checkpoint.pth}"
WORK="${WORK:-./cert_embeddings}"
mkdir -p "$WORK"

extract () {  # <split-name> <source> <input>
  python "$PKG_DIR/extract_embeddings.py" \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --source "$2" --input "$3" --out "$WORK/$1"
}

# ---- EuroSAT (multiclass, baseline 99.0 acc) ------------------------------
if [[ -n "${EUROSAT_TRAIN_DIR:-}" ]]; then
  extract eurosat_train npy "$EUROSAT_TRAIN_DIR"
  extract eurosat_val   npy "$EUROSAT_VAL_DIR"
  python "$HERE/linear_probe.py" --name EuroSAT --task multiclass --knn \
    --train-x "$WORK/eurosat_train.npy" --train-y "$EUROSAT_TRAIN_Y" \
    --val-x   "$WORK/eurosat_val.npy"   --val-y   "$EUROSAT_VAL_Y" \
    --baseline 99.0
fi

# ---- BigEarthNet (multilabel, baseline 90.5 mAP) --------------------------
if [[ -n "${BE_TRAIN_DIR:-}" ]]; then
  extract be_train npy "$BE_TRAIN_DIR"
  extract be_val   npy "$BE_VAL_DIR"
  python "$HERE/linear_probe.py" --name BigEarthNet --task multilabel \
    --train-x "$WORK/be_train.npy" --train-y "$BE_TRAIN_Y" \
    --val-x   "$WORK/be_val.npy"   --val-y   "$BE_VAL_Y" \
    --baseline 90.5
fi

# ---- So2Sat (multiclass, baseline 62.2 acc) -------------------------------
if [[ -n "${SS_TRAIN_DIR:-}" ]]; then
  extract ss_train npy "$SS_TRAIN_DIR"
  extract ss_val   npy "$SS_VAL_DIR"
  python "$HERE/linear_probe.py" --name So2Sat --task multiclass --knn \
    --train-x "$WORK/ss_train.npy" --train-y "$SS_TRAIN_Y" \
    --val-x   "$WORK/ss_val.npy"   --val-y   "$SS_VAL_Y" \
    --baseline 62.2
fi

echo "[certification] done (any FAIL above exits non-zero)."
