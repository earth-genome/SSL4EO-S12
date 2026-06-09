#!/usr/bin/env bash
# Resume the DINOv3 ViT-S/16 SSL4EO-S2 pretraining that OOM-killed at iter ~39030/67400.
#
# Why it died: a HOST-RAM OOM (kernel OOM-killer reaped a DataLoader worker:
#   "DataLoader worker (pid ...) is killed by signal: Killed").
#   GPU mem was only ~13GB/80GB, so this was NOT a GPU OOM. With num_workers=10
#   x 4 GPUs = 40 persistent workers (persistent_workers is forced True in
#   ssl4eo_dinov3/patches.py), per-worker RAM creeps upward and never releases,
#   so 334GB filled after ~22h.
#
# Fix applied here: cut workers 10 -> 6. In the prior run data loading was only
#   ~0.008s of a ~2.08s step (<0.5%), so fewer workers costs ~nothing in
#   throughput but multiplies the OOM headroom (~6.6x more iterations before the
#   same RAM ceiling -> far past the 67400 total). MALLOC_TRIM_THRESHOLD_=0 also
#   makes glibc return freed memory to the OS more aggressively.
#
# Resume is automatic: train.py defaults to resume=True, which loads the latest
#   checkpoint (/data/out/vits16/ckpt/37499 -> restarts at iter 37500) and
#   advances the data sampler. Do NOT pass --no-resume.
set -euo pipefail

cd /home/snau/code/SSL4EO-S12/src/benchmark/pretrain_ssl

# --- knobs ---------------------------------------------------------------
NUM_WORKERS="${NUM_WORKERS:-6}"          # was 10; lowered to curb host-RAM OOM
LMDB=/data/ssl4eo_s2c_uint8.lmdb
OUT=/data/out/vits16
CONFIG=configs/ssl4eo_s2_vits16.yaml
BATCH_PER_GPU=64
LOG="restart_train_$(date +%Y%m%d_%H%M%S).log"

# Corrected venv path (run_train.sh pointed at a non-existent /SSl4EO-12/.venv
# and silently fell back to PATH torchrun).
export VENV_DIR=/home/snau/code/SSL4EO-S12/.venv
export DATA_DIR=/data
export DATASET_SIZE=172348               # keep => OFFICIAL_EPOCH_LENGTH stays 674
export MALLOC_TRIM_THRESHOLD_=0          # release freed heap back to the OS

# --- launch (detached so it survives SSH disconnect) ---------------------
nohup bash ssl4eo_dinov3/scripts/train_gcp.sh \
  "$LMDB" "$OUT" "$CONFIG" "$BATCH_PER_GPU" \
  train.num_workers="$NUM_WORKERS" \
  > "$LOG" 2>&1 &

PID=$!
echo "Resuming training (PID $PID); logging to $LOG"
echo "Follow with:  tail -f $LOG"
echo "Stop with:    kill $PID"
