#!/usr/bin/env bash
set -euo pipefail

# Limited DINOv3 restart run (15 epochs) for quick validation.
# Wraps restart_dino_v3_vits16_torchrun.sh with short-run overrides.
#
# Usage:
#   bash limited_restart_dino_v3_vits16.sh
#
# All env-var overrides from the base script still work, e.g.:
#   DATA_PATH=/my/data.lmdb NPROC_PER_NODE=2 bash limited_restart_dino_v3_vits16.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

DATA_PATH="${DATA_PATH:-./data/ssl4eo_250k_s2c_uint8.lmdb}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints/dino_v3_limited_vits16}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LR="${LR:-1.5e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
IN_SIZE="${IN_SIZE:-224}"
SEED="${SEED:-42}"
SAVECKP_FREQ="${SAVECKP_FREQ:-5}"

V2_CKPT_FILE_ID="${V2_CKPT_FILE_ID:-1CseO5vvMReGlAulm5o4ZgbjUgj8VlAH7}"
RESTART_CKPT_PATH="${CHECKPOINTS_DIR}/checkpoint.pth"

DOWNLOAD_DATA_IF_MISSING="${DOWNLOAD_DATA_IF_MISSING:-false}"
DATA_URL="${DATA_URL:-}"

mkdir -p "${CHECKPOINTS_DIR}"

# Reuse helpers from the base script.
source "${SCRIPT_DIR}/restart_dino_v3_vits16_torchrun.sh" --source-only 2>/dev/null || true

download_from_gdrive() {
  local file_id="$1"
  local out_path="$2"
  if command -v gdown >/dev/null 2>&1; then
    echo "Downloading DINOv2 checkpoint via gdown -> ${out_path}"
    gdown --id "${file_id}" --output "${out_path}"
    return 0
  fi
  echo "gdown not found. Installing it in user site-packages..."
  python -m pip install --user gdown
  if ! command -v gdown >/dev/null 2>&1; then
    local pybin
    pybin="$(python -c 'import site; print(site.USER_BASE + "/bin")')"
    export PATH="${pybin}:${PATH}"
  fi
  gdown --id "${file_id}" --output "${out_path}"
}

if [[ ! -f "${RESTART_CKPT_PATH}" ]]; then
  download_from_gdrive "${V2_CKPT_FILE_ID}" "${RESTART_CKPT_PATH}"
fi

if [[ ! -f "${RESTART_CKPT_PATH}" ]]; then
  echo "Checkpoint download failed: ${RESTART_CKPT_PATH}"
  exit 1
fi

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "Training data not found at: ${DATA_PATH}"
  echo "Set DATA_PATH to an existing LMDB."
  exit 1
fi

echo "=== Limited DINOv3 restart run ==="
echo "Epochs:          ${EPOCHS} (warmup ${WARMUP_EPOCHS})"
echo "Checkpoint freq: ${SAVECKP_FREQ}"
echo "Checkpoint dir:  ${CHECKPOINTS_DIR}"
echo "Data:            ${DATA_PATH}"

cd "${PRETRAIN_DIR}"
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  pretrain_dino_s2c.py \
  --data "${DATA_PATH}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --bands B13 \
  --lmdb \
  --arch vit_small \
  --patch_size 16 \
  --num_workers "${NUM_WORKERS}" \
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --lr "${LR}" \
  --optimizer adamw \
  --objective dino_v3 \
  --dino_v3_mode full \
  --enable_ibot true \
  --ibot_weight 1.0 \
  --gram_weight 0.1 \
  --gram_teacher_checkpoint "${RESTART_CKPT_PATH}" \
  --enable_rope true \
  --drop_legacy_pos_embed true \
  --koleo_weight 0.1 \
  --use_koleo true \
  --seed "${SEED}" \
  --mode s2c \
  --dtype uint8 \
  --season augment \
  --in_size "${IN_SIZE}" \
  --schedule_mode constant_after_warmup \
  --saveckp_freq "${SAVECKP_FREQ}" \
  --resume
