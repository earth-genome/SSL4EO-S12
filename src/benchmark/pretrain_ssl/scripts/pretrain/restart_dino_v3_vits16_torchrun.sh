#!/usr/bin/env bash
set -euo pipefail

# ViT-S/16 DINOv2 -> DINOv3 restart launcher (local torchrun).
# Defaults target S2-only LMDB training (no S1 required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -------------------------------
# User-overridable settings
# -------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

DATA_PATH="${DATA_PATH:-./data/ssl4eo_250k_s2c_uint8.lmdb}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints/dino_v3_restart_vits16}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-64}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LR="${LR:-1.5e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
IN_SIZE="${IN_SIZE:-224}"
SEED="${SEED:-42}"

# DINOv2 checkpoint to restart from (provided by user).
V2_CKPT_FILE_ID="${V2_CKPT_FILE_ID:-1CseO5vvMReGlAulm5o4ZgbjUgj8VlAH7}"
RESTART_CKPT_PATH="${CHECKPOINTS_DIR}/checkpoint.pth"

# Optional training-data download behavior.
# If DATA_PATH is missing and this is true:
# - if DATA_URL is set, script downloads DATA_URL to DATA_PATH
# - otherwise it exits with actionable instructions.
DOWNLOAD_DATA_IF_MISSING="${DOWNLOAD_DATA_IF_MISSING:-false}"
DATA_URL="${DATA_URL:-}"

mkdir -p "${CHECKPOINTS_DIR}"

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

download_data_if_needed() {
  if [[ -e "${DATA_PATH}" ]]; then
    return 0
  fi

  if [[ "${DOWNLOAD_DATA_IF_MISSING}" != "true" ]]; then
    echo "Training data not found at: ${DATA_PATH}"
    echo "Set DATA_PATH to an existing LMDB or set DOWNLOAD_DATA_IF_MISSING=true with DATA_URL."
    return 1
  fi

  if [[ -z "${DATA_URL}" ]]; then
    echo "DOWNLOAD_DATA_IF_MISSING=true but DATA_URL is empty."
    echo "Provide a direct LMDB/archive URL in DATA_URL, or pre-create DATA_PATH."
    return 1
  fi

  echo "Training data missing; downloading from DATA_URL -> ${DATA_PATH}"
  mkdir -p "$(dirname "${DATA_PATH}")"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${DATA_URL}" -o "${DATA_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${DATA_PATH}" "${DATA_URL}"
  else
    echo "Neither curl nor wget found; cannot download training data."
    return 1
  fi
}

if [[ ! -f "${RESTART_CKPT_PATH}" ]]; then
  download_from_gdrive "${V2_CKPT_FILE_ID}" "${RESTART_CKPT_PATH}"
fi

if [[ ! -f "${RESTART_CKPT_PATH}" ]]; then
  echo "Checkpoint download failed: ${RESTART_CKPT_PATH}"
  exit 1
fi

download_data_if_needed

echo "Starting restart run from checkpoint: ${RESTART_CKPT_PATH}"
echo "Training data path: ${DATA_PATH}"
echo "Objective: dino_v3_full (dino + ibot + gram)"
echo "Backbone: vit_small / patch16 / in_size=${IN_SIZE}"

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
  --resume
