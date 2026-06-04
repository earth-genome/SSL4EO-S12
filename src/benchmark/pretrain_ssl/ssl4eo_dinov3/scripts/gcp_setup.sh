#!/usr/bin/env bash
# On-VM bootstrap for DINOv3 12-band Sentinel-2 pretraining.
# Uses uv for an isolated Python venv – no system-Python pollution.
#
#   1. install uv + create a local Python venv
#   2. get the code (use the surrounding checkout, or clone the public repo)
#   3. install deps (DINOv3 submodule + adapter + data-prep libs + hf_xet)
#   4. download the SSL4EO-S12 S2-L1C data from Hugging Face over Xet
#   5. extract the split tarball and arrange the layout SSL4EO expects
#   6. (optional) build the s2c LMDB the training reads
#
# Safe to run via pipe:
#   curl -fsSL https://raw.githubusercontent.com/earth-genome/SSL4EO-S12/main/\
#src/benchmark/pretrain_ssl/ssl4eo_dinov3/scripts/gcp_setup.sh | DATA_DIR=/data bash
#
# Or from a checkout:
#   DATA_DIR=/data BUILD_LMDB=1 bash gcp_setup.sh
#
# Knobs (env vars):
#   DATA_DIR       working dir for data + clone        (default /data)
#   VENV_DIR       uv venv location                    (default $DATA_DIR/venv)
#   PYTHON_VERSION Python version to pin               (default 3.11)
#   REPO_URL       git URL if cloning                  (default earth-genome public repo)
#   REPO_BRANCH    branch to clone                     (default main)
#   HF_REPO        HF dataset id                       (default wangyi111/SSL4EO-S12)
#   MODALITY       HF subfolder to pull                (default s2_l1c)
#   SKIP_INSTALL / SKIP_DOWNLOAD / SKIP_EXTRACT  set =1 to skip a stage
#   BUILD_LMDB     =1 to also build the LMDB           (default 0)
#   LMDB_DTYPE     uint8 | int16                       (default uint8)
#   SRC_DIR        override the detected extracted s2c folder
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REPO_URL="${REPO_URL:-https://github.com/earth-genome/SSL4EO-S12.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
HF_REPO="${HF_REPO:-wangyi111/SSL4EO-S12}"
MODALITY="${MODALITY:-s2_l1c}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
BUILD_LMDB="${BUILD_LMDB:-0}"
LMDB_DTYPE="${LMDB_DTYPE:-uint8}"

mkdir -p "$DATA_DIR"

# --- 0. uv + isolated Python venv -------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Ensure uv's default install dir is on PATH (idempotent).
export PATH="$HOME/.local/bin:$PATH"

PYTHON="$VENV_DIR/bin/python"
HF_CLI="$VENV_DIR/bin/hf"

if [[ ! -x "$PYTHON" ]]; then
  echo "[setup] creating Python $PYTHON_VERSION venv at $VENV_DIR"
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
echo "[setup] venv: $VENV_DIR"

# Export so downstream scripts (train_gcp.sh, train_prototype.sh) can pick it up.
export VENV_DIR

# --- 1. Locate (or clone) the repo -----------------------------------------
REPO_DIR=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]:-/nonexistent}" ]]; then
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _root="$(cd "$_here/../../../../.." && pwd)"   # scripts -> ... -> repo root
  [[ -d "$_root/.git" ]] && REPO_DIR="$_root"
fi
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="${REPO_DIR_OVERRIDE:-$DATA_DIR/SSL4EO-S12}"
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "[setup] cloning $REPO_URL ($REPO_BRANCH) -> $REPO_DIR"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
  fi
fi
echo "[setup] repo: $REPO_DIR"
PRETRAIN_SSL_DIR="$REPO_DIR/src/benchmark/pretrain_ssl"
PKG_DIR="$PRETRAIN_SSL_DIR/ssl4eo_dinov3"

# --- 2. Install into venv ---------------------------------------------------
if [[ "$SKIP_INSTALL" != "1" ]]; then
  echo "[setup] initializing DINOv3 submodule + installing deps into $VENV_DIR"
  git -C "$REPO_DIR" submodule update --init "src/benchmark/pretrain_ssl/dinov3"

  # Install torch first so the >=2.7.1 constraint wins before dinov3/requirements.txt
  # can pull in an older pin.
  uv pip install --python "$PYTHON" "torch>=2.7.1" torchvision
  uv pip install --python "$PYTHON" -r "$PRETRAIN_SSL_DIR/dinov3/requirements.txt"
  uv pip install --python "$PYTHON" -e "$PRETRAIN_SSL_DIR/dinov3"
  uv pip install --python "$PYTHON" -r "$PKG_DIR/requirements-gcp.txt"
  # Xet-accelerated HF transfers + GeoTIFF readers for build_lmdb / extract_embeddings.
  uv pip install --python "$PYTHON" "huggingface_hub[hf_xet]" hf_xet opencv-python-headless rasterio

  "$PYTHON" -c "
import torch, sys
v = torch.__version__
print('[setup] torch', v, 'cuda', torch.cuda.is_available())
major, minor = (int(x) for x in v.split('+')[0].split('.')[:2])
if (major, minor) < (2, 7):
    print('[setup] ERROR: torch', v, '< 2.7 – check requirements'); sys.exit(1)
"
fi

# --- 3. Download from Hugging Face over Xet --------------------------------
export HF_HOME="${HF_HOME:-$DATA_DIR/hf_cache}"   # keep Xet cache off the boot disk
export HF_HUB_ENABLE_HF_XET=1
export HF_XET_HIGH_PERFORMANCE=1                   # saturate NIC + all cores
HF_RAW_DIR="$DATA_DIR/hf"
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  echo "[setup] downloading $HF_REPO :: $MODALITY/* over Xet -> $HF_RAW_DIR"
  "$HF_CLI" download "$HF_REPO" \
    --repo-type dataset --include "$MODALITY/*" --local-dir "$HF_RAW_DIR"
fi

# --- 4. Extract the split tarball + arrange <root>/s2c ----------------------
EXTRACT_DIR="$DATA_DIR/extract"
S2C_ROOT="$DATA_DIR/s2c_root"
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  echo "[setup] extracting $MODALITY split tarball (streamed; no extra temp copy)"
  mkdir -p "$EXTRACT_DIR"
  # Parts are a single tar.gz split into .partaa..; concatenate in order and untar.
  cat "$HF_RAW_DIR/$MODALITY/"*.tar.gz.parta* | tar -xzf - -C "$EXTRACT_DIR"
  echo "[setup] extracted tree (top levels):"
  find "$EXTRACT_DIR" -maxdepth 2 -type d | head -n 20
fi

# Detect the folder that holds the per-patch directories, then expose it as
# <S2C_ROOT>/s2c so SSL4EO(root=S2C_ROOT, mode=['s2c']) finds it.
if [[ -z "${SRC_DIR:-}" ]]; then
  for cand in "$EXTRACT_DIR/s2c" "$EXTRACT_DIR/$MODALITY" "$EXTRACT_DIR/s2_l1c" "$EXTRACT_DIR"; do
    [[ -d "$cand" ]] && { SRC_DIR="$cand"; break; }
  done
fi
mkdir -p "$S2C_ROOT"
ln -sfn "$SRC_DIR" "$S2C_ROOT/s2c"
echo "[setup] s2c data root: $S2C_ROOT  (s2c -> $SRC_DIR)"
echo "[setup] sanity: $(find "$S2C_ROOT/s2c" -maxdepth 1 -mindepth 1 -type d | wc -l) patch dirs detected"
echo "[setup] If that count looks wrong, re-run with SRC_DIR=<correct patch folder>."

# --- 5. (Optional) build the LMDB ------------------------------------------
LMDB_OUT="$DATA_DIR/ssl4eo_s2c_${LMDB_DTYPE}.lmdb"
if [[ "$BUILD_LMDB" == "1" ]]; then
  MAP_GB=$([[ "$LMDB_DTYPE" == "int16" ]] && echo 2200 || echo 1100)
  echo "[setup] building $LMDB_DTYPE LMDB -> $LMDB_OUT (map_size=${MAP_GB} GiB)"
  "$PYTHON" "$PKG_DIR/scripts/build_lmdb.py" \
    --root "$S2C_ROOT" --out "$LMDB_OUT" \
    --dtype "$LMDB_DTYPE" --map-size-gb "$MAP_GB" --num-workers 8
fi

cat <<EOF

[setup] done.
  venv:   $VENV_DIR
  code:   $REPO_DIR
  s2c:    $S2C_ROOT/s2c
$( [[ "$BUILD_LMDB" == "1" ]] && echo "  lmdb:   $LMDB_OUT" || echo "  lmdb:   not built (set BUILD_LMDB=1, or run scripts/build_lmdb.py)" )

Next (activate the venv or let VENV_DIR propagate):
  source $VENV_DIR/bin/activate
  cd $PRETRAIN_SSL_DIR
  # build LMDB if not done:        python ssl4eo_dinov3/scripts/build_lmdb.py --root $S2C_ROOT --out $LMDB_OUT
  # recompute per-band stats:      python ssl4eo_dinov3/scripts/compute_band_stats.py --lmdb $LMDB_OUT --samples 20000
  # 1-GPU prototype:               bash ssl4eo_dinov3/scripts/train_prototype.sh $LMDB_OUT $DATA_DIR/proto_out
  # full run:                      bash ssl4eo_dinov3/scripts/train_gcp.sh $LMDB_OUT $DATA_DIR/out/vits16
EOF
