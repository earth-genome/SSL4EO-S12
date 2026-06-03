#!/usr/bin/env bash
# On-VM bootstrap for DINOv3 12-band Sentinel-2 pretraining:
#   1. get the code (use the surrounding checkout, or clone the public repo)
#   2. install deps (DINOv3 submodule + adapter + data-prep libs + hf_xet)
#   3. download the SSL4EO-S12 S2-L1C data from Hugging Face over Xet
#   4. extract the split tarball and arrange the layout SSL4EO expects
#   5. (optional) build the s2c LMDB the training reads
#
# Safe to run via pipe:
#   curl -fsSL https://raw.githubusercontent.com/earth-genome/SSL4EO-S12/main/\
#src/benchmark/pretrain_ssl/ssl4eo_dinov3/scripts/gcp_setup.sh | DATA_DIR=/data bash
#
# Or from a checkout:
#   DATA_DIR=/data BUILD_LMDB=1 bash gcp_setup.sh
#
# Knobs (env vars):
#   DATA_DIR      working dir for data + clone        (default /data)
#   REPO_URL      git URL if cloning                  (default earth-genome public repo)
#   REPO_BRANCH   branch to clone                     (default main)
#   HF_REPO       HF dataset id                       (default wangyi111/SSL4EO-S12)
#   MODALITY      HF subfolder to pull                (default s2_l1c)
#   SKIP_INSTALL / SKIP_DOWNLOAD / SKIP_EXTRACT  set =1 to skip a stage
#   BUILD_LMDB    =1 to also build the LMDB           (default 0)
#   LMDB_DTYPE    uint8 | int16                       (default uint8)
#   SRC_DIR       override the detected extracted s2c folder
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
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

# --- 2. Install -------------------------------------------------------------
if [[ "$SKIP_INSTALL" != "1" ]]; then
  echo "[setup] initializing DINOv3 submodule + installing deps"
  git -C "$REPO_DIR" submodule update --init "src/benchmark/pretrain_ssl/dinov3"
  python -c "import torch,sys;v=torch.__version__;print('[setup] torch',v,'cuda',torch.cuda.is_available());sys.exit(0 if tuple(int(x) for x in v.split('+')[0].split('.')[:2])>=(2,7) else 1)" \
    || { echo '[setup] torch < 2.7; upgrading'; pip install -U "torch>=2.7.1"; }
  pip install -r "$PRETRAIN_SSL_DIR/dinov3/requirements.txt"
  pip install -e "$PRETRAIN_SSL_DIR/dinov3"
  pip install -r "$PKG_DIR/requirements-gcp.txt"
  # Xet-accelerated HF transfers + data-prep libs (build_lmdb reads GeoTIFFs).
  pip install -U "huggingface_hub[hf_xet]" hf_xet opencv-python-headless rasterio
fi

# --- 3. Download from Hugging Face over Xet --------------------------------
export HF_HOME="${HF_HOME:-$DATA_DIR/hf_cache}"     # keep Xet cache off the boot disk
export HF_HUB_ENABLE_HF_XET=1
export HF_XET_HIGH_PERFORMANCE=1                    # saturate NIC + all cores
HF_RAW_DIR="$DATA_DIR/hf"
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  echo "[setup] downloading $HF_REPO :: $MODALITY/* over Xet -> $HF_RAW_DIR"
  hf download "$HF_REPO" --repo-type dataset --include "$MODALITY/*" --local-dir "$HF_RAW_DIR"
fi

# --- 4. Extract the split tarball + arrange <root>/s2c ----------------------
EXTRACT_DIR="$DATA_DIR/extract"
S2C_ROOT="$DATA_DIR/s2c_root"
if [[ "$SKIP_EXTRACT" != "1" ]]; then
  echo "[setup] extracting $MODALITY split tarball (streamed; no extra temp copy)"
  mkdir -p "$EXTRACT_DIR"
  # Parts are a single tar.gz split into .partaa.. ; concatenate in order and untar.
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
  python "$PKG_DIR/scripts/build_lmdb.py" \
    --root "$S2C_ROOT" --out "$LMDB_OUT" \
    --dtype "$LMDB_DTYPE" --map-size-gb "$MAP_GB" --num-workers 8
fi

cat <<EOF

[setup] done.
  code:   $REPO_DIR
  s2c:    $S2C_ROOT/s2c
$( [[ "$BUILD_LMDB" == "1" ]] && echo "  lmdb:   $LMDB_OUT" || echo "  lmdb:   not built (set BUILD_LMDB=1, or run scripts/build_lmdb.py)" )

Next:
  cd $PRETRAIN_SSL_DIR
  # build LMDB if not done:        python ssl4eo_dinov3/scripts/build_lmdb.py --root $S2C_ROOT --out $LMDB_OUT
  # recompute per-band stats:      python ssl4eo_dinov3/scripts/compute_band_stats.py --lmdb $LMDB_OUT --samples 20000
  # 1-GPU prototype:               bash ssl4eo_dinov3/scripts/train_prototype.sh $LMDB_OUT $DATA_DIR/proto_out
  # full run:                      bash ssl4eo_dinov3/scripts/train_gcp.sh $LMDB_OUT $DATA_DIR/out/vits16
EOF
