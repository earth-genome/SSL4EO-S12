# Container for DINOv3 12-band Sentinel-2 pretraining on GCP (GCE A100/H100 or
# Vertex AI custom training). Built with DINOv3 (facebookresearch/dinov3).
#
# Build context = repository ROOT (SSL4EO-S12/), so the adapter package can be
# copied. The DINOv3 submodule is cloned (pinned) inside the image to keep the
# image self-contained regardless of host submodule state.
#
#   docker build -f src/benchmark/pretrain_ssl/ssl4eo_dinov3/Dockerfile \
#       -t gcr.io/<PROJECT>/ssl4eo-dinov3:latest .
#
# Swap the base tag if you need a different CUDA/PyTorch (must be torch>=2.7.1).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DINOV3_COMMIT=31703e4cbf1ccb7c4a72daa1350405f86754b6d1
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/pretrain_ssl \
    NVIDIA_TF32_OVERRIDE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/pretrain_ssl

# --- DINOv3 (pinned) -------------------------------------------------------
RUN git clone https://github.com/facebookresearch/dinov3 dinov3 && \
    cd dinov3 && git checkout "${DINOV3_COMMIT}" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# --- Adapter package + extra deps -----------------------------------------
COPY src/benchmark/pretrain_ssl/ssl4eo_dinov3 /workspace/pretrain_ssl/ssl4eo_dinov3
RUN pip install --no-cache-dir -r /workspace/pretrain_ssl/ssl4eo_dinov3/requirements-gcp.txt

WORKDIR /workspace/pretrain_ssl/ssl4eo_dinov3

# Default: print help. Override CMD/entrypoint to launch training, e.g.
#   bash scripts/train_gcp.sh gs_local/data.lmdb /workspace/out
CMD ["python", "-c", "import ssl4eo_dinov3; print('SSL4EO-S12 x DINOv3 image ready. See scripts/train_gcp.sh')"]
