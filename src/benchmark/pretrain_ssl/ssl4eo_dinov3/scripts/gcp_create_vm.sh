#!/usr/bin/env bash
# Create a GCP GPU VM for DINOv3 12-band Sentinel-2 pretraining.
#
# Defaults: 1x A100-80GB (a2-ultragpu-1g) on an Ubuntu 22.04 + CUDA 12.8 + PyTorch
# 2.7 Deep Learning VM image (NVIDIA driver pre-installed), plus a large *persistent*
# SSD data disk sized for: HF tar parts (~557 GB) + extracted GeoTIFFs (~600 GB) +
# uint8 s2c LMDB (~0.9 TB) -> peak ~1.5 TB, so 2 TB by default.
#
# A persistent data disk (auto-delete=no) is used on purpose: the download + LMDB
# build is hours of work that should survive a VM stop / Spot preemption. For a
# true-ephemeral fast scratch instead, see LOCAL_SSD below.
#
# Usage:
#   PROJECT=my-proj bash gcp_create_vm.sh
#   PROJECT=my-proj MACHINE_TYPE=a2-ultragpu-8g SPOT=1 bash gcp_create_vm.sh
#
# Then:
#   gcloud compute ssh "$VM" --zone "$ZONE"
#   # on the VM: mount the data disk (see printed hint), then run gcp_setup.sh
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=<gcp-project-id>}"
ZONE="${ZONE:-us-central1-c}"           # a2-ultragpu lives in us-central1, europe-west4, asia-southeast1
VM="${VM:-ssl4eo-dinov3}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-1g}"   # 1x A100-80GB; full run: a2-ultragpu-8g / a3-highgpu-8g
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_GB="${BOOT_DISK_GB:-150}"
DATA_DISK_GB="${DATA_DISK_GB:-2048}"
DATA_DISK_TYPE="${DATA_DISK_TYPE:-pd-ssd}"
DATA_DISK_NAME="${DATA_DISK_NAME:-${VM}-data}"
SPOT="${SPOT:-0}"                       # 1 -> Spot (cheaper, preemptible); data disk survives.
LOCAL_SSD="${LOCAL_SSD:-0}"             # N -> attach N x 375 GB NVMe local SSD (true ephemeral scratch).

args=(
  compute instances create "$VM"
  --project="$PROJECT"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --maintenance-policy=TERMINATE
  --metadata=install-nvidia-driver=True
  --boot-disk-size="${BOOT_DISK_GB}GB"
  --boot-disk-type=pd-ssd
  --create-disk=name="${DATA_DISK_NAME}",size="${DATA_DISK_GB}GB",type="${DATA_DISK_TYPE}",auto-delete=no,device-name="${DATA_DISK_NAME}"
  --scopes=storage-rw
)

if [[ "$SPOT" != "0" ]]; then
  args+=( --provisioning-model=SPOT --instance-termination-action=STOP )
fi

if [[ "$LOCAL_SSD" != "0" ]]; then
  for _ in $(seq 1 "$LOCAL_SSD"); do
    args+=( --local-ssd=interface=NVME )
  done
fi

echo "[gcp_create_vm] project=$PROJECT zone=$ZONE vm=$VM machine=$MACHINE_TYPE"
echo "[gcp_create_vm] image=$IMAGE_PROJECT/$IMAGE_FAMILY  boot=${BOOT_DISK_GB}GB  data=${DATA_DISK_GB}GB ${DATA_DISK_TYPE} (persistent)"
echo "[gcp_create_vm] spot=$SPOT local_ssd=$LOCAL_SSD"
echo "+ gcloud ${args[*]}"
gcloud "${args[@]}"

cat <<EOF

[gcp_create_vm] VM created. Next steps:

  gcloud compute ssh "$VM" --zone "$ZONE"

  # On the VM, format + mount the persistent data disk (ONE TIME, destructive):
  sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard \\
      /dev/disk/by-id/google-${DATA_DISK_NAME}
  sudo mkdir -p /data
  sudo mount -o discard,defaults /dev/disk/by-id/google-${DATA_DISK_NAME} /data
  sudo chown -R "\$USER":"\$USER" /data
  nvidia-smi   # confirm driver + GPU

  # Then install + download + extract (+ optional LMDB build):
  curl -fsSL https://raw.githubusercontent.com/earth-genome/SSL4EO-S12/main/src/benchmark/pretrain_ssl/ssl4eo_dinov3/scripts/gcp_setup.sh | DATA_DIR=/data bash
EOF
