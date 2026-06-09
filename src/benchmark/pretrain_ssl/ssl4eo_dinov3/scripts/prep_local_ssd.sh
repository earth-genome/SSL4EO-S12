#!/usr/bin/env bash
# Stage the SSL4EO LMDB onto local ephemeral NVMe SSD scratch.
#
# WHY: /data is a GCP Persistent Disk (network-backed). DINOv3 training does
# shuffled random access over a ~582GB LMDB; on PD that is IOPS-bound, so the
# LMDB was opened with readahead=True to coalesce faults -- which then over-read
# adjacent pages into page cache and OOM-killed the run on a 167GB host.
#
# The fix is local NVMe: random reads are cheap, so we run with readahead OFF
# (SSL4EO_LMDB_READAHEAD unset) and page-cache use stays bounded. This script
# RAID0s the local SSD(s), mounts them at $SCRATCH, and copies the LMDB across.
#
# Local SSDs are EPHEMERAL: wiped on every VM stop / Spot preemption. So:
#   - keep /data (PD) as the source of truth for the LMDB and for OUTPUT_DIR
#     (checkpoints), and
#   - re-run this script after every boot before launching training.
#
# Usage:
#   bash prep_local_ssd.sh                      # copy default LMDB -> /scratch
#   SRC_LMDB=/data/foo.lmdb SCRATCH=/scratch bash prep_local_ssd.sh
#
# Then launch training against the scratch copy (output stays on PD):
#   DATASET_SIZE=172348 NUM_GPUS=2 \
#     bash train_gcp.sh /scratch/ssl4eo_s2c_uint8.lmdb /data/out/vits16
set -euo pipefail

SRC_LMDB="${SRC_LMDB:-/data/ssl4eo_s2c_uint8.lmdb}"
SCRATCH="${SCRATCH:-/scratch}"
RAID_DEV="${RAID_DEV:-/dev/md0}"

[[ -d "$SRC_LMDB" ]] || { echo "[prep] source LMDB not found: $SRC_LMDB" >&2; exit 1; }
DEST_LMDB="$SCRATCH/$(basename "$SRC_LMDB")"

# Discover local NVMe scratch devices. GCP exposes them as
# /dev/disk/by-id/google-local-nvme-ssd-* (or google-local-ssd-*).
mapfile -t SSDS < <(ls -1 /dev/disk/by-id/google-local-* 2>/dev/null | grep -v -- '-part[0-9]' || true)
N="${#SSDS[@]}"
if [[ "$N" -eq 0 ]]; then
  echo "[prep] No local NVMe SSDs found. Did you create the VM with --local-ssd?" >&2
  echo "[prep] (Local SSDs can only be attached at instance-create time.)" >&2
  exit 1
fi
echo "[prep] found $N local SSD(s): ${SSDS[*]}"

# Need enough raw capacity for the source LMDB (each local SSD is 375GB).
SRC_BYTES="$(du -sb "$SRC_LMDB" | cut -f1)"
RAW_BYTES=$(( N * 375 * 1000 * 1000 * 1000 ))
if (( RAW_BYTES < SRC_BYTES )); then
  echo "[prep] local SSD capacity (~$((RAW_BYTES/1000/1000/1000))GB across $N disk(s)) < LMDB ($((SRC_BYTES/1000/1000/1000))GB)." >&2
  echo "[prep] Recreate the VM with more local SSDs (LOCAL_SSD=$(( (SRC_BYTES/375/1000/1000/1000) + 1 )) or higher)." >&2
  exit 1
fi

# Idempotent: if scratch is already mounted with the LMDB, skip straight to verify.
if mountpoint -q "$SCRATCH" && [[ -d "$DEST_LMDB/data.mdb" || -f "$DEST_LMDB/data.mdb" ]]; then
  echo "[prep] $DEST_LMDB already present on mounted scratch; skipping format/copy."
else
  # Assemble the block device: RAID0 if >1 SSD, else use the single device.
  if [[ "$N" -gt 1 ]]; then
    if [[ ! -b "$RAID_DEV" ]]; then
      echo "[prep] creating RAID0 ($RAID_DEV) over $N devices"
      sudo mdadm --create "$RAID_DEV" --level=0 --raid-devices="$N" "${SSDS[@]}" --run
    fi
    BLOCK="$RAID_DEV"
  else
    BLOCK="${SSDS[0]}"
  fi

  # Format + mount (destructive on the ephemeral scratch device only).
  echo "[prep] mkfs.ext4 on $BLOCK and mount at $SCRATCH"
  sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$BLOCK"
  sudo mkdir -p "$SCRATCH"
  sudo mount -o discard,defaults "$BLOCK" "$SCRATCH"
  sudo chown -R "$USER":"$USER" "$SCRATCH"

  echo "[prep] copying $SRC_LMDB -> $DEST_LMDB (~$((SRC_BYTES/1000/1000/1000))GB)"
  # cp is fine for a one-shot bulk copy; rsync gives a resumable progress view.
  rsync -ah --info=progress2 "$SRC_LMDB/" "$DEST_LMDB/"
fi

# Verify size matches so a truncated copy doesn't silently corrupt training.
DST_BYTES="$(du -sb "$DEST_LMDB" | cut -f1)"
if (( DST_BYTES < SRC_BYTES )); then
  echo "[prep] WARNING: copied size ($DST_BYTES) < source ($SRC_BYTES). Copy may be incomplete." >&2
  exit 1
fi
echo "[prep] done. Train against: $DEST_LMDB"
echo "[prep] e.g.: DATASET_SIZE=172348 NUM_GPUS=2 bash $(dirname "${BASH_SOURCE[0]}")/train_gcp.sh $DEST_LMDB /data/out/vits16"
