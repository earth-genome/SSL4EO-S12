#!/usr/bin/env bash
# Periodically mirror a local training output directory to a GCS bucket so
# checkpoints/logs survive preemption. Run in the background alongside training:
#
#   bash gcs_sync.sh /workspace/out gs://my-bucket/ssl4eo-dinov3/run1 &
#
# Uses `gsutil rsync` (preinstalled on GCE / Vertex). Interval defaults to 300s.
set -euo pipefail

LOCAL_DIR="${1:?need local output dir}"
GCS_URI="${2:?need gs:// destination}"
INTERVAL="${3:-300}"

echo "[gcs_sync] mirroring $LOCAL_DIR -> $GCS_URI every ${INTERVAL}s"
while true; do
  gsutil -m rsync -r "$LOCAL_DIR" "$GCS_URI" || echo "[gcs_sync] rsync failed (will retry)"
  sleep "$INTERVAL"
done
