#!/usr/bin/env python
"""Training entrypoint: DINOv3 SSL pretraining on 12-band SSL4EO-S12 Sentinel-2.

This is the script launched by ``torchrun`` (one process per GPU). It:
1. puts the vendored DINOv3 submodule on ``sys.path``,
2. applies the SSL4EO-S12 adapter patches (12-channel input, dataset, MS aug),
3. hands control to the upstream ``dinov3.train.train.main`` argument parser.

Example (single node, 4 GPUs):

    torchrun --nproc_per_node=4 train_ssl4eo.py \
        --config-file configs/ssl4eo_s2_vits16.yaml \
        --output-dir /path/to/output \
        train.dataset_path=SSL4EOS2:root=/data/ssl4eo_s2c_uint8.lmdb

Any trailing ``key=value`` items override the config (LazyConfig style).
"""

from __future__ import annotations

import os
import sys

# Ensure ``import ssl4eo_dinov3`` works even when this file is launched directly
# by torchrun (i.e. without the package parent already on PYTHONPATH).
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


def main() -> None:
    from ssl4eo_dinov3._paths import ensure_dinov3_on_path

    ensure_dinov3_on_path()

    # Patches must be applied in every process before model/data construction.
    from ssl4eo_dinov3.patches import apply_dinov3_patches

    apply_dinov3_patches()

    from dinov3.train.train import main as dinov3_main

    dinov3_main()


if __name__ == "__main__":
    main()
