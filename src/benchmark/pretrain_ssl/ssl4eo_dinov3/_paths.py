"""Locate the vendored DINOv3 submodule and put it on ``sys.path``.

The submodule lives at ``<pretrain_ssl>/dinov3`` (a clone of
facebookresearch/dinov3); its importable package is ``<pretrain_ssl>/dinov3/dinov3``.
Importing this module ensures ``import dinov3`` resolves to the pinned submodule.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# <pretrain_ssl>/ssl4eo_dinov3/_paths.py -> <pretrain_ssl>
PRETRAIN_SSL_DIR = os.path.dirname(_THIS_DIR)
DINOV3_ROOT = os.path.join(PRETRAIN_SSL_DIR, "dinov3")


def ensure_dinov3_on_path() -> str:
    """Insert the DINOv3 submodule root on ``sys.path`` and return it."""
    if not os.path.isdir(os.path.join(DINOV3_ROOT, "dinov3")):
        raise RuntimeError(
            f"DINOv3 submodule not found at {DINOV3_ROOT!r}. "
            "Run: git submodule update --init src/benchmark/pretrain_ssl/dinov3"
        )
    if DINOV3_ROOT not in sys.path:
        sys.path.insert(0, DINOV3_ROOT)
    return DINOV3_ROOT
