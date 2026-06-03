"""SSL4EO-S12 x DINOv3 adapter package.

Built with DINOv3 (https://github.com/facebookresearch/dinov3). This package
adapts the official DINOv3 self-supervised training code to 12-band Sentinel-2
imagery (SSL4EO-S12) without modifying the upstream submodule. It registers a
custom dataset, a multispectral multi-crop augmentation, and injects the input
channel count into the ViT patch embedding via monkeypatching.

See ``patches.apply_dinov3_patches`` for the integration seams.
"""

from .patches import apply_dinov3_patches  # noqa: F401

__all__ = ["apply_dinov3_patches"]
