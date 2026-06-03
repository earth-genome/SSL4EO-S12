"""Multispectral multi-crop augmentation for DINOv3 on 12-band Sentinel-2.

This is a drop-in replacement for ``dinov3.data.augmentations.DataAugmentationDINO``
that produces the exact same output dictionary structure (so the upstream
``collate_data_and_cast`` and ``SSLMetaArch`` consume it unchanged) but operates
on N-channel (12) imagery instead of RGB PIL images.

Key differences vs the RGB pipeline:
- Input is a NumPy array of either ``(seasons, C, H, W)`` (full SSL4EO season
  stack) or ``(C, H, W)``. When a season stack is given, each of the two global
  crops and the local crops draw an independent season, reproducing the
  temporal-as-augmentation trick from the original ``DataAugmentationDINO_S2``.
- RGB-only photometric ops (``ColorJitter``, ``RandomGrayscale``, ``Solarize``)
  are replaced by channel-count-agnostic multispectral analogues that operate on
  ``uint8`` tensors, keeping behaviour parity with the original solarize/jitter.
- Normalization uses per-band Sentinel-2 statistics (passed as ``mean``/``std``
  of length C) rather than ImageNet RGB constants.
"""

from __future__ import annotations

import logging
import random
from typing import List, Sequence

import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2

logger = logging.getLogger("dinov3")


# ---------------------------------------------------------------------------
# Multispectral photometric ops (operate on uint8 CHW tensors, any #channels)
# ---------------------------------------------------------------------------
class _RandomApply:
    """Apply ``transform`` with probability ``p`` (works with plain callables)."""

    def __init__(self, transform, p: float):
        self.transform = transform
        self.p = p

    def __call__(self, x):
        if random.random() <= self.p:
            return self.transform(x)
        return x


class MSBrightness:
    """Per-image brightness jitter: multiply by a random factor."""

    def __init__(self, factor: float = 0.4):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        f = 1.0 + random.uniform(-self.factor, self.factor)
        out = x.float() * f
        return out.clamp_(0, 255).to(torch.uint8)


class MSContrast:
    """Per-band contrast jitter around the per-band mean."""

    def __init__(self, factor: float = 0.4):
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        f = 1.0 + random.uniform(-self.factor, self.factor)
        xf = x.float()
        mean = xf.mean(dim=(1, 2), keepdim=True)
        out = (xf - mean) * f + mean
        return out.clamp_(0, 255).to(torch.uint8)


class MSToGray:
    """Collapse all bands to their mean and broadcast back (grayscale analogue)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.float().mean(dim=0, keepdim=True)
        return gray.round().clamp_(0, 255).to(torch.uint8).expand_as(x).contiguous()


class MSSolarize:
    """Invert band values above ``threshold`` (multispectral solarize)."""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= self.threshold, 255 - x, x)


class MSColorJitter:
    """Brightness+contrast (p=0.8) then grayscale (p=0.2), like the RGB recipe."""

    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, p_jitter: float = 0.8, p_gray: float = 0.2):
        self.jitter = _RandomApply(
            lambda x: MSContrast(contrast)(MSBrightness(brightness)(x)), p=p_jitter
        )
        self.gray = _RandomApply(MSToGray(), p=p_gray)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.gray(self.jitter(x))


def _gaussian_blur(p: float):
    """uint8/float-safe Gaussian blur applied with probability ``p``."""
    blur = v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
    return _RandomApply(blur, p=p)


class _Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class DataAugmentationDINOv3MS:
    """Multispectral multi-crop transform matching DINOv3's output contract.

    The constructor signature mirrors ``dinov3.data.DataAugmentationDINO`` so it
    can be swapped in via :func:`ssl4eo_dinov3.patches.apply_dinov3_patches`.
    The extra ``season`` argument controls per-crop season sampling.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean: Sequence[float] = (0.5,) * 12,
        std: Sequence[float] = (0.25,) * 12,
        season: str = "augment",
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.season = season
        self.mean = list(mean)
        self.std = list(std)

        logger.info("###################################")
        logger.info("Using MULTISPECTRAL (12-band) DINOv3 data augmentation:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"season sampling: {season}")
        logger.info(f"n bands (mean/std): {len(self.mean)}/{len(self.std)}")
        logger.info("###################################")

        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size, scale=global_crops_scale, interpolation=v2.InterpolationMode.BICUBIC
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = v2.Resize(global_crops_size, interpolation=v2.InterpolationMode.BICUBIC)
            else:
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size, interpolation=v2.InterpolationMode.BICUBIC
                )
            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size, interpolation=v2.InterpolationMode.BICUBIC
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=v2.InterpolationMode.BICUBIC
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        color_jittering = MSColorJitter()
        global_transfo1_extra = _gaussian_blur(p=1.0)
        global_transfo2_extra = _Compose([_gaussian_blur(p=0.1), _RandomApply(MSSolarize(threshold=128), p=0.2)])
        local_transfo_extra = _gaussian_blur(p=0.5)

        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = _Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = _Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = _Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = _Compose([resize_global, color_jittering, global_transfo1_extra, self.normalize])
            self.global_transfo2 = _Compose([resize_global, color_jittering, global_transfo2_extra, self.normalize])
            self.local_transfo = _Compose([color_jittering, local_transfo_extra, self.normalize])

    # -- season handling ----------------------------------------------------
    def _to_image(self, arr: np.ndarray) -> tv_tensors.Image:
        """Convert a (C, H, W) uint8/float ndarray to a tv_tensors.Image."""
        t = torch.from_numpy(np.ascontiguousarray(arr))
        return tv_tensors.Image(t)

    def _pick_seasons(self, n_seasons: int):
        """Return (g1, g2, local) season indices according to ``self.season``."""
        if self.season == "augment" and n_seasons > 1:
            g1, g2, loc = (random.randint(0, n_seasons - 1) for _ in range(3))
            return g1, g2, loc
        if self.season == "random" and n_seasons > 1:
            s = random.randint(0, n_seasons - 1)
            return s, s, s
        return 0, 0, 0  # "fixed" or single-season input

    def _season_image(self, image: np.ndarray, season_idx: int) -> tv_tensors.Image:
        if image.ndim == 4:  # (seasons, C, H, W)
            return self._to_image(image[season_idx])
        return self._to_image(image)  # already (C, H, W)

    # -- main entry ----------------------------------------------------------
    def __call__(self, image: np.ndarray):
        output = {"weak_flag": True}
        n_seasons = image.shape[0] if image.ndim == 4 else 1
        s_g1, s_g2, s_loc = self._pick_seasons(n_seasons)

        img_g1 = self._season_image(image, s_g1)
        img_g2 = self._season_image(image, s_g2)
        img_loc = self._season_image(image, s_loc)

        if self.share_color_jitter:
            img_g1 = self.color_jittering(img_g1)
            img_g2 = self.color_jittering(img_g2)
            img_loc = self.color_jittering(img_loc)

        # global crops
        im1_base = self.geometric_augmentation_global(img_g1)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(img_g2)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [self.normalize(im1_base), self.normalize(im2_base)]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]
            local_crops: List = []
            offsets = []
            gs, ls = self.global_crops_size, self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))
            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            output["local_crops"] = [
                self.local_transfo(self.geometric_augmentation_local(img_loc))
                for _ in range(self.local_crops_number)
            ]
            output["offsets"] = ()

        return output
