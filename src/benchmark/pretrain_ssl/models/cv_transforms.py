"""Drop-in replacement for cvtorchvision.cvtransforms.

Only the subset actually used in this project is implemented.  All transforms
operate on numpy HWC uint8 arrays (arbitrary channel count) unless noted.
"""

import math
import random

import cv2
import numpy as np
import torch

_INTERP = {
    "NEAREST": cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "BICUBIC": cv2.INTER_CUBIC,
    "LANCZOS": cv2.INTER_LANCZOS4,
}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    """HWC uint8 ndarray -> CHW float32 tensor in [0, 1]."""

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            return torch.from_numpy(pic.transpose((2, 0, 1)).copy()).float().div_(255)
        raise TypeError(f"Expected np.ndarray, got {type(pic)}")


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return np.ascontiguousarray(img[:, ::-1])
        return img


class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            for t in self.transforms:
                img = t(img)
        return img


class RandomResizedCrop:
    """Random crop + resize, matching torchvision semantics on numpy HWC arrays."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation="BILINEAR"):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = _INTERP.get(interpolation.upper(), cv2.INTER_LINEAR)

    @staticmethod
    def _get_params(h, w, scale, ratio):
        area = h * w
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(10):
            target_area = random.uniform(scale[0], scale[1]) * area
            aspect = math.exp(random.uniform(*log_ratio))
            nw = int(round(math.sqrt(target_area * aspect)))
            nh = int(round(math.sqrt(target_area / aspect)))
            if 0 < nw <= w and 0 < nh <= h:
                return random.randint(0, h - nh), random.randint(0, w - nw), nh, nw

        # fallback: central crop respecting ratio bounds
        in_ratio = w / h
        if in_ratio < ratio[0]:
            nw, nh = w, int(round(w / ratio[0]))
        elif in_ratio > ratio[1]:
            nh, nw = h, int(round(h * ratio[1]))
        else:
            nw, nh = w, h
        return (h - nh) // 2, (w - nw) // 2, nh, nw

    def __call__(self, img):
        i, j, ch, cw = self._get_params(img.shape[0], img.shape[1],
                                         self.scale, self.ratio)
        crop = img[i:i + ch, j:j + cw]
        return cv2.resize(crop, (self.size[1], self.size[0]),
                          interpolation=self.interpolation)
