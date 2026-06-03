"""DINOv3-compatible dataset for the SSL4EO-S12 Sentinel-2 (L1C, ``s2c``) LMDB.

This wraps the existing 250k-sample LMDB produced for the original DINO v1
pipeline (``datasets/SSL4EO/ssl4eo_dataset_lmdb.py``) and exposes it through the
``torchvision`` ``VisionDataset`` interface that DINOv3's ``make_dataset``
expects, returning a single ``(image, target)`` pair per index.

Design choices:
- The raw ``s2c`` sample is stored as ``uint8`` with shape ``(seasons, 13, H, W)``
  in the original 13-band L1C order
  ``[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]``.
- We drop **B10** (cirrus, index 10) so the model trains on the **12 bands**
  that are actually available in Sentinel-2 **L2A** at inference time. This
  removes the all-zero channel the production pipeline was feeding before.
- We return the *full* season stack ``(seasons, 12, H, W)`` as the "image"; the
  multispectral multi-crop augmentation (:mod:`.augmentations`) is responsible
  for selecting season(s) per crop, mirroring the temporal-as-augmentation
  behaviour of the original ``DataAugmentationDINO_S2``.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np

try:  # torchvision is only available in the training env, not at lint time
    from torchvision.datasets import VisionDataset
except Exception:  # pragma: no cover - allows importing for unit-less checks
    VisionDataset = object  # type: ignore


# 13-band L1C order; index 10 is B10 (cirrus), which is absent from L2A.
S2C_13BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]
B10_INDEX = S2C_13BAND_ORDER.index("B10")  # == 10
# Indices to keep for the 12-band (L2A-compatible) input.
KEEP_12BAND_INDICES = [i for i in range(len(S2C_13BAND_ORDER)) if i != B10_INDEX]
S2C_12BAND_ORDER = [S2C_13BAND_ORDER[i] for i in KEEP_12BAND_INDICES]


class SSL4EOS2Dataset(VisionDataset):
    """SSL4EO-S12 ``s2c`` LMDB exposed for DINOv3 training.

    Args:
        root: Path to the LMDB directory (the ``s2c`` uint8 LMDB).
        dtype: ``"uint8"`` (default, matches the pretraining LMDB) or ``"int16"``
            (raw reflectance / 10000 -> float32).
        drop_b10: If ``True`` (default), drop the B10 cirrus band -> 12 channels.
        length: Optional explicit dataset length (used when the LMDB cannot be
            opened at construction time, e.g. on some networked filesystems).
        transform: Image transform (the DINOv3 multi-crop augmentation).
        target_transform: Target transform (DINOv3 passes ``lambda _: ()``).
        transforms: Combined transform; if given, takes precedence.
    """

    def __init__(
        self,
        root: str,
        dtype: str = "uint8",
        drop_b10: bool = True,
        length: Optional[int] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.lmdb_file = root
        self.dtype = dtype
        self.drop_b10 = drop_b10
        self.env = None

        if length is not None:
            self.length = int(length)
        else:
            import lmdb

            env = lmdb.open(
                self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
            )
            with env.begin(write=False) as txn:
                self.length = txn.stat()["entries"]
            env.close()

    def _init_db(self) -> None:
        import lmdb

        self.env = lmdb.open(
            self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
        )

    def get_image_data(self, index: int) -> np.ndarray:
        if self.env is None:
            self._init_db()
        with self.env.begin(write=False) as txn:
            data = txn.get(str(index).encode())
        s2c_bytes, s2c_shape = pickle.loads(data)
        if self.dtype == "uint8":
            sample = np.frombuffer(s2c_bytes, dtype=np.uint8).reshape(s2c_shape)
        else:
            sample = np.frombuffer(s2c_bytes, dtype=np.int16).reshape(s2c_shape)
            sample = (sample / 10000.0).astype(np.float32)
        # sample: (seasons, 13, H, W) -> keep 12 bands (drop B10)
        if self.drop_b10 and sample.shape[1] == len(S2C_13BAND_ORDER):
            sample = sample[:, KEEP_12BAND_INDICES, :, :]
        # Ensure a writable, contiguous array (np.frombuffer is read-only).
        return np.ascontiguousarray(sample)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.get_image_data(index)
        target: Any = ()
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return self.length
