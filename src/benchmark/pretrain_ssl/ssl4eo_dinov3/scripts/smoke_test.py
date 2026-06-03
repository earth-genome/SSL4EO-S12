#!/usr/bin/env python
"""CPU smoke test for the SSL4EO-S12 x DINOv3 adapter (no GPU required).

Validates, end-to-end on a tiny synthetic LMDB:
  1. the dataset-string registration (``SSL4EOS2:root=...``),
  2. LMDB decoding + B10 drop (13 -> 12 bands),
  3. the multispectral multi-crop augmentation output contract,
  4. ``collate_data_and_cast`` over a batch (channel-agnostic stacking),
  5. the ``in_chans`` injection into the ViT patch embedding.

Run (inside the DINOv3 conda env, after `git submodule update --init`):
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ssl4eo_dinov3._paths import ensure_dinov3_on_path  # noqa: E402

ensure_dinov3_on_path()

import torch  # noqa: E402

from ssl4eo_dinov3 import apply_dinov3_patches  # noqa: E402
from ssl4eo_dinov3.augmentations import DataAugmentationDINOv3MS  # noqa: E402
from ssl4eo_dinov3.datasets import SSL4EOS2Dataset  # noqa: E402


def _make_synthetic_lmdb(path: str, n: int = 8, seasons: int = 4, bands: int = 13, hw: int = 144) -> None:
    import lmdb

    env = lmdb.open(path, map_size=1 << 30)
    with env.begin(write=True) as txn:
        for i in range(n):
            arr = (np.random.rand(seasons, bands, hw, hw) * 255).astype(np.uint8)
            shape = arr.shape
            txn.put(str(i).encode(), pickle.dumps((arr.tobytes(), shape)))
    env.close()


def main() -> int:
    apply_dinov3_patches()

    # 1. dataset-string registration
    import dinov3.data.loaders as loaders

    cls, kwargs = loaders._parse_dataset_str("SSL4EOS2:root=/tmp/x:dtype=uint8")
    assert cls is SSL4EOS2Dataset and kwargs == {"root": "/tmp/x", "dtype": "uint8"}, (cls, kwargs)
    print("[ok] dataset-string registration")

    with tempfile.TemporaryDirectory() as tmp:
        lmdb_path = os.path.join(tmp, "syn.lmdb")
        _make_synthetic_lmdb(lmdb_path)

        # 2. dataset + B10 drop
        ds = SSL4EOS2Dataset(root=lmdb_path, dtype="uint8")
        assert len(ds) == 8, len(ds)
        sample = ds.get_image_data(0)
        assert sample.shape == (4, 12, 144, 144), sample.shape
        print("[ok] LMDB decode + B10 drop -> (4, 12, 144, 144)")

        # 3. multispectral multi-crop augmentation
        aug = DataAugmentationDINOv3MS(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
            mean=[0.5] * 12,
            std=[0.25] * 12,
            season="augment",
        )
        out = aug(sample)
        for key in ("global_crops", "global_crops_teacher", "local_crops", "offsets", "weak_flag"):
            assert key in out, key
        assert len(out["global_crops"]) == 2 and len(out["local_crops"]) == 8
        assert tuple(out["global_crops"][0].shape) == (12, 224, 224), out["global_crops"][0].shape
        assert tuple(out["local_crops"][0].shape) == (12, 96, 96), out["local_crops"][0].shape
        assert out["global_crops"][0].dtype == torch.float32
        print("[ok] multispectral multi-crop augmentation -> 12-ch global/local crops")

        # 4. collate
        from dinov3.data import MaskingGenerator, collate_data_and_cast
        from functools import partial

        img_size, patch_size = 224, 16
        n_tokens = (img_size // patch_size) ** 2
        mask_gen = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * (img_size // patch_size) ** 2,
        )
        collate = partial(
            collate_data_and_cast,
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=0.5,
            dtype=torch.float32,
            n_tokens=n_tokens,
            mask_generator=mask_gen,
        )
        batch = collate([(aug(ds.get_image_data(i)), ()) for i in range(2)])
        assert batch["collated_global_crops"].shape == (4, 12, 224, 224), batch["collated_global_crops"].shape
        assert batch["collated_local_crops"].shape == (16, 12, 96, 96), batch["collated_local_crops"].shape
        print("[ok] collate_data_and_cast -> global", tuple(batch["collated_global_crops"].shape))

    # 5. in_chans injection into the ViT patch embedding
    from ssl4eo_dinov3 import patches as P

    P._IN_CHANS["value"] = 12
    from dinov3.models import vision_transformer as vits

    model = vits.vit_small(patch_size=16)  # in_chans should be injected = 12
    in_ch = model.patch_embed.proj.weight.shape[1]
    assert in_ch == 12, f"expected 12 input channels, got {in_ch}"
    print("[ok] in_chans injection -> patch_embed conv has 12 input channels")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
