#!/usr/bin/env python
"""Extract DINOv3 CLS embeddings for 12-band Sentinel-2 tiles.

Used both for (a) offline quality certification and (b) production embedding
extraction. Input sources:
  - ``--source npy   --input <dir>``  : directory of ``.npy`` tiles, shape
        ``(C, H, W)`` or ``(seasons, C, H, W)`` with C in {12, 13}.
  - ``--source lmdb  --input <lmdb>`` : an SSL4EO ``s2c`` uint8 LMDB (a season is
        selected via ``--season-index``).
  - ``--source tiff  --input <dir>``  : directory of GeoTIFFs (needs rasterio).

13-band inputs have B10 (cirrus, index 10) dropped to match the 12-band model.

IMPORTANT — normalization must match training. The model was trained on the
uint8 SSL4EO encoding scaled to [0, 1] then standardized by the per-band stats
in the config. For production L2A tiles, encode them with the *same* uint8
2-sigma scheme used to build the LMDB (see the repo's dataset builder), or pass
already-[0,1] float tiles with ``--scale 1.0``.

Output: ``<out>.npy`` (float32, ``(N, D)``) and ``<out>.ids.txt`` (one id/line).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssl4eo_dinov3.datasets import KEEP_12BAND_INDICES  # noqa: E402
from ssl4eo_dinov3.eval.build_backbone import build_backbone  # noqa: E402


def _list_npy(d: str) -> List[str]:
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npy"))


def _drop_b10(arr: np.ndarray) -> np.ndarray:
    """arr is (C, H, W); drop B10 if 13-band."""
    if arr.shape[0] == 13:
        return arr[KEEP_12BAND_INDICES, :, :]
    return arr


def _iter_source(args) -> Tuple[List[str], List[np.ndarray]]:
    ids, tiles = [], []
    if args.source == "npy":
        for path in _list_npy(args.input):
            arr = np.load(path)
            if arr.ndim == 4:  # (seasons, C, H, W)
                arr = arr[args.season_index]
            tiles.append(_drop_b10(arr))
            ids.append(os.path.splitext(os.path.basename(path))[0])
    elif args.source == "lmdb":
        from ssl4eo_dinov3.datasets import SSL4EOS2Dataset

        ds = SSL4EOS2Dataset(root=args.input, dtype=args.input_dtype, drop_b10=True)
        n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
        for i in range(n):
            sample = ds.get_image_data(i)  # (seasons, 12, H, W)
            tiles.append(sample[args.season_index])
            ids.append(str(i))
    elif args.source == "tiff":
        import rasterio

        for path in sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith((".tif", ".tiff"))
        ):
            with rasterio.open(path) as src:
                arr = src.read()  # (bands, H, W)
            tiles.append(_drop_b10(arr))
            ids.append(os.path.splitext(os.path.basename(path))[0])
    else:
        raise ValueError(args.source)
    return ids, tiles


def _preprocess(tiles: List[np.ndarray], img_size: int, scale: float, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    out = []
    for arr in tiles:
        x = torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)  # (1, C, H, W)
        if x.shape[-1] != img_size or x.shape[-2] != img_size:
            x = F.interpolate(x, size=(img_size, img_size), mode="bicubic", align_corners=False)
        x = x / scale
        x = (x - mean_t) / std_t
        out.append(x)
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Training config yaml (for arch + normalization)")
    ap.add_argument("--checkpoint", required=True, help="Teacher checkpoint .pth")
    ap.add_argument("--source", choices=["npy", "lmdb", "tiff"], required=True)
    ap.add_argument("--input", required=True, help="Input dir or LMDB path")
    ap.add_argument("--out", required=True, help="Output prefix (writes <out>.npy and <out>.ids.txt)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--scale", type=float, default=255.0, help="Divide inputs by this (255 for uint8, 1.0 for [0,1] float)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--season-index", type=int, default=0)
    ap.add_argument("--input-dtype", default="uint8", choices=["uint8", "int16"])
    ap.add_argument("--limit", type=int, default=0, help="Max samples (lmdb source); 0 = all")
    args = ap.parse_args()

    _, embed_fn, (mean, std) = build_backbone(args.config, args.checkpoint, device=args.device)
    ids, tiles = _iter_source(args)
    if not tiles:
        raise SystemExit(f"No tiles found at {args.input}")
    print(f"[extract] {len(tiles)} tiles; embedding on {args.device}")

    x = _preprocess(tiles, args.img_size, args.scale, mean, std)
    feats = []
    for i in range(0, x.shape[0], args.batch_size):
        feats.append(embed_fn(x[i : i + args.batch_size]))
    embeddings = torch.cat(feats, dim=0).numpy().astype(np.float32)

    np.save(args.out + ".npy", embeddings)
    with open(args.out + ".ids.txt", "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"[extract] wrote {embeddings.shape} -> {args.out}.npy")


if __name__ == "__main__":
    main()
