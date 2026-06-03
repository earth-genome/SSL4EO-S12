#!/usr/bin/env python
"""Compute per-band mean/std for the 12-band SSL4EO-S12 s2c LMDB.

The values are computed in the **[0, 1] domain** (uint8 / 255), i.e. the same
domain the DINOv3 augmentation normalizes in (``ToDtype(float32, scale=True)``).
Paste the printed arrays into ``crops.rgb_mean`` / ``crops.rgb_std`` of the
training config for accurate per-band standardization.

Usage:
    python compute_band_stats.py --lmdb /data/ssl4eo_s2c_uint8.lmdb --samples 20000
"""

from __future__ import annotations

import argparse

import numpy as np

from ssl4eo_dinov3.datasets import S2C_12BAND_ORDER, SSL4EOS2Dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb", required=True, help="Path to the s2c uint8 LMDB")
    ap.add_argument("--samples", type=int, default=20000, help="Number of LMDB entries to sample")
    ap.add_argument("--dtype", default="uint8", choices=["uint8", "int16"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = SSL4EOS2Dataset(root=args.lmdb, dtype=args.dtype, drop_b10=True)
    n = min(args.samples, len(ds))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=n, replace=False)

    C = len(S2C_12BAND_ORDER)
    # Accumulate per-band sum and sum-of-squares in float64 (data in [0,1] so
    # no overflow concerns); mean/std derived at the end.
    count = 0
    s1 = np.zeros(C, dtype=np.float64)
    s2 = np.zeros(C, dtype=np.float64)

    for k, idx in enumerate(indices):
        sample = ds.get_image_data(int(idx))  # (seasons, C, H, W)
        x = sample.astype(np.float64).transpose(1, 0, 2, 3).reshape(C, -1)  # (C, npix)
        if args.dtype == "uint8":
            x = x / 255.0
        s1 += x.sum(axis=1)
        s2 += (x * x).sum(axis=1)
        count += x.shape[1]
        if (k + 1) % 1000 == 0:
            print(f"  processed {k + 1}/{n} samples...", flush=True)

    mean = s1 / max(count, 1)
    std = np.sqrt(np.maximum(s2 / max(count, 1) - mean * mean, 0.0))
    print("\nBands:", S2C_12BAND_ORDER)
    print("rgb_mean:", [round(float(v), 6) for v in mean])
    print("rgb_std: ", [round(float(v), 6) for v in std])


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
