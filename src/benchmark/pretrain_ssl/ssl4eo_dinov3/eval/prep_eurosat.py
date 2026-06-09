#!/usr/bin/env python
"""Prepare EuroSAT-MS as .npy tiles for DINOv3 certification.

Common steps regardless of encoding:
  * read the 13-band EuroSAT-MS GeoTIFF (band order B01..B12,B8A),
  * reorder B8A from last to index 8  -> SSL4EO L1C order
        [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12],
  * stratified 80/20 train/val split with random_state=42,
  * write uint8 tiles (extract_embeddings.py divides by 255 and drops B10).

``--encoding`` selects the uint8 scheme, which MUST match how the evaluated
model's training LMDB was built:

  * ``reflectance`` (default): ``clip(reflectance/10000, 0, 1) * 255`` -- matches
    the SSL4EO ``s2c`` uint8 LMDB the 12-band DINOv3 model was trained on (its
    config ``rgb_mean`` ~0.04-0.26 in the /255 domain confirms this scheme).
  * ``2sigma``: per-band 2-sigma normalization with the EuroSAT BAND_STATS --
    mirrors the ORIGINAL SSL4EO benchmark (comparable to the published DINO
    ViT-S/16 99.0 number, but only correct for a model trained on 2-sigma data;
    it centers values ~0.5 and will mis-normalize a reflectance-trained model).

Output layout (consumed by run_certification.sh):
    <out>/train/<idx>.npy   (13, 64, 64) uint8     <out>/train_y.npy  (Ntr,) int
    <out>/val/<idx>.npy                            <out>/val_y.npy    (Nva,) int

Tiles are zero-padded so sorted filename order == label-array order (the order
extract_embeddings.py emits embeddings in).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

# EuroSAT-MS GeoTIFF band order (B8A last); matches the original loader.
ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
             "B09", "B10", "B11", "B12", "B8A"]

# SSL4EO per-band stats (uint8 2-sigma scheme), copied verbatim from
# transfer_classification/datasets/EuroSat/eurosat_dataset.py.
BAND_STATS = {
    "mean": {"B01": 1353.72696296, "B02": 1117.20222222, "B03": 1041.8842963,
             "B04": 946.554, "B05": 1199.18896296, "B06": 2003.00696296,
             "B07": 2374.00874074, "B08": 2301.22014815, "B8A": 2599.78311111,
             "B09": 732.18207407, "B10": 12.09952894, "B11": 1820.69659259,
             "B12": 1118.20259259},
    "std": {"B01": 897.27143653, "B02": 736.01759721, "B03": 684.77615743,
            "B04": 620.02902871, "B05": 791.86263829, "B06": 1341.28018273,
            "B07": 1595.39989386, "B08": 1545.52915718, "B8A": 1750.12066835,
            "B09": 475.11595216, "B10": 98.26600935, "B11": 1216.48651476,
            "B12": 736.6981037},
}


def _encode_band(ch, band, encoding):
    """Single band reflectance -> uint8 under the chosen scheme."""
    if encoding == "reflectance":
        return np.clip(ch / 10000.0, 0.0, 1.0) * 255.0
    # 2-sigma per-band (original benchmark)
    mean, std = BAND_STATS["mean"][band], BAND_STATS["std"][band]
    lo, hi = mean - 2 * std, mean + 2 * std
    return np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)


def load_tile(path: str, encoding: str) -> np.ndarray:
    """GeoTIFF -> (13, H, W) uint8 in SSL4EO L1C band order."""
    with rasterio.open(path) as f:
        arr = f.read().astype(np.float32)  # (13, H, W), ALL_BANDS order
    channels = []
    for i, b in enumerate(ALL_BANDS):
        ch = _encode_band(arr[i], b, encoding).round().astype(np.uint8)
        if b == "B8A":          # EuroSAT puts B8A last; SSL4EO wants it at idx 8
            channels.insert(8, ch)
        else:
            channels.append(ch)
    return np.stack(channels, axis=0)  # (13, H, W)


def find_samples(root: Path):
    """Replicate EurosatDataset: sorted classes, sorted os.walk, folder labels."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples, targets = [], []
    exts = (".tif", ".tiff", ".jpg", ".jpeg", ".png")
    for froot, _, fnames in sorted(os.walk(root, followlinks=True)):
        for fname in sorted(fnames):
            if fname.lower().endswith(exts):
                path = os.path.join(froot, fname)
                samples.append(path)
                targets.append(class_to_idx[Path(path).parts[-2]])
    return samples, np.array(targets), classes


def write_split(samples, targets, indices, out_dir: Path, pad: int, encoding: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ys = []
    for i, idx in enumerate(indices):
        tile = load_tile(samples[idx], encoding)
        np.save(out_dir / f"{i:0{pad}d}.npy", tile)
        ys.append(targets[idx])
        if (i + 1) % 2000 == 0:
            print(f"  {out_dir.name}: {i + 1}/{len(indices)}", flush=True)
    return np.array(ys, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="EuroSAT-MS dir (class subfolders of .tif)")
    ap.add_argument("--out", required=True, help="output dir for tiles + labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoding", choices=["reflectance", "2sigma"], default="reflectance",
                    help="uint8 scheme; must match the model's training LMDB "
                         "(reflectance for the 12-band DINOv3 s2c model)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    samples, targets, classes = find_samples(root)
    print(f"[prep] {len(samples)} tiles, {len(classes)} classes: {classes}")

    idx = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        idx, train_size=0.8, stratify=targets, random_state=args.seed
    )
    pad = len(str(len(samples)))
    print(f"[prep] split: {len(train_idx)} train / {len(val_idx)} val "
          f"(seed={args.seed}, encoding={args.encoding})")

    train_y = write_split(samples, targets, train_idx, out / "train", pad, args.encoding)
    val_y = write_split(samples, targets, val_idx, out / "val", pad, args.encoding)
    np.save(out / "train_y.npy", train_y)
    np.save(out / "val_y.npy", val_y)
    print(f"[prep] done -> {out}  (train_y {train_y.shape}, val_y {val_y.shape})")


if __name__ == "__main__":
    main()
