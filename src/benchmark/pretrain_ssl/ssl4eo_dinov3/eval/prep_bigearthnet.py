#!/usr/bin/env python
"""Prepare BigEarthNet-S2 v1.0 as .npy tiles for DINOv3 certification (multilabel).

BigEarthNet-S2 ships one folder per patch, each holding 12 single-band GeoTIFFs
(``<patch>_B01.tif`` ... ``_B12.tif``; **no B10**, B8A present) at mixed native
resolutions (10 m -> 120x120, 20 m -> 60x60, 60 m -> 20x20) plus a
``<patch>_labels_metadata.json`` with the original 43-class CORINE labels.

The 12 bands BigEarthNet provides --
    [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
-- are *exactly* the 12-band order the DINOv3 model expects (the SSL4EO s2c
13-band order with B10/cirrus dropped), so tiles are written 12-band and
``extract_embeddings.py`` consumes them as-is (no B10 to drop).

What this does:
  * stream the 70 GB ``.tar.gz`` ONCE (no full disk extraction); for each patch
    in the subsampled split, read its band tiffs straight from the archive,
  * resample every band to ``--img-size`` (bicubic) so they stack,
  * reflectance-encode to uint8: ``clip(reflectance/10000, 0, 1) * 255`` -- the
    SAME scheme the model's training LMDB used (NOT the repo's 2-sigma BAND_STATS
    normalize, which is for the original SSL4EO models),
  * map the 43 CORINE labels -> 19-class multi-hot (``use_new_labels=True``),
    reusing GROUP_LABELS/NEW_LABELS from the repo's BigEarthNet loader.

Splits come from the official TU Berlin 19-class CSVs (column 0 = S2 patch name);
they already exclude cloud/snow/zero-label patches. Subsampled (seeded) because a
linear-probe certification doesn't need all 269k/123k patches.

Output layout (consumed by run_certification.sh / linear_probe.py --task multilabel):
    <out>/train/<idx>.npy  (12, S, S) uint8     <out>/train_y.npy  (Ntr, 19) int
    <out>/val/<idx>.npy                          <out>/val_y.npy    (Nva, 19) int
Tiles are zero-padded so sorted filename order == label-array order.
"""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

# Band order + 43->19 label mapping, copied verbatim from the repo's BigEarthNet
# loader (transfer_classification/datasets/BigEarthNet/bigearthnet_dataset_seco.py).
# Inlined rather than imported so this script doesn't pull in cv2/torchvision.
# These 12 bands ARE the model's 12-band order (s2c 13-band minus B10/cirrus).
ALL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

NEW_LABELS = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest",
    "Mixed forest", "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation", "Transitional woodland/shrub",
    "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands",
    "Inland waters", "Marine waters",
]

GROUP_LABELS = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Annual crops associated with permanent crops": "Permanent crops",
    "Natural grassland": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Inland marshes": "Inland wetlands",
    "Peatbogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Water bodies": "Inland waters",
    "Water courses": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
}

TAR_SUBDIR = "BigEarthNet-v1.0"  # top-level dir inside the archive
N_CLASSES = len(NEW_LABELS)  # 19


def multihot_new(labels) -> np.ndarray:
    """43 CORINE label strings -> (19,) multi-hot, matching Bigearthnet.get_multihot_new."""
    t = np.zeros((N_CLASSES,), dtype=np.int64)
    new_set = set(NEW_LABELS)
    for label in labels:
        if label in GROUP_LABELS:
            t[NEW_LABELS.index(GROUP_LABELS[label])] = 1
        elif label in new_set:
            t[NEW_LABELS.index(label)] = 1
    return t


def read_patch_names(csv_path: str) -> list[str]:
    names = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


def subsample(names: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if n <= 0 or n >= len(names):
        return names
    idx = rng.choice(len(names), size=n, replace=False)
    return [names[i] for i in sorted(idx)]


def encode_band(arr_2d: np.ndarray) -> np.ndarray:
    """Single-band reflectance -> uint8 via clip(reflectance/10000,0,1)*255."""
    return (np.clip(arr_2d.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def tif_to_band(tif_bytes: bytes, size: int) -> np.ndarray:
    """Decode a single-band GeoTIFF (bytes) and bicubic-resample to (size, size)."""
    with MemoryFile(tif_bytes) as mem, mem.open() as src:
        arr = src.read(1, out_shape=(size, size), resampling=Resampling.cubic)
    return encode_band(arr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tar", required=True, help="BigEarthNet-S2-v1.0.tar.gz")
    ap.add_argument("--out", required=True, help="output dir (train/ val/ *_y.npy)")
    ap.add_argument("--train-csv", required=True, help="TU Berlin 19-class train split CSV")
    ap.add_argument("--val-csv", required=True, help="TU Berlin 19-class val split CSV")
    ap.add_argument("--n-train", type=int, default=30000, help="subsample size (<=0 = all)")
    ap.add_argument("--n-val", type=int, default=10000, help="subsample size (<=0 = all)")
    ap.add_argument("--img-size", type=int, default=120, help="resample every band to SxS")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    train_names = subsample(read_patch_names(args.train_csv), args.n_train, rng)
    val_names = subsample(read_patch_names(args.val_csv), args.n_val, rng)
    print(f"[prep] subsampled train={len(train_names)}  val={len(val_names)}  "
          f"(seed={args.seed}, img-size={args.img_size}, encoding=reflectance)", flush=True)

    # patch name -> (split, output_index); zero-padded for stable sorted order.
    pad_tr, pad_va = len(str(len(train_names))), len(str(len(val_names)))
    plan: dict[str, tuple[str, int, int]] = {}
    for i, name in enumerate(train_names):
        plan[name] = ("train", i, pad_tr)
    for i, name in enumerate(val_names):
        plan[name] = ("val", i, pad_va)  # train/val patch sets are disjoint

    train_y = np.zeros((len(train_names), N_CLASSES), dtype=np.int64)
    val_y = np.zeros((len(val_names), N_CLASSES), dtype=np.int64)
    ys = {"train": train_y, "val": val_y}

    # Stream the archive once, buffering one patch's files at a time.
    needed = set(plan)
    buf: dict[str, dict[str, bytes]] = {}
    done = 0
    total = len(needed)
    band_suffixes = {f"_{b}.tif": b for b in ALL_BANDS}

    def flush_patch(patch: str) -> None:
        nonlocal done
        files = buf.pop(patch)
        split, idx, pad = plan[patch]
        bands = [tif_to_band(files[b], args.img_size) for b in ALL_BANDS]
        tile = np.stack(bands, axis=0)  # (12, S, S) uint8
        np.save(out / split / f"{idx:0{pad}d}.npy", tile)
        ys[split][idx] = multihot_new(json.loads(files["json"])["labels"])
        done += 1
        if done % 2000 == 0:
            print(f"  processed {done}/{total} patches", flush=True)

    with tarfile.open(args.tar, mode="r|gz") as tar:  # streaming mode
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            patch = parts[1]
            if patch not in needed:
                continue
            fname = parts[-1]
            key = None
            if fname.endswith("_labels_metadata.json"):
                key = "json"
            else:
                for suf, band in band_suffixes.items():
                    if fname.endswith(suf):
                        key = band
                        break
            if key is None:
                continue
            buf.setdefault(patch, {})[key] = tar.extractfile(member).read()
            # Complete once we have all 12 bands + the json.
            if len(buf[patch]) == len(ALL_BANDS) + 1:
                flush_patch(patch)
                if done == total:
                    break

    missing = needed - set(p for p in needed if (out / plan[p][0] / f"{plan[p][1]:0{plan[p][2]}d}.npy").exists())
    if missing:
        print(f"[prep] WARNING: {len(missing)} patches not found in archive (e.g. "
              f"{sorted(missing)[:3]})", flush=True)

    np.save(out / "train_y.npy", train_y)
    np.save(out / "val_y.npy", val_y)
    print(f"[prep] done -> {out}  (train_y {train_y.shape} pos/class "
          f"min={train_y.sum(0).min()} max={train_y.sum(0).max()}; "
          f"val_y {val_y.shape})", flush=True)


if __name__ == "__main__":
    main()
