#!/usr/bin/env python
"""Build the SSL4EO-S12 ``s2c`` LMDB (for DINOv3 pretraining) from raw GeoTIFFs.

The Hugging Face copy (``wangyi111/SSL4EO-S12``, ``s2_l1c/``) extracts to a folder
of per-patch / per-season / per-band GeoTIFFs. This wraps the existing reader
``datasets.SSL4EO.ssl4eo_dataset.SSL4EO`` with an LMDB writer that takes a
**configurable ``map_size``** -- the in-repo ``make_lmdb`` hardcodes a 1 TiB cap,
which is fine for the ~0.9 TB uint8 LMDB but too small for an int16 one (~1.8 TB).

The on-disk record format matches what :class:`ssl4eo_dinov3.datasets.SSL4EOS2Dataset`
reads: ``pickle.dumps((sample.tobytes(), sample.shape))`` keyed by ``str(index)``,
with ``sample`` of shape ``(seasons, 13, 264, 264)`` (B10 is dropped later, at load
time, to get the 12-band L2A-compatible input).

Expected input layout (what ``SSL4EO`` expects)::

    <root>/s2c/<patch_id>/<season>/<band>.tif      # band in B01..B12 (L1C, 13 bands)

If your extracted folder is named ``s2_l1c`` (HF) instead of ``s2c``, either rename
it or create a symlink ``<root>/s2c -> <root>/s2_l1c`` (``scripts/gcp_setup.sh``
does this automatically).

Examples
--------
Full 250k uint8 LMDB (matches the original pretraining encoding)::

    python ssl4eo_dinov3/scripts/build_lmdb.py \
        --root /data/s2c_root --out /data/ssl4eo_s2c_uint8.lmdb

Quick 200-sample smoke LMDB to validate the end-to-end build::

    python ssl4eo_dinov3/scripts/build_lmdb.py \
        --root /data/s2c_root --out /tmp/ssl4eo_smoke.lmdb --limit 200

int16 (full precision; needs a bigger map_size and a larger disk)::

    python ssl4eo_dinov3/scripts/build_lmdb.py \
        --root /data/s2c_root --out /data/ssl4eo_s2c_int16.lmdb \
        --dtype int16 --map-size-gb 2200
    # NB: train with SSL4EOS2:root=...,dtype=int16 so the loader decodes int16.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Make the in-repo `datasets` package importable (pretrain_ssl/ is parents[2]).
PRETRAIN_SSL_DIR = Path(__file__).resolve().parents[2]
if str(PRETRAIN_SSL_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_SSL_DIR))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the s2c LMDB for DINOv3 pretraining from raw SSL4EO GeoTIFFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="Directory that CONTAINS the 's2c' modality folder "
                        "(i.e. <root>/s2c/<patch>/<season>/<band>.tif).")
    p.add_argument("--out", required=True, help="Output LMDB path (must not exist).")
    p.add_argument("--mode", default="s2c", choices=["s2c", "s2a", "s1"],
                   help="Modality to encode. The DINOv3 pipeline uses s2c.")
    p.add_argument("--dtype", default="uint8", choices=["uint8", "int16"],
                   help="On-disk dtype. uint8 = reflectance/10000*255 (smaller, "
                        "matches original LMDB); int16 = raw reflectance (full precision).")
    p.add_argument("--normalize", action="store_true",
                   help="Use 2-sigma per-band standardization to uint8 (overrides "
                        "--dtype encoding for s2a/s2c). Recompute band stats afterwards.")
    p.add_argument("--limit", type=int, default=None,
                   help="Encode only the first N patches (smoke build). Default: all.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers for parallel GeoTIFF reading.")
    p.add_argument("--map-size-gb", type=float, default=1100.0,
                   help="LMDB max map size in GiB. uint8 250k ~ 900 GB (1100 is safe); "
                        "use ~2200 for an int16 250k LMDB.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into an existing --out path.")
    p.add_argument("--skip-errors", action="store_true",
                   help="Skip patches whose GeoTIFFs fail to read (e.g. truncated/"
                        "corrupt tiles) instead of aborting. Skipped patch ids are "
                        "logged to <out>.bad_patches.txt; LMDB keys stay contiguous.")
    return p.parse_args(argv)


class _SafeDataset:
    """Wrap a dataset so a per-item read error yields ``None`` instead of killing
    the DataLoader worker. Used by ``--skip-errors`` to tolerate the handful of
    truncated/corrupt GeoTIFFs in the HF export. Returns ``(None, patch_id)`` on
    failure so the writer can log which patch was dropped."""

    def __init__(self, base, patch_ids):
        self._base = base
        self._patch_ids = patch_ids

    def __len__(self):
        return len(self._base)

    def __getitem__(self, index):
        try:
            return self._base[index], None
        except Exception as exc:  # rasterio.RasterioIOError, OSError, ...
            pid = self._patch_ids[index] if index < len(self._patch_ids) else str(index)
            return None, (pid, f"{type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    args = _parse_args(argv)

    import lmdb  # noqa: E402  (heavy / training-env-only deps imported lazily)
    from torch.utils.data import DataLoader, Subset  # noqa: E402
    from tqdm import tqdm  # noqa: E402

    from datasets.SSL4EO.ssl4eo_dataset import SSL4EO  # noqa: E402

    modality_dir = os.path.join(args.root, args.mode)
    if not os.path.isdir(modality_dir):
        raise SystemExit(
            f"[build_lmdb] expected modality folder '{modality_dir}' not found.\n"
            f"  Arrange the data as <root>/{args.mode}/<patch_id>/<season>/<band>.tif, "
            f"or symlink it (e.g. ln -s <root>/s2_l1c <root>/s2c)."
        )
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"[build_lmdb] refusing to overwrite existing path: {args.out} "
                         f"(pass --overwrite to allow).")

    base = SSL4EO(root=args.root, normalize=args.normalize, mode=[args.mode], dtype=args.dtype)
    patch_ids = list(base.ids)
    n_total = len(base)
    ds = base
    if args.limit is not None:
        n = min(args.limit, n_total)
        ds = Subset(ds, list(range(n)))
    else:
        n = n_total
    if args.skip_errors:
        # patch_ids[index] resolves correctly for both the full set and the
        # range(n) Subset, since the Subset is a contiguous prefix.
        ds = _SafeDataset(ds, patch_ids)

    enc = "2sigma-uint8" if args.normalize else args.dtype
    print(f"[build_lmdb] root={args.root} mode={args.mode} encoding={enc} "
          f"samples={n}/{n_total} workers={args.num_workers} "
          f"skip_errors={args.skip_errors} -> {args.out}")

    map_size = int(args.map_size_gb * (1024 ** 3))
    env = lmdb.open(args.out, map_size=map_size)
    # batch_size=1 + this collate yields the raw (s1, s2a, s2c) tuple per item.
    loader = DataLoader(ds, batch_size=1, num_workers=args.num_workers,
                        collate_fn=lambda batch: batch[0])

    bad = []
    txn = env.begin(write=True)
    written = 0
    for index, item in enumerate(tqdm(loader, total=n, desc="Creating LMDB")):
        if args.skip_errors:
            sample_tuple, err = item
            if sample_tuple is None:
                bad.append(err)
                continue
        else:
            sample_tuple = item
        s1, s2a, s2c = sample_tuple
        sample = {"s1": s1, "s2a": s2a, "s2c": s2c}[args.mode]
        sample = np.ascontiguousarray(np.asarray(sample))
        obj = (sample.tobytes(), sample.shape)
        # Key by the running written-count so keys stay contiguous (0..N-1) even
        # when patches are skipped -- SSL4EOS2Dataset reads keys str(0..len-1).
        txn.put(str(written).encode(), pickle.dumps(obj))
        written += 1
        if written % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.sync()
    env.close()

    if bad:
        bad_path = args.out + ".bad_patches.txt"
        with open(bad_path, "w") as fh:
            for pid, msg in bad:
                fh.write(f"{pid}\t{msg}\n")
        print(f"[build_lmdb] skipped {len(bad)} unreadable patch(es); logged to {bad_path}")
    print(f"[build_lmdb] done: wrote {written} samples to {args.out}")
    print("[build_lmdb] next: recompute per-band stats and paste into the config:")
    print(f"  python ssl4eo_dinov3/scripts/compute_band_stats.py --lmdb {args.out} --samples 20000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
