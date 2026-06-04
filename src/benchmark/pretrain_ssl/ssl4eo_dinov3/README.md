# DINOv3 pretraining on 12-band Sentinel-2 (SSL4EO-S12)

Re-implements the SSL4EO-S12 geospatial foundation model with the **official
DINOv3** objective/architecture (DINO + iBOT + KoLeo, register tokens, RoPE,
SwiGLU, optional Gram anchoring), pretrained **from scratch** on the 250k
SSL4EO-S12 LMDB at **12 bands** — the Sentinel-2 **L2A**-compatible set (B10
cirrus dropped), matching production inference.

> **Built with DINOv3.** This pipeline uses Meta's
> [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) as a
> pinned git submodule (`./dinov3`). Weights trained with this code are
> derivatives governed by the **DINOv3 License** — see [Licensing](#licensing).

## Design: zero edits to the upstream submodule

The DINOv3 submodule is kept pristine. All adaptation lives in this package and
is applied via three import-time monkeypatches (`patches.py`):

| Seam | Why | What we patch |
|------|-----|---------------|
| `dinov3.models.build_model` | upstream builds the ViT **without** `in_chans`, so `student.in_chans: 12` would be ignored | record `in_chans` and inject it into every `vit_*` factory → 12-channel patch embed |
| `dinov3.data.loaders._parse_dataset_str` | dataset names are hard-coded | register `SSL4EOS2:root=<lmdb>` → `SSL4EOS2Dataset` |
| `SSLMetaArch.build_data_augmentation_dino` | RGB-only `ColorJitter`/`Grayscale`/`Solarize` don't apply to 12 bands | swap in `DataAugmentationDINOv3MS` (multispectral, per-band S2 normalization, per-crop season sampling) |
| `dinov3.train.train.make_data_loader` | throughput | enable `persistent_workers` |

Layout:
```
ssl4eo_dinov3/
  datasets.py            # SSL4EOS2Dataset (12-band LMDB, B10 dropped)
  augmentations.py       # DataAugmentationDINOv3MS (multispectral multi-crop)
  patches.py             # the 4 monkeypatches (apply_dinov3_patches)
  train_ssl4eo.py        # torchrun entrypoint (patches -> dinov3.train.main)
  extract_embeddings.py  # production CLS-embedding extraction
  configs/               # ssl4eo_s2_vits16{,_gram}.yaml, ssl4eo_s2_vitb16.yaml
  eval/                  # build_backbone, linear_probe, run_certification.sh
  scripts/               # train_gcp.sh, train_prototype.sh, smoke_test.py,
                         # compute_band_stats.py, gcs_sync.sh, sync_metrics.py
  Dockerfile             # CUDA + torch>=2.7.1 image (Built with DINOv3)
../dinov3/               # pinned submodule (commit 31703e4)
```

## Setup

```bash
# from repo root
git submodule update --init src/benchmark/pretrain_ssl/dinov3

# env (PyTorch >= 2.7.1)
conda env create -f src/benchmark/pretrain_ssl/dinov3/conda.yaml && conda activate dinov3
pip install -r src/benchmark/pretrain_ssl/dinov3/requirements.txt
pip install -e src/benchmark/pretrain_ssl/dinov3
pip install -r src/benchmark/pretrain_ssl/ssl4eo_dinov3/requirements-gcp.txt
```

## 0. Validate the integration (CPU, no GPU/data needed)

```bash
cd src/benchmark/pretrain_ssl
python ssl4eo_dinov3/scripts/smoke_test.py
```
Builds a synthetic LMDB and checks dataset→augmentation→collate shapes (12-ch
global 224 / local 96) and that the ViT patch embed is built with `in_chans=12`.

## 1. (Recommended) Compute exact per-band normalization

The configs default to `mean=0.5, std=0.25` per band (a principled default for
the 2σ uint8 encoding). For best results recompute and paste into the config:
```bash
python ssl4eo_dinov3/scripts/compute_band_stats.py --lmdb /data/ssl4eo_s2c_uint8.lmdb --samples 20000
```

## 2. Prototype (1 GPU, ~100 steps)

```bash
cd src/benchmark/pretrain_ssl
bash ssl4eo_dinov3/scripts/train_prototype.sh /data/ssl4eo_s2c_uint8.lmdb /tmp/proto_out
```

## 3. Full pretraining (single multi-GPU node)

```bash
cd src/benchmark/pretrain_ssl
# args: <lmdb> <output_dir> [config] [batch_per_gpu]
bash ssl4eo_dinov3/scripts/train_gcp.sh \
    /data/ssl4eo_s2c_uint8.lmdb /out/vits16 \
    ssl4eo_dinov3/configs/ssl4eo_s2_vits16.yaml 64
```
`OFFICIAL_EPOCH_LENGTH` is auto-computed from the detected GPU count so
`optim.epochs=100` ≈ 100 passes over the 250k dataset. Tracking + GCS:
```bash
python ssl4eo_dinov3/scripts/sync_metrics.py --output-dir /out/vits16 --wandb-project ssl4eo-dinov3 --run-name vits16 &
bash   ssl4eo_dinov3/scripts/gcs_sync.sh /out/vits16 gs://my-bucket/ssl4eo-dinov3/vits16 &
```
Optional **stage-2 Gram anchoring** (sharper dense features), continued from the
stage-1 teacher:
```bash
bash ssl4eo_dinov3/scripts/train_gcp.sh /data/ssl4eo_s2c_uint8.lmdb /out/vits16_gram \
    ssl4eo_dinov3/configs/ssl4eo_s2_vits16_gram.yaml 32 \
    MODEL.WEIGHTS=/out/vits16/eval/<iter>/teacher_checkpoint.pth
```
H100 fp8: add `student.fp8_enabled=true` to the launch command.

> **Slow run / GPUs idling?** Training is I/O-bound on the LMDB, not compute-bound.
> See [PERFORMANCE.md](PERFORMANCE.md) for diagnosis commands and the readahead fix
> (a 2-line change that took a 2× A100 run from ~3 days to ~1 day).

## 4. Certify quality vs the old DINO (gate the run)

Export each benchmark's train/val tiles to `.npy` (12- or 13-band; B10 auto-dropped)
with label arrays, then:
```bash
CKPT=/out/vits16/eval/<iter>/teacher_checkpoint.pth \
CONFIG=ssl4eo_dinov3/configs/ssl4eo_s2_vits16.yaml \
EUROSAT_TRAIN_DIR=... EUROSAT_VAL_DIR=... EUROSAT_TRAIN_Y=... EUROSAT_VAL_Y=... \
bash ssl4eo_dinov3/eval/run_certification.sh
```
Gates against the original DINO ViT-S/16 baselines (**BigEarthNet 90.5 mAP,
EuroSAT 99.0 acc, So2Sat 62.2 acc**); a shortfall exits non-zero.

## 5. Production embedding extraction

```bash
python ssl4eo_dinov3/extract_embeddings.py \
    --config ssl4eo_dinov3/configs/ssl4eo_s2_vits16.yaml \
    --checkpoint /out/vits16/eval/<iter>/teacher_checkpoint.pth \
    --source tiff --input /data/l2a_tiles --out /out/embeddings
```
> **Train/serve normalization must match.** The model is trained on the uint8
> SSL4EO encoding scaled to [0,1]. Encode L2A tiles with the same 2σ uint8 scheme
> used to build the LMDB (default `--scale 255`), or pass [0,1] floats with
> `--scale 1.0`.

## Docker / Vertex AI

```bash
docker build -f src/benchmark/pretrain_ssl/ssl4eo_dinov3/Dockerfile -t gcr.io/<PROJECT>/ssl4eo-dinov3:latest .
docker push gcr.io/<PROJECT>/ssl4eo-dinov3:latest
# Vertex AI custom job: use this image; command =
#   bash ssl4eo_dinov3/scripts/train_gcp.sh <gcsfuse-lmdb> /workspace/out ...
```

## Expected improvements (from-scratch, realistic)

- **Dense/embedding quality**: register tokens remove attention artifacts; the
  iBOT patch loss + (stage-2) Gram anchoring yield sharper, spatially coherent
  patch features — DINOv3's headline win for dense/downstream tasks.
- **Train/serve consistency**: 12-band input drops the all-zero B10 channel the
  production pipeline was previously feeding.
- **Speed**: SDPA/FlashAttention + bf16 + `torch.compile` + TF32 (+ fp8 on H100)
  vs the old hand-written attention + fp16 GradScaler.
- *Caveat*: DINOv3's small-model quality publicly comes from distilling a ViT-7B
  teacher; from scratch on 250k images the gain is the **objective/architecture**,
  not distilled web weights. Hence the certification gate before committing the
  full run.

## Rough GCP cost (order-of-magnitude)

ViT-S/16, 100 epochs over 250k images ≈ a few hundred GPU-hours.

| Setup | Throughput-ish | Wall-clock | On-demand cost* |
|-------|----------------|-----------|-----------------|
| 8× A100-80GB (a2-ultragpu-8g) | high | ~0.5–1.5 days | ~$150–$500 |
| 8× H100 (a3-highgpu-8g) | higher (+fp8) | ~0.3–0.8 days | ~$200–$600 |
| Prototype: 1× A100/L4 | — | minutes | a few $ |
| Inference/embedding: 1× L4 | — | — | cheapest |

\* Very rough; depends on region, spot/committed-use discounts, and final
`batch_size_per_gpu`. Measure the first ~100 iters to extrapolate before
committing the full run.

## Licensing

- This adapter code is Apache-2.0 (consistent with the rest of SSL4EO-S12).
- The `./dinov3` submodule and any weights trained with it are governed by the
  **DINOv3 License** (commercial use permitted; requires a "Built with DINOv3"
  notice and shipping the license). See `dinov3/LICENSE.md`. When distributing
  trained checkpoints, include the DINOv3 License and the attribution above.
