#!/usr/bin/env python
"""Fine-tune the DINOv3 12-band backbone end-to-end on EuroSAT (multiclass).

Counterpart to the frozen linear probe: this unfreezes the backbone and trains a
linear head on top of the CLS token, to compare against the SSL4EO-S12 paper's
FINE-TUNING number (Table IV: DINO ViT-S/16 EuroSAT 99.0 acc) rather than the
linear-probing number (Table III: 97.7).

Consumes the same prepared tiles as the probe (``prep_eurosat.py`` output):
``<prep>/train/*.npy`` + ``train_y.npy`` and ``<prep>/val`` + ``val_y.npy``, where
tiles are 13-band uint8 in SSL4EO s2c order. B10 is dropped to 12 bands, tiles are
bicubic-resized to ``--img-size``, scaled by 1/255, and standardized with the
config's per-band stats -- identical normalization to ``extract_embeddings.py``.

The repo's transfer_classification finetune scripts target the OLD ViT, so this is
a standalone trainer for the DINOv3 architecture (RoPE / register tokens / SwiGLU).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the pretrain_ssl package root importable (eval/ -> ssl4eo_dinov3/ -> pretrain_ssl/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _load_split(prep_dir: str, split: str):
    d = os.path.join(prep_dir, split)
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    tiles = np.stack([np.load(os.path.join(d, f)) for f in files])  # (N,13,H,W) uint8
    y = np.load(os.path.join(prep_dir, f"{split}_y.npy"))
    assert len(tiles) == len(y), (len(tiles), len(y))
    return torch.from_numpy(tiles), torch.from_numpy(y).long()


class _Head(nn.Module):
    def __init__(self, backbone, dim, n_classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if isinstance(feats, list):
            feats = feats[0]
        return self.fc(feats["x_norm_clstoken"])


def _augment(x):
    """Cheap label-preserving aug for EuroSAT: random h/v flips + k*90 rotation."""
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, ()).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prep", required=True, help="prep_eurosat.py output dir (train/ val/ *_y.npy)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from ssl4eo_dinov3.datasets import KEEP_12BAND_INDICES
    from ssl4eo_dinov3.eval.build_backbone import build_backbone

    dev = args.device
    backbone, _, (mean, std) = build_backbone(args.config, args.checkpoint, device=dev)
    for p in backbone.parameters():  # build_backbone froze these; unfreeze for FT
        p.requires_grad_(True)

    keep = torch.tensor(KEEP_12BAND_INDICES)
    mean_t = torch.tensor(mean, device=dev).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, -1, 1, 1)

    def prep_batch(tiles_u8, train):
        x = tiles_u8.to(dev).float()
        if x.shape[1] == 13:
            x = x.index_select(1, keep.to(dev))
        if x.shape[-1] != args.img_size:
            x = F.interpolate(x, size=(args.img_size, args.img_size), mode="bicubic", align_corners=False)
        x = (x / 255.0 - mean_t) / std_t
        if train:
            x = _augment(x)
        return x

    Xtr, ytr = _load_split(args.prep, "train")
    Xva, yva = _load_split(args.prep, "val")
    n_classes = int(max(ytr.max(), yva.max())) + 1
    print(f"[ft] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  classes={n_classes}")

    model = _Head(backbone, dim=backbone.embed_dim, n_classes=n_classes).to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.fc.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [args.lr_backbone, args.lr_head]
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def set_lr(epoch_frac):  # cosine with linear warmup
        for gi, g in enumerate(opt.param_groups):
            if epoch_frac < args.warmup_epochs:
                scale = epoch_frac / max(1, args.warmup_epochs)
            else:
                prog = (epoch_frac - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                scale = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            g["lr"] = base_lrs[gi] * scale

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = 0
        for i in range(0, len(Xva), args.batch_size):
            xb = prep_batch(Xva[i : i + args.batch_size], train=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(xb)
            correct += (logits.argmax(1).cpu() == yva[i : i + args.batch_size]).sum().item()
        return 100.0 * correct / len(yva)

    n = len(Xtr)
    iters = math.ceil(n / args.batch_size)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for it in range(iters):
            set_lr(ep + it / iters)
            idx = perm[it * args.batch_size : (it + 1) * args.batch_size]
            xb = prep_batch(Xtr[idx], train=True)
            yb = ytr[idx].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        msg = f"[ft] epoch {ep + 1}/{args.epochs}  loss {run / iters:.4f}"
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            acc = evaluate()
            best = max(best, acc)
            msg += f"  val-acc {acc:.2f}%  best {best:.2f}%"
        print(msg, flush=True)

    print(f"[ft] DONE  best val-acc {best:.2f}%  (paper DINO ViT-S/16 FT = 99.0)")


if __name__ == "__main__":
    main()
