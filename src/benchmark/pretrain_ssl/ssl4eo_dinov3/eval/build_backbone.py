"""Load a trained DINOv3 12-band backbone for embedding extraction / probing.

Reconstructs the ViT from the training config, loads the teacher (EMA) weights
saved by ``dinov3.train.train.do_test`` (a ``{"teacher": state_dict}`` file whose
keys are prefixed ``backbone.``), and exposes a frozen CLS-embedding function.
Shared by the offline eval harness and the production inference script.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Sequence, Tuple

import torch

from ssl4eo_dinov3._paths import ensure_dinov3_on_path

logger = logging.getLogger("ssl4eo_dinov3.eval")


def _load_cfg(config_path: str):
    from omegaconf import OmegaConf
    from dinov3.configs import get_default_config

    cfg = OmegaConf.merge(get_default_config(), OmegaConf.load(config_path))
    return cfg


def _state_dict_from_checkpoint(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "teacher" in ckpt:
        sd = ckpt["teacher"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    # Keep only the backbone, stripping the ModuleDict "backbone." prefix.
    backbone_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
    return backbone_sd if backbone_sd else dict(sd)


def build_backbone(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    rope_dtype: str = "fp32",
) -> Tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor], Tuple[List[float], List[float]]]:
    """Return ``(model, embed_fn, (mean, std))``.

    ``embed_fn(x)`` maps a normalized ``(B, C, H, W)`` tensor to ``(B, D)`` CLS
    embeddings. ``mean``/``std`` are the per-band normalization stats from cfg.
    """
    ensure_dinov3_on_path()
    from dinov3.models import vision_transformer as vits

    cfg = _load_cfg(config_path)
    s = cfg.student

    vit_kwargs = dict(
        patch_size=s.patch_size,
        in_chans=s.in_chans,
        ffn_layer=s.ffn_layer,
        # ffn_ratio is hard-coded by the vit_* factories (4 for small/base), so
        # passing it here would collide as a duplicate keyword. The config value
        # matches the factory default, so we let the factory set it.
        # layerscale_init must be non-None or the blocks use nn.Identity and the
        # trained ls1/ls2.gamma weights are silently dropped (wrong forward pass).
        layerscale_init=getattr(s, "layerscale", None),
        n_storage_tokens=s.n_storage_tokens,
        norm_layer=s.norm_layer,
        qkv_bias=getattr(s, "qkv_bias", True),
        pos_embed_rope_base=getattr(s, "pos_embed_rope_base", 100.0),
        pos_embed_rope_normalize_coords=getattr(s, "pos_embed_rope_normalize_coords", "separate"),
        pos_embed_rope_dtype=rope_dtype,
    )
    model = vits.__dict__[s.arch](**vit_kwargs)

    sd = _state_dict_from_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"loaded backbone: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        logger.info(f"  missing (first 10): {missing[:10]}")
    if unexpected:
        logger.info(f"  unexpected (first 10): {unexpected[:10]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed_fn(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        feats = model.forward_features(x)
        if isinstance(feats, list):
            feats = feats[0]
        return feats["x_norm_clstoken"].float().cpu()

    mean: Sequence[float] = list(cfg.crops.rgb_mean)
    std: Sequence[float] = list(cfg.crops.rgb_std)
    return model, embed_fn, (list(mean), list(std))
