"""Monkeypatches that adapt the pristine DINOv3 submodule to 12-band Sentinel-2.

We deliberately do **not** edit the vendored ``dinov3`` submodule (keeps it a
clean, license-clear pointer to Meta's release and trivially updatable). Instead
we patch three integration seams at import time:

1. ``in_chans`` injection: upstream ``dinov3.models.build_model`` builds the ViT
   kwargs *without* passing ``in_chans`` (so ``student.in_chans: 12`` in the
   config would otherwise be silently ignored and the patch-embed conv would be
   3-channel). We wrap every ``vit_*`` factory to default ``in_chans`` from the
   config so the patch embedding is created with 12 input channels.
2. Dataset registration: we extend ``dinov3.data.loaders._parse_dataset_str`` so
   that ``train.dataset_path = "SSL4EOS2:root=/path/to/lmdb"`` resolves to
   :class:`ssl4eo_dinov3.datasets.SSL4EOS2Dataset`.
3. Augmentation swap: we replace ``SSLMetaArch.build_data_augmentation_dino`` to
   build the multispectral multi-crop transform
   (:class:`ssl4eo_dinov3.augmentations.DataAugmentationDINOv3MS`).

Call :func:`apply_dinov3_patches` once per process *before* building the model /
data loader (the training entrypoint does this).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("dinov3")

_PATCHED = False


def _patch_in_chans() -> None:
    """Make every ``vit_*`` factory default ``in_chans`` from the config.

    ``build_model`` reads ``cfg.student`` (an OmegaConf node) and calls
    ``vits.__dict__[arch](**vit_kwargs)`` without ``in_chans``. We wrap the
    factories so that, when ``in_chans`` is absent from the call, we inject the
    value recorded on the model-config node by :func:`_patch_build_model`.
    """
    from dinov3.models import vision_transformer as vits

    for name in list(vars(vits)):
        if not name.startswith("vit_"):
            continue
        fn = getattr(vits, name)
        if not callable(fn) or getattr(fn, "_ssl4eo_wrapped", False):
            continue

        def _make(orig):
            def wrapped(*args, **kwargs):
                if "in_chans" not in kwargs and _IN_CHANS["value"] is not None:
                    kwargs["in_chans"] = _IN_CHANS["value"]
                return orig(*args, **kwargs)

            wrapped._ssl4eo_wrapped = True
            wrapped.__name__ = getattr(orig, "__name__", name)
            return wrapped

        setattr(vits, name, _make(fn))


# Module-level holder so the wrapped factories can see the desired channel count.
_IN_CHANS = {"value": None}


def _patch_build_model() -> None:
    """Record ``cfg.student.in_chans`` so wrapped factories can pick it up."""
    import dinov3.models as models

    orig_build_model = models.build_model

    if getattr(orig_build_model, "_ssl4eo_wrapped", False):
        return

    def build_model(args, only_teacher=False, img_size=224, device=None):
        in_chans = getattr(args, "in_chans", None)
        if in_chans is not None:
            _IN_CHANS["value"] = int(in_chans)
            logger.info(f"[ssl4eo] building backbone with in_chans={in_chans}")
        return orig_build_model(args, only_teacher=only_teacher, img_size=img_size, device=device)

    build_model._ssl4eo_wrapped = True
    models.build_model = build_model
    # ``build_model_from_cfg`` references ``build_model`` via the module global,
    # so reassigning the attribute is sufficient.


def _patch_dataset_registry() -> None:
    """Register ``SSL4EOS2:root=...`` in the dataset string parser."""
    import dinov3.data.loaders as loaders

    if getattr(loaders._parse_dataset_str, "_ssl4eo_wrapped", False):
        return

    from .datasets import SSL4EOS2Dataset

    orig_parser = loaders._parse_dataset_str
    allowed_keys = ("root", "dtype", "length", "drop_b10")

    def _parse_dataset_str(dataset_str: str):
        name = dataset_str.split(":")[0]
        if name != "SSL4EOS2":
            return orig_parser(dataset_str)
        kwargs = {}
        for token in dataset_str.split(":")[1:]:
            key, value = token.split("=")
            assert key in allowed_keys, f"Unsupported SSL4EOS2 key '{key}' (allowed: {allowed_keys})"
            if key == "length":
                kwargs[key] = int(value)
            elif key == "drop_b10":
                kwargs[key] = value.lower() in ("1", "true", "yes")
            else:
                kwargs[key] = value
        return SSL4EOS2Dataset, kwargs

    _parse_dataset_str._ssl4eo_wrapped = True
    loaders._parse_dataset_str = _parse_dataset_str


def _patch_dataloader() -> None:
    """Keep LMDB workers alive across the (infinite) loader for throughput.

    Upstream already uses ``pin_memory=True`` but leaves ``persistent_workers``
    at its ``False`` default. For a map-style LMDB this avoids repeatedly
    re-opening the environment and re-spawning workers.
    """
    import dinov3.train.train as train_mod

    orig = train_mod.make_data_loader
    if getattr(orig, "_ssl4eo_wrapped", False):
        return

    def make_data_loader(*args, **kwargs):
        if kwargs.get("num_workers", 0) and "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = True
        return orig(*args, **kwargs)

    make_data_loader._ssl4eo_wrapped = True
    train_mod.make_data_loader = make_data_loader


def _patch_augmentation() -> None:
    """Swap the meta-arch's data-augmentation builder for the MS variant."""
    from dinov3.train.ssl_meta_arch import SSLMetaArch

    if getattr(SSLMetaArch.build_data_augmentation_dino, "_ssl4eo_wrapped", False):
        return

    from .augmentations import DataAugmentationDINOv3MS

    def build_data_augmentation_dino(self, cfg):
        season = getattr(cfg.crops, "season", "augment") if hasattr(cfg, "crops") else "augment"
        return DataAugmentationDINOv3MS(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
            gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
            local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
            share_color_jitter=cfg.crops.share_color_jitter,
            horizontal_flips=cfg.crops.horizontal_flips,
            mean=list(cfg.crops.rgb_mean),
            std=list(cfg.crops.rgb_std),
            season=season,
        )

    build_data_augmentation_dino._ssl4eo_wrapped = True
    SSLMetaArch.build_data_augmentation_dino = build_data_augmentation_dino


def apply_dinov3_patches() -> None:
    """Apply all SSL4EO-S12 adapter patches (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_build_model()
    _patch_in_chans()
    _patch_dataset_registry()
    _patch_dataloader()
    _patch_augmentation()
    _PATCHED = True
    logger.info("[ssl4eo] DINOv3 adapter patches applied (in_chans, dataset, augmentation)")
