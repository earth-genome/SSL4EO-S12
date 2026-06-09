#!/usr/bin/env python
"""Consolidate an FSDP DCP training checkpoint into a ``teacher_checkpoint.pth``.

Training ran with ``evaluation.eval_period_iterations: 0``, so ``do_test`` never
wrote the ``eval/<it>/teacher_checkpoint.pth`` that :func:`build_backbone` expects.
The sharded DCP checkpoint in ``ckpt/<it>/`` holds the full ``SSLMetaArch`` state;
the EMA teacher backbone lives under ``model.teacher.backbone.*``.

This reads *only* those tensors (single process, no distributed init, skipping the
heavy optimizer state), strips the ``model.teacher.`` prefix so keys become
``backbone.*``, and saves ``{"teacher": state_dict}`` -- exactly the layout
``build_backbone._state_dict_from_checkpoint`` consumes.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="DCP checkpoint dir (contains .metadata + *.distcp)")
    ap.add_argument("--out", required=True, help="Output teacher_checkpoint.pth")
    ap.add_argument("--prefix", default="teacher.backbone.", help="Sub-key (under 'model') to extract")
    args = ap.parse_args()

    # The checkpoint stores the SSLMetaArch state as {"model": {<dotted keys>}}.
    # Its metadata only unflattens one level ("model" / "teacher.backbone...."),
    # so the prefix planner can only select the whole "model" subtree -- which
    # still skips the (much larger) optimizer state. We then filter in-process.
    full = _load_state_dict_from_keys("model", checkpoint_id=args.ckpt)
    model_sd = full.get("model", {})

    sd = {k[len("teacher."):]: v for k, v in model_sd.items() if k.startswith(args.prefix)}
    if not sd:
        raise SystemExit(
            f"[consolidate] no keys under 'model.{args.prefix}' in {args.ckpt} "
            f"(model subtree had {len(model_sd)} keys)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"teacher": sd}, args.out)
    print(f"[consolidate] {len(sd)} backbone tensors -> {args.out}")
    print("  sample keys:", list(sd)[:3])


if __name__ == "__main__":
    main()
