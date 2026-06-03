#!/usr/bin/env python
"""Tail DINOv3's ``training_metrics.json`` and mirror it to W&B / TensorBoard.

DINOv3 logs JSON-lines (one dict per logged step, with an ``iteration`` field)
to ``<output_dir>/training_metrics.json``. This watcher follows that file and
forwards each record, so we get live dashboards without modifying upstream code.

Run alongside training (rank-0 host):
    python sync_metrics.py --output-dir /workspace/out --wandb-project ssl4eo-dinov3 --run-name vits16
    # or TensorBoard only:
    python sync_metrics.py --output-dir /workspace/out --tensorboard
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _follow(path: str, poll: float = 2.0):
    """Yield successive JSON records appended to ``path`` (like ``tail -f``)."""
    while not os.path.exists(path):
        time.sleep(poll)
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, help="Training output dir (contains training_metrics.json)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tensorboard", action="store_true")
    args = ap.parse_args()

    metrics_path = os.path.join(args.output_dir, "training_metrics.json")

    wb = None
    if args.wandb_project:
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, dir=args.output_dir)

    tb = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    print(f"[sync_metrics] following {metrics_path}")
    for record in _follow(metrics_path):
        step = int(record.get("iteration", 0))
        scalars = {k: v for k, v in record.items() if isinstance(v, (int, float)) and k != "iteration"}
        if wb is not None:
            wb.log(scalars, step=step)
        if tb is not None:
            for k, v in scalars.items():
                tb.add_scalar(k, v, step)


if __name__ == "__main__":
    main()
