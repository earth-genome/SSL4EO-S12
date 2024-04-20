import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import torch

PRETRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(PRETRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ROOT))

from models.dino.objectives import DINOV2Loss, DINOV3Loss, build_dino_objective


class TestDinoObjectives(unittest.TestCase):
    def test_dinov2_forward_finite(self):
        loss_fn = DINOV2Loss(
            out_dim=32,
            ncrops=4,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=0,
            nepochs=2,
        )
        student = torch.randn(8, 32)
        teacher = torch.randn(4, 32)
        total, terms = loss_fn(student, teacher, epoch=0)
        self.assertTrue(torch.isfinite(total).item())
        self.assertIn("dino_loss", terms)
        self.assertIn("total_loss", terms)

    def test_dinov3_forward_finite_and_koleo(self):
        loss_fn = DINOV3Loss(
            out_dim=32,
            ncrops=4,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=0,
            nepochs=2,
            koleo_weight=0.1,
            use_koleo=True,
        )
        student = torch.randn(8, 32)
        teacher = torch.randn(4, 32)
        total, terms = loss_fn(student, teacher, epoch=0)
        self.assertTrue(torch.isfinite(total).item())
        self.assertTrue(torch.isfinite(terms["koleo_loss"]).item())

    def test_objective_factory(self):
        args = SimpleNamespace(
            objective="dino_v3",
            out_dim=16,
            local_crops_number=2,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=0,
            epochs=1,
            student_temp=0.1,
            center_momentum=0.9,
            koleo_weight=0.1,
            use_koleo=True,
        )
        objective = build_dino_objective(args)
        self.assertIsInstance(objective, DINOV3Loss)


if __name__ == "__main__":
    unittest.main()
