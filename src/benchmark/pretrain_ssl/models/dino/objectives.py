import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOV2Loss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def _cross_entropy_terms(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= max(n_loss_terms, 1)
        return total_loss

    def forward(self, student_output, teacher_output, epoch):
        dino_loss = self._cross_entropy_terms(student_output, teacher_output, epoch)
        self.update_center(teacher_output)
        return dino_loss, {
            "dino_loss": dino_loss.detach(),
            "koleo_loss": torch.zeros_like(dino_loss.detach()),
            "total_loss": dino_loss.detach(),
        }

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            world_size = dist.get_world_size()
        else:
            world_size = 1
        batch_center = batch_center / (len(teacher_output) * world_size)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOV3Loss(DINOV2Loss):
    """
    DINOv3-style objective:
    - DINO teacher-student cross-view loss
    - KoLeo regularization to spread embeddings on the hypersphere
    """

    def __init__(self, *args, koleo_weight=0.1, use_koleo=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.koleo_weight = koleo_weight
        self.use_koleo = use_koleo

    def _koleo_loss(self, student_output):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            z = F.normalize(student_output.float(), p=2, dim=-1)
            similarity = torch.matmul(z, z.T)
            similarity.fill_diagonal_(-1.0)
            nn_similarity, _ = similarity.max(dim=1)
            distances = (2 - 2 * nn_similarity).clamp_min(1e-12).sqrt()
            return -torch.log(distances).mean()

    def forward(self, student_output, teacher_output, epoch):
        dino_loss = self._cross_entropy_terms(student_output, teacher_output, epoch)
        koleo_loss = self._koleo_loss(student_output) if self.use_koleo else torch.zeros_like(dino_loss)
        total = dino_loss + (self.koleo_weight * koleo_loss)
        self.update_center(teacher_output)
        return total, {
            "dino_loss": dino_loss.detach(),
            "koleo_loss": koleo_loss.detach(),
            "total_loss": total.detach(),
        }


def build_dino_objective(args):
    common = dict(
        out_dim=args.out_dim,
        ncrops=args.local_crops_number + 2,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    )
    if args.objective == "dino_v2":
        return DINOV2Loss(**common)
    if args.objective == "dino_v3":
        return DINOV3Loss(
            **common,
            koleo_weight=args.koleo_weight,
            use_koleo=args.use_koleo,
        )
    raise ValueError(f"Unsupported objective: {args.objective}")
