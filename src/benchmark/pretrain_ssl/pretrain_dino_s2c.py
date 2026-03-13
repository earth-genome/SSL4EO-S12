# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import builtins

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from models.dino import utils
import models.dino.vision_transformer as vits
from models.dino.vision_transformer import DINOHead
from models.dino.objectives import build_dino_objective


from datasets.SSL4EO.ssl4eo_dataset_lmdb import LMDBDataset
from models import cv_transforms as cvtransforms
from models.rs_transforms_uint8 import RandomChannelDrop, GaussianBlur, Solarize, RandomBrightness, RandomContrast, ToGray, RandomSensorDrop_S1S2
### end of change ###
import pdb

from torch.utils.tensorboard import SummaryWriter

#import warnings
#warnings.filterwarnings("error")

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs, help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--objective', default='dino_v3', type=str, choices=['dino_v2', 'dino_v3'],
        help='Objective to optimize during pretraining.')
    parser.add_argument('--head_nlayers', default=3, type=int,
        help='Number of MLP layers in DINO projection head.')
    parser.add_argument('--head_hidden_dim', default=2048, type=int,
        help='Hidden dimension in DINO projection head.')
    parser.add_argument('--head_bottleneck_dim', default=256, type=int,
        help='Bottleneck dimension in DINO projection head.')
    parser.add_argument('--head_use_weight_norm', default=True, type=utils.bool_flag,
        help='Use weight normalization in DINO projection head output layer.')
    parser.add_argument('--enable_rope', default=False, type=utils.bool_flag,
        help='Enable RoPE positional encoding in ViT attention (replaces learned absolute pos_embed).')
    parser.add_argument('--rope_base', default=10000, type=int,
        help='RoPE frequency base.')
    parser.add_argument('--drop_legacy_pos_embed', default=True, type=utils.bool_flag,
        help='Drop legacy pos_embed keys when loading old checkpoints into RoPE-enabled models.')
    parser.add_argument('--dino_v3_mode', default='default', type=str, choices=['default', 'full'],
        help="DINOv3 mode: 'default' keeps DINO+KoLeo, 'full' enables DINO+iBOT+Gram.")
    parser.add_argument('--enable_ibot', default=False, type=utils.bool_flag,
        help='Enable iBOT branch in full DINOv3 mode.')
    parser.add_argument('--ibot_weight', default=1.0, type=float,
        help='Weight for iBOT term in full DINOv3 mode.')
    parser.add_argument('--gram_weight', default=0.0, type=float,
        help='Weight for Gram anchoring term in full DINOv3 mode.')
    parser.add_argument('--gram_teacher_checkpoint', default='', type=str,
        help='Checkpoint used to initialize frozen Gram teacher. If empty and --resume is set, uses checkpoint.pth.')

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--student_temp', default=0.1, type=float,
        help='Temperature for student logits.')
    parser.add_argument('--center_momentum', default=0.9, type=float,
        help='EMA momentum for teacher output centering.')
    parser.add_argument('--koleo_weight', default=0.1, type=float,
        help='KoLeo regularization weight used by dino_v3 objective.')
    parser.add_argument('--use_koleo', default=True, type=utils.bool_flag,
        help='Enable KoLeo regularization for dino_v3 objective.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--schedule_mode', default='cosine', type=str,
        choices=['cosine', 'constant_after_warmup', 'constant'],
        help='Schedule mode for lr/wd.')

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--checkpoints_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # new
    parser.add_argument('--data', default='/path/to/imagenet/', type=str,
        help='Please specify path to the ImageNet folder.')
    parser.add_argument('--bands', type=str, default='all', help="input bands")
    parser.add_argument("--lmdb", action='store_true', help="use lmdb dataset")
    parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")
    parser.add_argument("--resume", action='store_true', help="resume from checkpoint")    
    
    
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--mode', nargs='*', default=['s2c'])
    parser.add_argument('--dtype', type=str, default='uint8')
    parser.add_argument('--season', type=str, default='augment')
    parser.add_argument('--in_size', type=int, default=224)    
        
    
    return parser


def _build_constant_after_warmup_schedule(base_value, epochs, niter_per_ep, warmup_epochs=0, start_value=0.0):
    warmup_iters = warmup_epochs * niter_per_ep
    total_iters = epochs * niter_per_ep
    if warmup_iters > 0:
        warmup = np.linspace(start_value, base_value, warmup_iters)
    else:
        warmup = np.array([])
    hold = np.ones(max(total_iters - warmup_iters, 0)) * base_value
    return np.concatenate((warmup, hold))


def _load_backbone_weights_for_gram(model, checkpoint_path, drop_legacy_pos_embed=False):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"Gram teacher checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["teacher_backbone", "teacher", "student_backbone", "student", "state_dict", "model"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"Gram teacher loading checkpoint key: {key}")
                break

    if not isinstance(state_dict, dict):
        print("Gram teacher checkpoint payload is not a state dict. Skipping.")
        return

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    if drop_legacy_pos_embed:
        before = len(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}
        dropped = before - len(state_dict)
        if dropped > 0:
            print(f"Gram teacher dropped {dropped} legacy pos_embed keys.")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Gram teacher loaded with msg: {msg}")


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    
    # suppress printing if not master
    if args.is_slurm_job and args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    if args.dino_v3_mode == "full":
        print("DINOv3 full mode enabled: DINO + iBOT + Gram.")
        if not args.enable_ibot:
            print("WARNING: full mode requested but --enable_ibot=false.")
        if args.gram_weight <= 0:
            print("WARNING: full mode requested but --gram_weight<=0.")

    
    # ============ preparing data ... ============
    transform = DataAugmentationDINO_S2(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.season,
        args.in_size,
    )
    
    if args.bands == 'RGB':
        bands = ['B04', 'B03', 'B02']
        args.n_channels = 3
    elif args.bands == 'B12':
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']        
        args.n_channels = 12
    elif args.bands == 'B13':
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']        
        args.n_channels = 13    
    elif args.bands == 'B2':
        bands = ['VH', 'VV']
    elif args.bands == 'B14':
        bands_s1 = ['VH', 'VV']
        bands_s2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        args.n_channels = 14
        
    if args.lmdb:
        dataset = LMDBDataset(
            lmdb_file=args.data,
            s2c_transform=transform,
            is_slurm_job=args.is_slurm_job,
            normalize=args.normalize,
            dtype=args.dtype,
            mode=args.mode            
        )
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
            in_chans=args.n_channels,
            use_rope=args.enable_rope,
            rope_base=args.rope_base,
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            in_chans=args.n_channels,
            use_rope=args.enable_rope,
            rope_base=args.rope_base,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models, [yi:need to adjust for more in_channels]
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
        student.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        teacher.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # if the network is a XCiT, [yi:need to adjust for more in_channels]
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    if args.enable_rope and args.arch not in vits.__dict__.keys():
        raise ValueError("--enable_rope currently supports ViT architectures only.")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        nlayers=args.head_nlayers,
        hidden_dim=args.head_hidden_dim,
        bottleneck_dim=args.head_bottleneck_dim,
        use_weight_norm=args.head_use_weight_norm,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(
            embed_dim,
            args.out_dim,
            args.use_bn_in_head,
            nlayers=args.head_nlayers,
            hidden_dim=args.head_hidden_dim,
            bottleneck_dim=args.head_bottleneck_dim,
            use_weight_norm=args.head_use_weight_norm,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = build_dino_objective(args).cuda()

    gram_teacher = None
    if args.objective == "dino_v3" and args.dino_v3_mode == "full" and args.gram_weight > 0:
        if args.arch not in vits.__dict__.keys():
            raise ValueError("Gram teacher is currently implemented for ViT backbones only.")
        gram_teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            in_chans=args.n_channels,
            use_rope=args.enable_rope,
            rope_base=args.rope_base,
        ).cuda()
        gram_teacher.eval()
        for p in gram_teacher.parameters():
            p.requires_grad = False
        gram_ckpt = args.gram_teacher_checkpoint or os.path.join(args.checkpoints_dir, "checkpoint.pth")
        _load_backbone_weights_for_gram(
            gram_teacher,
            gram_ckpt,
            drop_legacy_pos_embed=(args.enable_rope and args.drop_legacy_pos_embed),
        )

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.
    if args.schedule_mode == "cosine":
        lr_schedule = utils.cosine_scheduler(
            scaled_lr,
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            args.epochs, len(data_loader),
        )
    elif args.schedule_mode == "constant_after_warmup":
        lr_schedule = _build_constant_after_warmup_schedule(
            scaled_lr,
            args.epochs,
            len(data_loader),
            warmup_epochs=args.warmup_epochs,
            start_value=0.0,
        )
        wd_schedule = np.ones(args.epochs * len(data_loader)) * args.weight_decay
    elif args.schedule_mode == "constant":
        lr_schedule = np.ones(args.epochs * len(data_loader)) * scaled_lr
        wd_schedule = np.ones(args.epochs * len(data_loader)) * args.weight_decay
    else:
        raise ValueError(f"Unsupported schedule mode: {args.schedule_mode}")
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.resume:
        drop_key_substrings = None
        if args.enable_rope and args.drop_legacy_pos_embed:
            drop_key_substrings = ["pos_embed"]
            print("RoPE migration enabled: legacy pos_embed checkpoint keys will be skipped.")
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoints_dir, "checkpoint.pth"),
            run_variables=to_restore,
            drop_key_substrings=drop_key_substrings,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
            objective=dino_loss,
        )
    else:
        print("WARNING: --resume is not set; training starts from scratch.")
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, gram_teacher=gram_teacher)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'student_backbone': student.module.backbone.state_dict(),
            'teacher_backbone': teacher_without_ddp.backbone.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'objective': dino_loss.state_dict(),
            'objective_name': args.objective,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.checkpoints_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.checkpoints_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.checkpoints_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, gram_teacher=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        #images = [torch.cat((images_s2[i],images_s1[i]),axis=1) for i in range(len(images_s2))]
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        student_feats = None
        gram_teacher_feats = None
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            if args.objective == "dino_v3" and args.dino_v3_mode == "full" and args.gram_weight > 0 and gram_teacher is not None:
                global_views = torch.cat(images[:2], dim=0)
                student_feats = student.module.backbone(global_views)
                with torch.no_grad():
                    gram_teacher_feats = gram_teacher(global_views)
            loss, loss_terms = dino_loss(
                student_output,
                teacher_output,
                epoch,
                student_feats=student_feats,
                gram_teacher_feats=gram_teacher_feats,
            )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dino=loss_terms["dino_loss"].item())
        metric_logger.update(koleo=loss_terms["koleo_loss"].item())
        metric_logger.update(ibot=loss_terms["ibot_loss"].item())
        metric_logger.update(gram=loss_terms["gram_loss"].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = cvtransforms.Compose([
            cvtransforms.RandomHorizontalFlip(p=0.5),
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(14)], p=0.2),
        ])
        normalize = cvtransforms.Compose([
            cvtransforms.ToTensor(),
            #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        sensor_drop = cvtransforms.RandomApply([RandomSensorDrop_S1S2()], p=0.5)

        # first global crop
        self.global_transfo1 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            normalize,
            sensor_drop
        ])
        # second global crop
        self.global_transfo2 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            cvtransforms.RandomApply([Solarize(128)], p=0.2),
            normalize,
            sensor_drop
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(48, scale=local_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            normalize,
            sensor_drop
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops        
        
        
        
        
        
        
        
        
class DataAugmentationDINO_S2(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, season='fixed', in_size=224):
        flip_and_color_jitter = cvtransforms.Compose([
            cvtransforms.RandomHorizontalFlip(p=0.5),
            cvtransforms.RandomApply([
                RandomBrightness(0.4),
                RandomContrast(0.4)
            ], p=0.8),
            cvtransforms.RandomApply([ToGray(13)], p=0.2),
        ])
        normalize = cvtransforms.Compose([
            cvtransforms.ToTensor(),
            #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # first global crop
        self.global_transfo1 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(in_size, scale=global_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(in_size, scale=global_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            cvtransforms.RandomApply([Solarize(128)], p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation='BICUBIC'),
            flip_and_color_jitter,
            cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])
        
        
        self.season = season

    def __call__(self, image):
        
        if self.season=='augment':
            season1 = np.random.choice([0,1,2,3])
            season2 = np.random.choice([0,1,2,3])
            season3 = np.random.choice([0,1,2,3])
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice([0,1,2,3])
            season2 = season1
            season3 = season1
        elif self.season=='random':
            season1 = np.random.choice([0,1,2,3])
            season2 = season1
            season3 = season1

        x1 = np.transpose(image[season1,:,:,:],(1,2,0))
        x2 = np.transpose(image[season2,:,:,:],(1,2,0))
        x3 = np.transpose(image[season3,:,:,:],(1,2,0))
        
        crops = []
        crops.append(self.global_transfo1(x1))
        crops.append(self.global_transfo2(x2))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(x3))
        return crops

class DataAugmentationDINO_S1(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = cvtransforms.Compose([
            #cvtransforms.RandomHorizontalFlip(p=0.5),
            #cvtransforms.RandomApply([
            #    RandomBrightness(0.4),
            #    RandomContrast(0.4)
            #], p=0.8),
            #cvtransforms.RandomApply([ToGray(2)], p=0.2),
        ])
        normalize = cvtransforms.Compose([
            cvtransforms.ToTensor(),
            #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation='BICUBIC'),
            #flip_and_color_jitter,
            #cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(112, scale=global_crops_scale, interpolation='BICUBIC'),
            #flip_and_color_jitter,
            #cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            #cvtransforms.RandomApply([Solarize(128)], p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = cvtransforms.Compose([
            cvtransforms.RandomResizedCrop(48, scale=local_crops_scale, interpolation='BICUBIC'),
            #flip_and_color_jitter,
            #cvtransforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    
    
    
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
