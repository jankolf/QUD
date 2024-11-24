import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from utils import losses
from config.config import config
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import (
    CallBackVerification,
    CallBackLogging,
    CallBackModelCheckpoint,
)
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, iresnet18
from backbones.mobilefacenet import MobileFaceNet
from backbones.vit import VisionTransformer
from backbones.mixnetm import mixnet_s

from utils.buffer import BufferKD, MoCo
from copy import deepcopy

torch.backends.cudnn.benchmark = True


def get_trainable_model(
    cfg,
    nn_architecture: str,
    local_rank: int,
    nn_weights_path: str = None,
):

    if "resnet100" in nn_architecture:
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=False).to(
            local_rank
        )
    elif "resnet50" in nn_architecture:
        backbone = iresnet50(num_features=cfg.embedding_size, use_se=False).to(
            local_rank
        )
    elif "resnet18" in nn_architecture:
        backbone = iresnet18(num_features=cfg.embedding_size, use_se=False).to(
            local_rank
        )
    elif "mobilefacenet" in nn_architecture:
        backbone = MobileFaceNet(input_size=(112, 112)).to(local_rank)
    elif "transface-b" in nn_architecture:
        backbone = VisionTransformer(
            img_size=112,
            patch_size=9,
            num_classes=cfg.embedding_size,
            embed_dim=512,
            depth=24,
            num_heads=8,
            drop_path_rate=0.1,
            norm_layer="ln",
            mask_ratio=0.1,
            using_checkpoint=True,
        ).to(local_rank)
    elif "shufflemixfacenet" in nn_architecture:
        backbone = mixnet_s(
            embedding_size=cfg.embedding_size,
            width_scale=1.0,
            gdw_size=512,
            shuffle=True,
        ).to(local_rank)
    else:
        raise ValueError("Unknown model architecture given.")

    if nn_weights_path:
        logging.info(f"Loading Weights {nn_weights_path}.")
        backbone.load_state_dict(
            torch.load(nn_weights_path, map_location=torch.device(local_rank))
        )

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank]
    )

    return backbone


def main(args, cfg):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
        with open(f"{cfg.output}settings.txt", "w") as f:
            for key, val in cfg.__dict__.items():
                f.write(f"{key}:\t\t{val}\n")
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    if cfg.dataset == "ms1mv2":
        trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    else:
        trainset = FaceDatasetFolder(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    backbone_teacher = get_trainable_model(
        cfg=cfg,
        nn_architecture=cfg.teacher,
        local_rank=local_rank,
        nn_weights_path=cfg.teacher_weights,
    )
    backbone_teacher.eval()

    backbone_student = get_trainable_model(
        cfg=cfg,
        nn_architecture=cfg.student,
        local_rank=local_rank,
        nn_weights_path=cfg.student_weights,
    )
    backbone_student.train()

    opt_backbone = torch.optim.SGD(
        params=[{"params": backbone_student.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )

    buffer = MoCo(
        backbone_student,
        backbone_teacher,
        dim=512,
        K=cfg.buffer_size,
        T=cfg.buffer_temperature,
        queue_type=cfg.queue_type,
    ).cuda(local_rank)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func
    )

    criterion = CrossEntropyLoss().cuda(local_rank)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(
        cfg.eval_step, rank, cfg.val_targets, cfg.benchmarks
    )
    callback_logging = CallBackLogging(
        50, rank, total_step, cfg.batch_size, world_size, writer=None
    )
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)

            output, target = buffer(img, img, epoch=epoch)

            loss_v = criterion(output, target)
            loss_v.backward()

            clip_grad_norm_(backbone_student.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_backbone.zero_grad()

            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch)
            results = callback_verification(global_step, backbone_student)

        scheduler_backbone.step()

        callback_checkpoint(global_step, backbone_student, None)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument("--k", type=int, default=1024, help="Queue size")
    parser.add_argument("--t", type=float, default=0.07, help="Temperature")
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--teacher", type=str, help="Teacher")
    parser.add_argument("--student", type=str, help="Student")
    args_ = parser.parse_args()

    print(
        f"k: {args_.k} | t: {args_.t} | dataset: {args_.dataset} | teacher: {args_.teacher} | student: {args_.student}"
    )

    args_.experiment_id = f"{args_.student}_{args_.dataset}_t={args_.t}_k={args_.k}_teacher={args_.teacher}"

    cfg = deepcopy(config)

    cfg.buffer_size = args_.k
    cfg.buffer_temperature = args_.t
    cfg.batch_size = 256

    cfg.output = f"{cfg.prefix_output}{args_.experiment_id}/"

    """
    Dataset
    """
    cfg.dataset = args_.dataset
    if args_.dataset == "ms1mv2":
        cfg.rec = f"{cfg.prefix_data}ms1mv2_112x112"
        cfg.num_classes = 85742
        cfg.num_image = 5822653
        cfg.num_epoch = 26
        cfg.warmup_epoch = -1
        cfg.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
        cfg.eval_step = 5686

        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < -1
                else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
            )  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

        cfg.lr_func = lr_step_func

    elif args_.dataset == "webface":
        cfg.rec = f"{cfg.prefix_data}casia_webface"
        cfg.num_classes = 10572
        cfg.num_image = 501195
        cfg.num_epoch = 40  #  [22, 30, 35]
        cfg.warmup_epoch = -1
        cfg.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
        cfg.eval_step = 958  # 33350

        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < cfg.warmup_epoch
                else 0.1 ** len([m for m in [22, 30, 40] if m - 1 <= epoch])
            )

        cfg.lr_func = lr_step_func

    elif args_.dataset == "stylegan2-2m":
        cfg.rec = f"{cfg.prefix_data}StyleGAN2-2M"
        cfg.num_classes = 0
        cfg.num_image = 2130042
        cfg.num_epoch = 26
        cfg.warmup_epoch = -1
        cfg.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
        cfg.eval_step = 5686

        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < -1
                else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
            )  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

        cfg.lr_func = lr_step_func

    elif args_.dataset == "idiff-face":
        cfg.rec = f"{cfg.prefix_data}IDiff-Face"
        cfg.num_classes = 10049
        cfg.num_image = 502403
        cfg.num_epoch = 40  #  [22, 30, 35]
        cfg.warmup_epoch = -1
        cfg.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
        cfg.eval_step = 958  # 33350

        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < cfg.warmup_epoch
                else 0.1 ** len([m for m in [22, 30, 40] if m - 1 <= epoch])
            )

        cfg.lr_func = lr_step_func

    """
    Teacher
    """
    if args_.teacher == "resnet100-ms1mv2":
        cfg.teacher = "resnet100"
        cfg.teacher_weights = (
            f"{cfg.prefix_data}models/resnet100_ms1mv2_arcface/295672backbone.pth"
        )

    elif args_.teacher == "resnet50-ms1mv2":
        cfg.teacher = "resnet50"
        cfg.teacher_weights = (
            f"{cfg.prefix_data}models/resnet50_ms1mv2/181952backbone.pth"
        )

    elif args_.teacher == "resnet50-webface":
        cfg.teacher = "resnet50"
        cfg.teacher_weights = (
            f"{cfg.prefix_data}models/resnet50_webface/resnet50_webface_backbone.pth"
        )

    elif args_.teacher == "transface-b":
        cfg.teacher = "transface-b"
        cfg.teacher_weights = (
            f"{cfg.prefix_data}models/transface-b_ms1mv2/ms1mv2_model_TransFace_B.pt"
        )

    """
    Student
    """
    cfg.student = args_.student

    main(args_, cfg)
