from types import SimpleNamespace
from pathlib import Path

config = SimpleNamespace()

config.project_id = "QUD"


config.prefix_data = "/data/"
config.prefix_output = "/output/"
config.benchmarks = "/test/"


# Settings for Buffer Size
config.buffer_size = 1024
config.buffer_temperature = 0.07
config.queue_type = "normal"

# Model config
config.teacher = "resnet50"
config.teacher_weights = None

config.student = "mobilefacenet"
config.student_weights = None

config.embedding_size = 512  # embedding size of model
config.SE = False  # SEModule
config.s = 64.0
config.m = 0.50
# Dataset
config.dataset = "emoreIresNet"  # training dataset

# Optimization
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512  # batch size per GPU
config.lr = 0.1
config.global_step = 0  # step to resume

# Output
config.output = f"{config.prefix_output}"

# type of network to train [iresnet100 | iresnet50]


if config.dataset == "emoreIresNet":
    config.rec = f"{config.prefix_data}file"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 26
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 5686

    def lr_step_func(epoch):
        return (
            ((epoch + 1) / (4 + 1)) ** 2
            if epoch < -1
            else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
        )  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

    config.lr_func = lr_step_func
