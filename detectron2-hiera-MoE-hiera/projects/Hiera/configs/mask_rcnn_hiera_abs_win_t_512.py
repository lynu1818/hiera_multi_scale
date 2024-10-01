from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone.hiera_abs_win import HieraAbsWin

from .common.coco_loader_lsj import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(HieraAbsWin)(
    model_name = "mae_hiera_abs_win_tiny_224",
    img_size=1024,
    embed_dim=96,
    num_heads=1,
    stages=(1, 2, 7, 2),
    out_features=["stage_0", "stage_1", "stage_2", "stage_3"],
)

model.backbone.in_features = "${.bottom_up.out_features}"

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "/home/lynuc/model_0079999.pth"
train.output_dir = "/home/lynuc/hiera_multi_scale/output/mask_rcnn_hiera_abs_win_t_512"
train.bottom_up_ckpt = '/home/s108061519/hiera_moe//logs/in1k_mae/mae_hiera_abs_win_tiny_512/epoch-epoch=99.ckpt'
dataloader.train.total_batch_size = 32

# 36 epochs
train.max_iter = 67500 * 64 // dataloader.train.total_batch_size
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[52500, 62500, 67500],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 1.6e-4
