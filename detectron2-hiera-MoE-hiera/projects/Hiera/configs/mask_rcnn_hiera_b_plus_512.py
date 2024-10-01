from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling import Hiera

from .common.coco_loader_lsj import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(Hiera)(
    model_name = "mae_hiera_base_plus_512",
    img_size=1024,
    embed_dim=112,
    num_heads=2,
    stages=(2, 3, 16, 3),
    out_features=["stage_0", "stage_1", "stage_2", "stage_3"],
)

model.backbone.in_features = "${.bottom_up.out_features}"

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
#train.init_checkpoint = "/media/Pluto/andy/detectron2/projects/Hiera/init_ckpt/hiera_official/mae_hiera_tiny_224-model_state.pth"
train.output_dir = "./output/mask_rcnn_hiera_b_plus_512"
train.bottum_up_ckpt = '/home/s108061519/hiera_moe/logs/in1k_mae/mae_hiera_base_plus_512/epoch-epoch=259.ckpt'
dataloader.train.total_batch_size = 1


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
