from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from detectron2.modeling.proposal_generator import RPN, StandardRPNHead

from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling.backbone import HieraDet, FpnNeck_HieraDet, ImageEncoder_HieraDet

from sam2.modeling.position_encoding import PositionEmbeddingSine

from .common.coco_loader_lsj import dataloader

constants = model_zoo.get_config("common/data/constants.py").constants


model = L(GeneralizedRCNN)(
    backbone=L(ImageEncoder_HieraDet)(
        trunk=L(HieraDet)(
            embed_dim=112,
            num_heads=2,
            stages=(2, 3, 16, 3),
        ),
        neck=L(FpnNeck_HieraDet)(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112],
            fpn_top_down_levels= [2, 3],
            fpn_interp_model = 'bicubic'
        )
    ),
    proposal_generator=L(RPN)(
        in_features=["stage_0", "stage_1", "stage_2", "stage_3"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["stage_0", "stage_1", "stage_2", "stage_3"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["stage_0", "stage_1", "stage_2", "stage_3"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=constants.imagenet_rgb256_mean,
    pixel_std=constants.imagenet_rgb256_std,
    input_format="RGB",
)

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
#train.init_checkpoint = "/media/Pluto/andy/detectron2/projects/Hiera/init_ckpt/hiera_official/mae_hiera_tiny_224-model_state.pth"
# train.output_dir = "./output/mask_rcnn_hieradet_b_plus_512"
# train.trunk_ckpt = '/home/s108061519/hiera_moe/logs/in1k_mae/mae_hiera_base_plus_512/epoch-epoch=259.ckpt'
dataloader.train.total_batch_size = 1
train.init_checkpoint = "/home/lynuc/model_0079999.pth"
train.output_dir = "/home/lynuc/hiera_multi_scale/output/mask_rcnn_hieradet_b_plus_512"
train.trunk_ckpt = '/home/lynuc/last.ckpt'


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
