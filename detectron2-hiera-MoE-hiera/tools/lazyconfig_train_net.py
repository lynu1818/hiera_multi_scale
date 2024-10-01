#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import logging

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

import math
import torch
import torch.nn.functional as F
from collections import OrderedDict

logger = logging.getLogger("detectron2")

def get_state_dict_buttom_up(model, cfg):
    state_dict = torch.load(cfg.train.bottum_up_ckpt, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        new_state_dict[k] = v
    state_dict = new_state_dict

    new_seq_length = model.backbone.bottom_up.pos_embed.shape[1]
    pos_embed = state_dict["pos_embed"]
    if 'pos_embed_window' in state_dict:
        return state_dict
    n, seq_length, hidden_dim = pos_embed.shape
    if n!=1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embed.shape}")
    if seq_length != new_seq_length:
        # (0, seq_length, hidden_dim) -> (0, hidden_dim, seq_length)
        pos_embed = pos_embed.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embed = pos_embed.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)

        new_seq_length_1d = int(math.sqrt(new_seq_length))
        torch._assert(new_seq_length_1d * new_seq_length_1d == new_seq_length, "new_seq_length is not a perfect square!")

        # Perform interpolation.
        new_pos_embedding_img = F.interpolate(
            pos_embed,
            size=new_seq_length_1d,
            mode="bicubic",
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        state_dict["pos_embed"] = new_pos_embedding_img
        assert model.backbone.bottom_up.pos_embed.shape == new_pos_embedding_img.shape
    return state_dict

def get_state_dict_trunk(model, cfg):
    state_dict = torch.load(cfg.train.trunk_ckpt, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[6:]
        new_state_dict[k] = v
    state_dict = new_state_dict

    # new_seq_length = model.backbone.trunk.pos_embed.shape[1]
    # pos_embed = state_dict["pos_embed"]
    # n, seq_length, hidden_dim = pos_embed.shape

    # if n!=1:
    #     raise ValueError(f"Unexpected position embedding shape: {pos_embed.shape}")
    # if seq_length != new_seq_length:
    #     # (0, seq_length, hidden_dim) -> (0, hidden_dim, seq_length)
    #     pos_embed = pos_embed.permute(0, 2, 1)
    #     seq_length_1d = int(math.sqrt(seq_length))
    #     torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

    #     # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
    #     pos_embed = pos_embed.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)

    #     new_seq_length_1d = int(math.sqrt(new_seq_length))

    #     torch._assert(new_seq_length_1d * new_seq_length_1d == new_seq_length, f"new_seq_length={new_seq_length} is not a perfect square!")

    #     # Perform interpolation.
    #     new_pos_embedding_img = F.interpolate(
    #         pos_embed,
    #         size=new_seq_length_1d,
    #         mode="bicubic",
    #         align_corners=True,
    #     )

    #     # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
    #     new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

    #     # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
    #     new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
    #     state_dict["pos_embed"] = new_pos_embedding_img
    #     assert model.backbone.trunk.pos_embed.shape == new_pos_embedding_img.shape
    del state_dict["pos_embed"]
    return state_dict

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)


    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    
    #cfg.train.ddp.find_unused_parameters = True
    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    
    if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
        model = trainer.model.module
    else:
        model = trainer.model

    # Load bottom-up weights (for self customed backbone)
    if 'bottum_up_ckpt' in cfg.train:
        state_dict = get_state_dict_buttom_up(model, cfg)
        if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
            trainer.model.module.backbone.bottom_up.load_state_dict(state_dict, strict=False)
        else:
            trainer.model.backbone.bottom_up.load_state_dict(state_dict, strict=False)
        del state_dict
        
        print(f"Successfully loaded bottom-up weights from {cfg.train.bottum_up_ckpt}")
    elif 'trunk_ckpt' in cfg.train:
        state_dict = get_state_dict_trunk(model, cfg)
        if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
            trainer.model.module.backbone.trunk.load_state_dict(state_dict, strict=False)
        else:
            trainer.model.backbone.trunk.load_state_dict(state_dict, strict=False)
        del state_dict
        
        print(f"Successfully loaded bottom-up weights from {cfg.train.trunk_ckpt}")

    torch.cuda.empty_cache()
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
