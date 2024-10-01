# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from functools import partial
from typing import List, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone

from timm.models.layers import DropPath, Mlp

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP


from .hiera_utils import pretrained_model, conv_nd, do_pool, do_masked_conv, Unroll, Reroll

from st_moe_pytorch import MoE as STMoE
from st_moe_pytorch import SparseMoEBlock

class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


def do_pool_HieraDet(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool_HieraDet(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        mlp_dropout: float = 0.,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        # self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer, drop=mlp_dropout)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool_HieraDet(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
            print(f"after window partition, xshape:{x.shape}, pad_hw: {pad_hw}")

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
        print(f"x.shape: {x.shape}, window_size: {window_size}, pad_hw: {pad_hw}, (H, W): {H}, {W}")
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        num_experts = None, # for compatibility with ST-MoE
        mlp_dropout: float = 0.,
        expert_dropout = None, # for compatibility with ST-MoE
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

        self.pool, self.q_stride = None, q_stride
        
        
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
        )

        
        self.norm1 = norm_layer(dim)
        
        if use_mask_unit_attn:
          print("mask unit attn")
          self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
          )
        else:
          print("multiscale attn")
          self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=heads,
            q_pool=self.pool,
          )
        

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer, drop=mlp_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        print(f"x.shape: {x.shape}")
        
        if self.use_mask_unit_attn:
            x = _flatten_images(x)
            x_norm = self.norm1(x)
            if self.dim != self.dim_out:
                x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
                x = x + self.drop_path(self.attn(x_norm))
        else:
            shortcut = x  # B, H, W, C
            x = self.norm1(x)

            # Skip connection
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)

            # Window partition
            window_size = self.window_size
            if window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)

            # Window Attention + Q Pooling (if stage change)
            x = self.attn(x)
            if self.q_stride:
                # Shapes have changed due to Q pooling
                window_size = self.window_size // self.q_stride[0]
                H, W = shortcut.shape[1:3]

                pad_h = (window_size - H % window_size) % window_size
                pad_w = (window_size - W % window_size) % window_size
                pad_hw = (H + pad_h, W + pad_w)

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, window_size, pad_hw, (H, W))

            x = shortcut + self.drop_path(x)


        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HieraBlockSTMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
        num_experts: int = 16,
        mlp_dropout = None, # for compatibility with Standard HieraBlock
        expert_dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        # self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)
        moe = STMoE(
            dim=dim_out,
            num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
            expert_dropout = expert_dropout,    # dropout rate for the experts
        )
        self.moe = SparseMoEBlock(
            moe,
            add_ff_before = True,
            add_ff_after = True,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        # MLP
        x_tmp = self.norm2(x)
        x_tmp = self.moe(x_tmp)
        if isinstance(x_tmp, tuple):
            x_tmp, _total_aux_loss, _balance_loss, _router_z_loss = x_tmp
        elif isinstance(x, torch.Tensor):
            x_tmp = x_tmp
            _total_aux_loss, _balance_loss, _router_z_loss = 0., 0., 0.
        else:
            raise ValueError("Invalid output type from MoE")
        
        x = x + self.drop_path(x_tmp)
        return x, _total_aux_loss, _balance_loss, _router_z_loss


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


# class PatchEmbed(nn.Module):
#     """
#     Image to Patch Embedding.
#     """

#     def __init__(
#         self,
#         kernel_size: Tuple[int, ...] = (7, 7),
#         stride: Tuple[int, ...] = (4, 4),
#         padding: Tuple[int, ...] = (3, 3),
#         in_chans: int = 3,
#         embed_dim: int = 768,
#     ):
#         """
#         Args:
#             kernel_size (Tuple): kernel size of the projection layer.
#             stride (Tuple): stride of the projection layer.
#             padding (Tuple): padding size of the projection layer.
#             in_chans (int): Number of input image channels.
#             embed_dim (int):  embed_dim (int): Patch embedding dimension.
#         """
#         super().__init__()
#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.proj(x)
#         # B C H W -> B H W C
#         x = x.permute(0, 2, 3, 1)
#         return x

class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        #x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x

def _flatten_images(x: torch.Tensor) -> torch.Tensor:
    # x is B, H, W, C
    # out is B, H*W, C
    return x.view(x.shape[0], -1, x.shape[-1])

class HieraAbsWin(Backbone):
    def __init__(
        self,
        model_name: str,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        mlp_dropout: float = 0.0,
        expert_dropout = None, # for compatibility with ST-MoE
        # ==============================================
        # Windowed positional embedding
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        out_features: List[str] = ["stage_0", "stage_1", "stage_2", "stage_3"],
    ):
        super().__init__()
        self.model_name = model_name

        self.window_spec = window_spec
        assert len(stages) == len(window_spec)

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.input_size = [img_size, img_size]
        print("input size: ", self.input_size)
        self.num_classes = num_classes

        self._out_features = out_features
        self._out_feature_channels = {}

        depth = sum(stages)
        self._out_features = out_features
        self._out_feature_channels = {}

        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(self.input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        # Setup roll and reroll modules
        self.unroll = Unroll(
            self.input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )

        self.reroll = Reroll(
            self.input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]
            
            window_size = self.window_spec[cur_stage - 1]

            if i - 1 in self.stage_ends:
                self._out_feature_channels[f'stage_{cur_stage}'] = dim_out

                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
                mlp_dropout=mlp_dropout,
            )

            
            # block = MultiScaleBlock(
            #     dim=embed_dim,
            #     dim_out=dim_out,
            #     num_heads=num_heads,
            #     mlp_ratio=mlp_ratio,
            #     drop_path=dpr[i],
            #     norm_layer=norm_layer,
            #     q_stride=self.q_stride if i in q_pool_blocks else None,
            #     window_size=window_size,
            #     mlp_dropout=mlp_dropout,
            # )


            embed_dim = dim_out
            self.blocks.append(block)
        self._out_feature_channels[f'stage_{cur_stage}'] = dim_out
        self._out_feature_strides = {f'stage_{i}': patch_stride[0] * 2 ** (i) for i in range(len(stages))}

        #self.norm = norm_layer(embed_dim)
        #self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # Initialize everything
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_window, std=0.02)
        self.apply(partial(self._init_weights))
        #self.head.projection.weight.data.mul_(head_init_scale)
        #self.head.projection.bias.data.mul_(head_init_scale)

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return ["pos_embed", "pos_embed_window"]
        
    @torch.jit.ignore
    def num_layers(self):
        return len(self.blocks)
    
    @torch.jit.ignore
    def get_layer_id(self, name: str):
        if name.startswith("pos_embed") or name.startswith("pos_embed_window") or name.startswith("patch_embed"):
            return 0
        elif "blocks" in name:
            return int(name.split(".")[1]) + 1
        
        return self.num_layers()

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        print(f'h: {h}, w:{w}, window_embed.shape:{window_embed.shape}')
        
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        print(f'pos_embed.shape:{pos_embed.shape}')
        
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]

        H, W = x.shape[2], x.shape[3]
        self.input_size = [H, W]
        print(f"Input shape before unroll: {x.shape}")

        self.tokens_spatial_shape = [i // s for i, s in zip(self.input_size, self.patch_stride)]
        
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]

        self.unroll = Unroll(
            self.input_size, self.patch_stride, [self.q_stride] * len(self.stage_ends[:-1])
        )

        self.reroll = Reroll(
            self.input_size,
            self.patch_stride,
            [self.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            self.q_pool,
        )

        # X is B, C, H, W
        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        # x = self.patch_embed(x)
        print(f"x.shape after patch embed: {x.shape}")
        
        # convert X to B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        x += self._get_pos_embed(x.shape[1:3])
        # print(f"x.shape after patch embed and positional embedding: {x.shape}")
        # convert X to B, H*W, C
        x = _flatten_images(x)
        print(f"x.shape without flatten image: {x.shape}")
                
        
        x = self.unroll(x)
        # print(f"x.shape after unroll: {x.shape}")


        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        outputs = {}
        stage = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in self.stage_ends:
                #intermediates.append(self.reroll(x, i, mask=mask))
                outputs[f'stage_{stage}'] = self.reroll(x, i, mask=mask).permute(0, 3, 1, 2)
                print(f"x.shape before permute in stage_{stage}: {x.shape}")
                # outputs[f'stage_{stage}'] = x.permute(0, 3, 1, 2)
                stage += 1

        return outputs

class HieraAbsWinSTMoE(nn.Module):
    def __init__(
        self,
        model_name: str,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        num_experts: int = 16,
        moe_stages: Tuple[bool, ...] = None,
        mlp_dropout: float = 0.0,
        expert_dropout: float = 0.0,
        # ==============================================
        # Windowed positional embedding
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
    ):
        super().__init__()
        self.model_name = model_name

        self.window_spec = window_spec
        assert len(stages) == len(window_spec)

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.input_size = [img_size, img_size]
        self.num_classes = num_classes

        depth = sum(stages)
        if moe_stages is None:
            moe_stages = [True] * depth
        assert len(moe_stages) == depth
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(self.input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        # Setup roll and reroll modules
        self.unroll = Unroll(
            self.input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            self.input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            blk_func = HieraBlockSTMoE if moe_stages[i] else HieraBlock
            block = blk_func(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
                num_experts=num_experts,
                mlp_dropout=mlp_dropout,
                expert_dropout=expert_dropout,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        # self.norm = norm_layer(embed_dim)
        # self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # Initialize everything
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_window, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return ["pos_embed", "pos_embed_window"]
        
    @torch.jit.ignore
    def num_layers(self):
        return len(self.blocks)
    
    @torch.jit.ignore
    def get_layer_id(self, name: str):
        if name.startswith("pos_embed") or name.startswith("pos_embed_window") or name.startswith("patch_embed"):
            return 0
        elif "blocks" in name:
            return int(name.split(".")[1]) + 1
        
        return self.num_layers()

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []

        # X is B, C, H, W
        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        # convert X to B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x += self._get_pos_embed(x.shape[1:3])
        # convert X to B, H*W, C
        x = _flatten_images(x)

        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if isinstance(x, torch.Tensor):
                x = x
                _total_aux_loss, _balance_loss, _router_z_loss = 0., 0., 0.
            elif isinstance(x, tuple):
                x, _total_aux_loss, _balance_loss, _router_z_loss = x

            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))

        if mask is None:
            x = x.mean(dim=1)
            # x = self.norm(x)
            # x = self.head(x)

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        return x
