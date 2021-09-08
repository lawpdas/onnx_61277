# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from torchvision.ops import box_convert
from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # proj_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):  # (B, C, H, W)
        bs = 1

        # nl, _, cs = x.shape
        #
        # h = int(math.sqrt(nl))
        # w = int(math.sqrt(nl))
        #
        # not_mask = torch.ones((bs, h, w), dtype=torch.bool, device=x.device)

        not_mask = torch.ones_like(x[0, 0, :, :], dtype=torch.bool, device=x.device).unsqueeze(0)  # (1, H, W)

        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (1, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # (1, H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (1, H, W, 1) / (C/2, ) -> (1, H, W, C/2)
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # (B, H, W, C)

        pos = pos.reshape(bs, -1, self.num_pos_feats * 2).permute(1, 0, 2)  # (L, B, C)
        return pos


class PositionEmbeddingImage(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, raw_img: torch.Tensor):
        bs, _, imh, imw = raw_img.shape

        if self.normalize:
            # ab = (torch.arange(imw * imh) / (imw * imh)).reshape(imh, imw)
            ww = (torch.arange(imw) - imw * 0.5) / (imw * 0.5)
            hh = (torch.arange(imh) - imh * 0.5) / (imh * 0.5)
        else:
            # ab = torch.arange(imw * imh).reshape(imh, imw)
            ww = (torch.arange(imw) - imw * 0.5)
            hh = (torch.arange(imh) - imh * 0.5)

        hh, ww = torch.meshgrid(hh, ww)

        pos_embedding = torch.cat((
            # ab.unsqueeze(0).unsqueeze(0),
            hh.unsqueeze(0).unsqueeze(0),
            ww.unsqueeze(0).unsqueeze(0)), dim=1)

        return pos_embedding.repeat(bs, 1, 1, 1)
