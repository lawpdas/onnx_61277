# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoder(nn.Module):

    def __init__(self, layer, num_layers, norm=None):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, in_channels, num_heads, dim_feed, dropout, activation, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_channels, dim_feed)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feed, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    @staticmethod
    def _get_activation_fn(activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_trans_encoder(_args):
    _layer = TransformerEncoderLayer(in_channels=_args.in_channels, num_heads=_args.num_heads, dim_feed=_args.dim_feed,
                                     dropout=_args.dropout, activation=_args.activation)
    _norm = nn.LayerNorm(_args.in_channels) if _args.norm_before else None
    _encoder = TransformerEncoder(layer=_layer, num_layers=_args.num_encoders, norm=_norm)
    return _encoder


if __name__ == '__main__':
    from easydict import EasyDict as Edict

    args = Edict()
    args.transformer = Edict()

    args.transformer.in_channels = 256
    args.transformer.num_heads = 4
    args.transformer.dim_feed = 1024
    args.transformer.dropout = 0.3
    args.transformer.activation = "relu"
    args.transformer.norm_before = False

    args.transformer.encoder_layers = 6

    trans_encoder = build_trans_encoder(args.transformer)

    x = torch.rand(64, 1, 256)
    y = trans_encoder(x)
    print(y.shape)

    from ptflops import get_model_complexity_info

    def prepare_input(resolution):
        src = torch.rand(resolution[0], 1, resolution[1])
        return dict(src=src)

    flops, params = get_model_complexity_info(trans_encoder, input_res=(64, 256),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  0.32 GMac
    #       - Params: 4.74 M
