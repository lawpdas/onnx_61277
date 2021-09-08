# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerDecoder(nn.Module):

    def __init__(self, layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))  # <--duplicated normalization
            # nn.LayerNorm(nn.LayerNorm(x)) is equal to nn.LayerNorm(x)
            # https://github.com/facebookresearch/detr/issues/94
            # https://github.com/pytorch/pytorch/issues/24930

        if self.norm is not None:
            output = self.norm(output)  # <--duplicated normalization
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, in_channels, num_heads, dim_feed, dropout, activation, normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.multi_head_attn = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_channels, dim_feed)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feed, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multi_head_attn(self.with_pos_embed(tgt, query_pos),
                                    self.with_pos_embed(memory, pos),
                                    memory,
                                    attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multi_head_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

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


def build_trans_decoder(_args):
    _layer = TransformerDecoderLayer(in_channels=_args.in_channels, num_heads=_args.num_heads, dim_feed=_args.dim_feed,
                                     dropout=_args.dropout, activation=_args.activation)
    _norm = nn.LayerNorm(_args.in_channels)
    _encoder = TransformerDecoder(layer=_layer, num_layers=_args.num_decoders, norm=_norm,
                                  return_intermediate=_args.return_inter_decode)
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

    args.transformer.decoder_layers = 6
    args.transformer.return_inter_decode = True

    trans_decoder = build_trans_decoder(args.transformer)

    x = torch.rand(64, 1, 256)
    m = torch.rand(64, 1, 256)
    y = trans_decoder(x, m)
    print(y.shape)

    from ptflops import get_model_complexity_info

    def prepare_input(resolution):
        tgt = torch.rand(resolution[0], 1, resolution[1])
        memory = torch.rand(resolution[0], 1, resolution[1])
        query_embed = torch.rand(1, 1, resolution[1])
        return dict(tgt=tgt, memory=memory, query_pos=query_embed)

    flops, params = get_model_complexity_info(trans_decoder, input_res=(64, 256),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  0.43 GMac
    #       - Params: 6.32 M
