# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from lib.model.modules import build_trans_encoder, build_trans_decoder


class Transformer(nn.Module):

    def __init__(self, _args):

        super(Transformer, self).__init__()

        self.encoder = build_trans_encoder(_args)
        self.decoder = build_trans_decoder(_args)

        self._reset_parameters()

        self.in_channels = _args.in_channels
        self.num_head = _args.num_head

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, key_mask, pos_embed, query_embed):
        
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        src = src.flatten(2).permute(2, 0, 1)  # (HW, N, C)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, N, C)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (Num, N, C)
        key_mask = key_mask.flatten(1)  # (N, HW)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=key_mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=key_mask, pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

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


def build_transformer(_args):
    return Transformer(_args)


if __name__ == '__main__':
    from easydict import EasyDict as Edict

    args = Edict()
    args.transformer = Edict()

    args.transformer.in_channels = 256
    args.transformer.num_head = 4
    args.transformer.dim_feed = 1024
    args.transformer.dropout = 0.3
    args.transformer.activation = "relu"
    args.transformer.norm_before = False

    args.transformer.encoder_layers = 6
    args.transformer.decoder_layers = 6
    args.transformer.return_inter_decode = False

    transformer = build_transformer(args.transformer)
    
    s = torch.Tensor(torch.rand(1, 256, 8, 8))
    k = torch.Tensor(torch.zeros(1, 8, 8))
    p = torch.Tensor(torch.rand(1, 256, 8, 8))
    q = torch.Tensor(torch.rand(100, 256))

    out, pos = transformer(s, k, p, q)
    print([o.shape for o in out])
    print([p.shape for p in pos])

    from ptflops import get_model_complexity_info

    def prepare_input(resolution):
        x1 = torch.rand(*resolution[0])
        x2 = torch.rand(*resolution[1])
        x3 = torch.rand(*resolution[2])
        x4 = torch.rand(*resolution[3])
        return dict(src=x1, key_mask=x2, pos_embed=x3, query_embed=x4)

    flops, params = get_model_complexity_info(transformer,
                                              input_res=((2, 256, 8, 8), (2, 8, 8), (2, 256, 8, 8), (1, 256)),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  0.74 GMac
    #       - Params: 11.06 M

