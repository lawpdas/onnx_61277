# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from typing import Dict, List, Optional, Callable

import torch
import torch.nn as nn
import torchvision

from lib.model.backbones.helper import IntermediateLayerGetter
from lib.model.layers import FrozenBatchNorm2d

from utils.misc import is_main_process


class ResBackbone(nn.Module):

    def __init__(self,
                 arch: str,
                 pretrain: bool = True,
                 zero_init_residual: bool = False,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,

                 top_layer: str = None,
                 return_inter_layers: bool = False,

                 train_flag: bool = False,
                 train_all: bool = False,
                 ):
        super(ResBackbone, self).__init__()

        resnet = getattr(torchvision.models, arch)(
            pretrained=is_main_process() and pretrain,
            zero_init_residual=zero_init_residual,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )

        # extract layer1, ..., and layer4, giving as names `layer1`, ..., and layer4`
        if 'layer4' in top_layer:
            if return_inter_layers:
                return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}
            else:
                return_layers = {'layer4': 'layer4'}  # {name: out_name}

        elif 'layer3' in top_layer:
            if return_inter_layers:
                return_layers = {'layer1': 'layer1', 'layer2': 'layer2', 'layer3': 'layer3'}
            else:
                return_layers = {'layer3': 'layer3'}

        elif 'layer2' in top_layer:
            if return_inter_layers:
                return_layers = {'layer1': 'layer1', 'layer2': 'layer2'}
            else:
                return_layers = {'layer2': 'layer2'}

        else:
            return_layers = {'layer1': 'layer1'}

        self.body = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.out_layer = top_layer

        out_stride = 4
        dilation2, dilation3, dilation4 = replace_stride_with_dilation
        if 'layer4' in top_layer:
            out_channels = [64, 128, 256, 512] if arch in ('resnet18', 'resnet34') else [256, 512, 1024, 2048]
            out_stride = out_stride * (1 if dilation2 else 2) * (1 if dilation3 else 2) * (1 if dilation4 else 2)

        elif 'layer3' in top_layer:
            out_channels = [64, 128, 256] if arch in ('resnet18', 'resnet34') else [256, 512, 1024]
            out_stride = out_stride * (1 if dilation2 else 2) * (1 if dilation3 else 2)

        elif 'layer2' in top_layer:
            out_channels = [64, 128] if arch in ('resnet18', 'resnet34') else [256, 512]
            out_stride = out_stride * (1 if dilation2 else 2)

        else:
            out_channels = [64] if arch in ('resnet18', 'resnet34') else [256]

        self.out_channels_list = [out_channels[-1]] if not return_inter_layers else out_channels
        self.out_num_layers = len(self.out_channels_list)
        self.out_stride = out_stride

        if train_flag:
            for name, parameter in self.body.named_parameters():
                parameter.requires_grad_(False)

                if 'layer4' in top_layer:
                    if 'layer4' in name or 'layer3' in name or 'layer2' in name or 'layer1' in name:
                        parameter.requires_grad_(True)

                elif 'layer3' in top_layer:
                    if 'layer3' in name or 'layer2' in name or 'layer1' in name:
                        parameter.requires_grad_(True)

                elif 'layer2' in top_layer:
                    if 'layer2' in name or 'layer1' in name:
                        parameter.requires_grad_(True)

                else:
                    if 'layer1' in name:
                        parameter.requires_grad_(True)

                if train_all:
                    parameter.requires_grad_(True)
        else:
            for name, parameter in self.body.named_parameters():
                parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor):
        ys: Dict = self.body(inputs)

        # outs: Dict[str, torch.Tensor] = {}
        # for name, y in ys.items():
        #     outs[name] = y

        outs: List[torch.Tensor] = []
        for name, y in ys.items():
            outs.append(y)

        return outs


def build_backbone(_args):
    resnet = ResBackbone(arch=_args.arch, pretrain=_args.use_pretrain, zero_init_residual=_args.zero_init_res,
                         replace_stride_with_dilation=_args.dilation_list,
                         norm_layer=FrozenBatchNorm2d if _args.norm_layer is None else _args.norm_layer,
                         top_layer=_args.top_layer, return_inter_layers=_args.use_inter_layer,
                         train_flag=_args.lr_mult > 0, train_all=_args.train_all)

    return resnet


if __name__ == '__main__':
    from easydict import EasyDict as Edict

    args = Edict()

    args.backbone = Edict()
    args.backbone.arch = 'resnet50'
    args.backbone.use_pretrain = True
    args.backbone.zero_init_res = False  # True for SimSiam
    args.backbone.dilation_list = [False, False, False]  # layer2 layer3 layer4, in increasing depth order
    args.backbone.norm_layer = None  # None for frozenBN

    args.backbone.top_layer = 'layer4'
    args.backbone.use_inter_layer = True

    args.backbone.lr_backbone = 0.1
    args.backbone.train_all = (args.backbone.lr_backbone > 0) & False

    backbone = build_backbone(args.backbone)
    print(backbone)

    x = torch.rand(1, 3, 224, 224)
    ys = backbone(x)
    print([(_y[0], _y[1].shape) for _y in ys.items()])

    from torchvision.ops import FeaturePyramidNetwork
    fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048], out_channels=128)
    y = fpn(ys)
    print([(_y[0], _y[1].shape) for _y in y.items()])

    from lib.model.modules import FPNCoordEmbed
    fpn = FeaturePyramidNetwork(in_channels_list=[256, 512, 1024, 2048],
                                out_channels=128, extra_blocks=FPNCoordEmbed())
    y = fpn(ys)
    print([(_y[0], _y[1].shape) for _y in y.items()])

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(backbone, input_res=(3, 224, 224),
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  8.83 GMac
    #       - Params: 30.96 M
