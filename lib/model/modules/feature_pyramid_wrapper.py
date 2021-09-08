import copy
import torch
from torch import nn, Tensor
from typing import List, Callable


class FeaturePyramidWrapper(nn.Module):

    def __init__(self, in_channels_list, out_channels, module: Callable = None):
        super(FeaturePyramidWrapper, self).__init__()

        if module is None:
            self.ops = nn.ModuleList([
                nn.Conv2d(in_c, out_channels, kernel_size=(1, 1)) for in_c in in_channels_list
            ])
        else:
            self.ops = nn.ModuleList([
                module(in_channels=in_c, out_channels=out_channels) for in_c in in_channels_list
            ])

    def forward(self, in_tensor_list: List[Tensor]):  # (N, C, H, W)

        results = []
        for op, in_tensor in zip(self.ops, in_tensor_list):

            res = op(in_tensor)

            results.append(res)

        return results
