import torch
import torch.nn as nn
from torchvision.ops import roi_align
from typing import List


class ExtremeV3(nn.Module):
    def __init__(self, args):
        super(ExtremeV3, self).__init__()

        self.cfg = args
        self.in_channels = self.cfg.in_channels
        self.inter_channels = self.cfg.inter_channels
        self.output_size_w, self.output_size_h = self.cfg.output_size

        beta_x = torch.arange(self.output_size_w).reshape(1, -1).repeat(self.output_size_w, 1).reshape(1, -1)
        beta_y = torch.arange(self.output_size_h).reshape(-1, 1).repeat(1, self.output_size_h).reshape(1, -1)

        self.register_buffer("beta_x", beta_x)
        self.register_buffer("beta_y", beta_y)

        self.coord_branch = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, 4, kernel_size=(3, 3), padding=(1, 1)),
            )

        self.conf_branch = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, 5, kernel_size=(3, 3), padding=(1, 1)),
            )

    def forward(self, att_coord_map: torch.Tensor, att_conf_map: torch.Tensor):
        """

        Args:
            att_coord_map: Tensor (N, C, out_sz, out_sz)
            att_conf_map: Tensor (N, C, out_sz, out_sz)
        Returns:
            pred_box: Tensor (N, 1)
            pred_conf: Tensor (N, 1)
        """
        assert list(att_coord_map.shape[2:]) == self.cfg.output_size, \
            "shape of input feature map != output_size of config"
        assert list(att_conf_map.shape[2:]) == self.cfg.output_size, \
            "shape of input feature map != output_size of config"

        coord_map = self.coord_branch(att_coord_map)
        conf_map = self.conf_branch(att_conf_map)

        pred_box, point_list = self.predict_box(coord_map)
        pred_conf = self.predict_conf(conf_map, point_list, pred_box)

        return [pred_box, pred_conf]

    def predict_box(self, coord_map: torch.Tensor):
        """

        Args:
            coord_map: Tensor (N, 4, H, W) outputs map

        Returns:

        """

        point1 = self.soft_argmax(coord_map[:, 0])  # (N, 2) left
        point2 = self.soft_argmax(coord_map[:, 1])  # (N, 2) top
        point3 = self.soft_argmax(coord_map[:, 2])  # (N, 2) right
        point4 = self.soft_argmax(coord_map[:, 3])  # (N, 2) bootom

        x_min = point1[:, :1]
        y_min = point2[:, 1:]
        x_max = point3[:, :1]
        y_max = point4[:, 1:]

        outputs_coord = torch.cat((x_min, y_min, x_max, y_max), dim=1)  # (N, 4)

        return outputs_coord, [point1, point2, point3, point4]  # (N, 2) [[x y x y]]

    def predict_conf(self, conf_map: torch.Tensor, coord_list: List[torch.Tensor], pred_box: torch.Tensor):
        """

        Args:
            conf_map: Tensor (N, 4, H, W) outputs map
            coord_list: Tensor (N, 2) normalized [[x y]]
            pred_box: Tensor (N, 4) normalized [[x y]]

        Returns:

        """

        conf1 = self.point2conf(coord_list[0], conf_map[:, 0:1])
        conf2 = self.point2conf(coord_list[1], conf_map[:, 1:2])
        conf3 = self.point2conf(coord_list[2], conf_map[:, 2:3])
        conf4 = self.point2conf(coord_list[3], conf_map[:, 3:4])

        conf5 = self.map2semantic_conf(pred_box, conf_map[:, 4:5])

        # default
        conf = conf1 * conf2 * conf3 * conf4 * conf5

        # # extreme
        # conf = conf1 * conf2 * conf3 * conf4
        #
        # # target
        # conf = conf5

        return conf  # (N, 1)

    def soft_argmax(self, coord_map: torch.Tensor):  # (N, H, W)
        """

        Args:
            coord_map: Tensor (N, H, W) outputs map

        Returns:
            coord: Tensor (N, 2) normalized [[x y]]

        """
        assert self.output_size_w == coord_map.shape[2], "output_size_w != w of coord_map"
        assert self.output_size_h == coord_map.shape[1], "output_size_h != h of coord_map"

        _map = torch.softmax(coord_map.flatten(1), dim=-1)

        x = (_map * self.beta_x).sum(-1, keepdim=True) / self.output_size_w
        y = (_map * self.beta_y).sum(-1, keepdim=True) / self.output_size_h

        coord = torch.cat((x, y), dim=1)

        return coord  # (N, 2)

    def point2conf(self, point: torch.Tensor, var_map: torch.Tensor):  # (N, 1, H, W)
        """

        Args:
            point: Tensor (N, 2) normalized [x y]
            var_map: Tensor (N, 1, H, W) outputs map

        Returns:
            conf: Tensor (N, 1)

        """
        ns, cs, hs, ws = var_map.shape

        var_map = torch.softmax(var_map.flatten(2), dim=-1)
        var_map = var_map.reshape(ns, cs, hs, ws)

        px = point[:, 0] * self.output_size_w
        py = point[:, 1] * self.output_size_h

        x1 = px - 1
        y1 = py - 1
        x2 = px + 1
        y2 = py + 1

        rois = torch.stack((torch.arange(ns, dtype=torch.int32, device=x1.device).float(),
                            x1, y1, x2, y2), dim=1)

        conf = roi_align(var_map, rois.detach(), output_size=3, sampling_ratio=0).sum((2, 3))

        return conf  # (N, 1)

    def map2semantic_conf(self, box, var_map):  # (B, 4)
        ns, cs, hs, ws = var_map.shape

        var_map = torch.softmax(var_map.flatten(2), dim=-1)
        var_map = var_map.reshape(ns, cs, hs, ws)

        x1 = box[:, 0] * self.output_size_w  # not use shape tensor -- ws
        y1 = box[:, 1] * self.output_size_h
        x2 = box[:, 2] * self.output_size_w
        y2 = box[:, 3] * self.output_size_h

        rois = torch.stack((torch.arange(ns, dtype=torch.int32, device=x1.device).float(),
                            x1, y1, x2, y2), dim=1)  # (B, 5)

        conf = roi_align(var_map, rois.detach(), output_size=7, sampling_ratio=0).sum((2, 3))

        return conf


if __name__ == '__main__':
    from easydict import EasyDict as Edict

    cfg = Edict()

    cfg.type = 'Extreme'
    cfg.in_channels = 256
    cfg.inter_channels = 256
    cfg.output_size = [16, 16]

    head = ExtremeV3(cfg)

    t = torch.ones(2, 256, 16, 16)
    t = head(t, t)

    [print(tt.shape) for tt in t]
