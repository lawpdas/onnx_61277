import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List

from lib.model.modules import FeaturePyramidWrapper
from lib.model.modules import build_trans_encoder
from lib.model.modules.position_encoding import PositionEmbeddingSine


class ModalFusion(nn.Module):
    def __init__(self, args):
        super(ModalFusion, self).__init__()

        self.cfg = args
        self.in_channels_list = self.cfg.in_channels_list
        self.inter_channels = self.cfg.inter_channels
        self.num_tasks = self.cfg.num_tasks
        self.use_language = self.cfg.use_language
        self.search_token_length = np.prod(self.cfg.output_size).astype(np.int32)

        # assert args.backbone.use_inter_layer is True, "FPN requires intermediate feature map"
        # self.fpn = FeaturePyramidNetwork(in_channels_list=self.cfg.in_channels_list,
        #                                  out_channels=self.inter_channels)

        self.projector = FeaturePyramidWrapper(in_channels_list=self.in_channels_list,
                                               out_channels=self.inter_channels,
                                               module=None)

        self.pos_embedding = PositionEmbeddingSine(self.inter_channels // 2, normalize=True)

        self.ling_proj_pre = nn.Sequential(
            nn.Linear(768, self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_channels, self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_channels, self.inter_channels),
        )

        self.task_token = nn.Parameter(torch.rand(self.num_tasks, 1, self.inter_channels))
        self.task_token_pos = nn.Parameter(torch.rand(self.num_tasks, 1, self.inter_channels))

        assert self.cfg.transformer.num_encoders > 0, "num_encoders <= 0"
        self.trans_encoder = build_trans_encoder(self.cfg.transformer)

    def forward(self,
                template_features: List[torch.Tensor],
                search_features: List[torch.Tensor],
                lang: Optional[torch.Tensor] = None,
                lang_mask: Optional[torch.Tensor] = None,
                **kwargs):
        """

        Args:
            template_features: List[Tensor] (N, C, H, W)
            search_features: List[Tensor] (N, C, H, W)
            lang: Tensor  # (N, L, 768)
            lang_mask: Tensor  # (N, L, 1)

        Returns:
            task_feature_list: List[Tensor] (N, C, H, W)

        """
        t_feats, t_tokens, t_poses = self.parse_backbone_feature(template_features)
        s_feats, s_tokens, s_poses = self.parse_backbone_feature(search_features)

        # use last layer
        t_feat, t_token, t_pos = map(lambda ll: ll[-1], [t_feats, t_tokens, t_poses])
        s_feat, s_token, s_pos = map(lambda ll: ll[-1], [s_feats, s_tokens, s_poses])

        ns, _, hs, ws = s_feat.shape

        encode = self.trans_encoder(
            torch.cat((self.task_token.expand(-1, ns, -1), t_token, s_token), dim=0),
            pos=torch.cat((self.task_token_pos, t_pos, s_pos), dim=0))  # (*, N, C)

        kernels = encode[:self.num_tasks]  # (T, N, C)

        s_encode = encode[-self.search_token_length:]  # (HW, N, C)
        s_encode = s_encode.permute(1, 2, 0).reshape_as(s_feat)  # (N, C, H, W)

        task_feat1 = self.token_kernel_conv(s_encode, kernels[0], s_feat)
        task_feat2 = self.token_kernel_conv(s_encode, kernels[1], s_feat)

        # modulate language feature
        ling_embedding_pre = self.ling_proj_pre(lang)  # (N, L, C)
        modulate_vector2 = (ling_embedding_pre * kernels[2].unsqueeze(1)).sum(dim=1)  # (N, C)
        modulate_vector3 = (ling_embedding_pre * kernels[3].unsqueeze(1)).sum(dim=1)

        task_feat3 = self.token_kernel_conv(s_encode, modulate_vector2, s_feat)
        task_feat4 = self.token_kernel_conv(s_encode, modulate_vector3, s_feat)

        return [(task_feat1, task_feat2), (task_feat3, task_feat4)]

    def parse_backbone_feature(self, backbone_features: List[torch.Tensor]):
        """

        Args:
            backbone_features: List[Tensor] {'layer1': (N, C, H, W), ... , 'layer4': (N, C, H, W)}

        Returns:
            feat_maps: List[Tensor] (N, C, H, W)
            feat_tokens: List[Tensor] (HW, N, C)
            cosine_poses: List[Tensor] (HW, 1, C)

        """
        feat_maps: list = backbone_features  # (N, C, H, W)
        feat_maps: list = self.projector(feat_maps)  # (N, C, H, W)
        cosine_poses: list = [self.pos_embedding(v) for v in feat_maps]  # (L, N, C)

        feat_tokens: list = [v.flatten(2).permute(2, 0, 1) for v in feat_maps]  # (L, N, C)

        return feat_maps, feat_tokens, cosine_poses

    @staticmethod
    def token_kernel_conv(encode_feature, kernel, feature, reduction='none'):
        """

        Args:
            encode_feature: Tensor (N, C, H, W)
            kernel: Tensor (N, C)
            feature: Tensor (N, C, H, W)
            reduction: String
        Returns:

        """
        ns, cs, hs, ws = encode_feature.shape

        kernel = kernel.reshape(ns, cs, 1, 1)

        if reduction == 'none':
            encode_feature = (encode_feature * kernel)  # (N, C, H, W)
        else:
            encode_feature = (encode_feature * kernel).sum(dim=1, keepdim=True)  # (N, 1, H, W)

        return encode_feature * feature


if __name__ == '__main__':
    from config.cfg_multimodal import cfg

    cfg.model.neck.in_channels_list = [1024]
    neck = ModalFusion(cfg.model.neck)

    t = torch.ones(2, 1024, 8, 8)
    s = torch.ones(2, 1024, 16, 16)
    l = torch.ones(2, 54, 768)
    l[0, ...] = 0
    m = torch.ones(2, 54, 1)
    m[0, 5:] = 0
    m[1, 15:] = 0

    t = neck([t], [s], l, m)

    [print(tt[0].shape) for tt in t]
