import importlib
from functools import partial
from typing import List, Union, Optional, Dict, Any, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops import box_convert, clip_boxes_to_image, box_iou


class MultimodalModel(nn.Module):

    def __init__(self, args):
        super(MultimodalModel, self).__init__()

        self.pretrained_param = None

        self.register_buffer("pytorch_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))  # RGB
        self.register_buffer("pytorch_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))  # RGB

        self.cfg = args
        self.num_tasks = self.cfg.neck.num_tasks
        self.output_sz = self.cfg.head.output_size
        self.use_language = self.cfg.use_language

        # build backbone
        backbone_module = importlib.import_module('lib.model.backbones')
        self.backbone = getattr(backbone_module, self.cfg.backbone.type)(self.cfg.backbone)

        # build neck
        neck_module = importlib.import_module('lib.model.necks')
        self.cfg.neck.in_channels_list = [c for c in self.backbone.out_channels_list]
        self.neck = getattr(neck_module, self.cfg.neck.type)(self.cfg.neck)

        # build multi-head
        head_module = importlib.import_module('lib.model.heads')
        self.heads = nn.ModuleList([
            getattr(head_module, tp)(self.cfg.head) for tp in self.cfg.head.type])

        # build criterion
        criteria_module = importlib.import_module('lib.model.criteria')
        self.criteria = [
            partial(getattr(criteria_module, tp),
                    alpha_giou=self.cfg.criterion.alpha_giou,
                    alpha_l1=self.cfg.criterion.alpha_l1,
                    alpha_conf=self.cfg.criterion.alpha_conf
                    ) for tp in self.cfg.criterion.type]

    def forward(self, input_dict: Dict[str, Union[Tensor, Any]]):

        device = next(self.parameters()).device
        training: bool = input_dict['training']

        img_t: Tensor = input_dict['template'].to(device)
        img_s: Tensor = input_dict['search'].to(device)

        target_box: Optional[Tensor] = input_dict['target'].to(device)
        lang: Optional[Tensor] = input_dict.get('language', None)
        lang_mask: Optional[Tensor] = input_dict.get('language_mask', None)

        ns, _, hs, ws = img_s.shape

        t_features: List[Tensor] = self.backbone(self._pytorch_norm(img_t))  # (N, C, H, W)
        s_features: List[Tensor] = self.backbone(self._pytorch_norm(img_s))  # (N, C, H, W)

        task_features: List[Tuple] = self.neck(template_features=t_features,
                                               search_features=s_features,
                                               lang=lang, lang_mask=lang_mask)

        predictions = [self.heads[0](*task_features[0]),  # visual feature
                       self.heads[0](*task_features[1])]  # cross modal feature

        # Aggregation
        weights = torch.cat((predictions[0][1], predictions[1][1]), dim=1).softmax(dim=1)  # (N, 2)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, 2, 1, 1, 1)

        prediction = self.heads[0](task_features[0][0] * weights[:, 0] + task_features[1][0] * weights[:, 1],
                                   task_features[0][1] * weights[:, 0] + task_features[1][1] * weights[:, 1])

        targets = [target_box, self._compute_iou(prediction[0], target_box.detach())]
        losses, metrics = self.criteria[0](prediction, targets)

        if not training:
            return predictions

        else:

            total_loss = losses[0]

            loss_dict = dict()
            loss_dict.update({'giou': losses[1].item()})
            loss_dict.update({'l1': losses[2].item()})
            loss_dict.update({'conf': losses[3].item()})

            metric_dict = dict()
            metric_dict.update({'miou': metrics[0].item()})

            return total_loss, [loss_dict, metric_dict]

    def init(self, img_t):

        t_features: List[Tensor] = self.backbone(self._pytorch_norm(img_t))  # (N, C, H, W)

        return t_features

    def track(self,
              img_s: Tensor,  # (N, C, H, W)
              t_features: Union[Tensor, List],  # (N, C, H, W)
              bert: Optional[Tuple[Tensor]] = None,  # (1, N, 768) (1, N, 768)
              ):
        ns, _, hs, ws = img_s.shape

        s_features: List[Tensor] = self.backbone(self._pytorch_norm(img_s))  # (N, C, H, W)

        task_features: List[Tuple] = self.neck(template_features=t_features,
                                               search_features=s_features,
                                               lang=bert[0], lang_mask=bert[1])

        # Aggregation
        predictions = [self.heads[0](*task_features[0]),
                       self.heads[0](*task_features[1])]

        weights = torch.cat((predictions[0][1], predictions[1][1]), dim=1).softmax(dim=1)  # (N, 2)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, 2, 1, 1, 1)

        prediction = self.heads[0](task_features[0][0] * weights[:, 0] + task_features[1][0] * weights[:, 1],
                                   task_features[0][1] * weights[:, 0] + task_features[1][1] * weights[:, 1])

        # # Static Aggregation
        # prediction = self.heads[0](task_features[0][0] * 0.5 + task_features[1][0] * 0.5,
        #                            task_features[0][1] * 0.5 + task_features[1][1] * 0.5)

        # # Visual feature
        # prediction = self.heads[0](task_features[0][0],
        #                            task_features[0][1])

        # # Cross feature
        # prediction = self.heads[0](task_features[1][0],
        #                            task_features[1][1])

        # -------------------------------------------------------------------
        pred_box = prediction[0]
        pred_conf = prediction[1]

        outputs_coord = box_convert(pred_box, in_fmt='xyxy', out_fmt='cxcywh')

        pred_dict = dict()
        pred_dict['box'] = outputs_coord.squeeze().detach().cpu().numpy()
        pred_dict['score'] = pred_conf.item()

        return pred_dict

    def _pytorch_norm(self, img):
        img = img.div(255.0)
        img = img.sub(self.pytorch_mean).div(self.pytorch_std)
        return img

    @staticmethod
    def _compute_iou(pred_box: Tensor, target_box: Tensor):
        """

        Args:
            pred_box: Tensor (N, 4) normalized box [[x1, y1, x2, y2]]
            target_box: Tensor (N, 4) normalized box [[x1, y1, x2, y2]]

        Returns:
            iou: Tensor (N, 1)
        """
        pred_box = clip_boxes_to_image(pred_box, size=(1, 1))
        pred_iou = box_iou(pred_box, target_box)
        iou = torch.diag(pred_iou)

        return iou.reshape(-1, 1)


def build_model(args):
    model = MultimodalModel(args)
    # if args.pretrain is not None:
    #     load_pretrain(model, args.pretrain)
    return model


if __name__ == '__main__':
    from config.cfg_multimodal import cfg as settings

    net = build_model(settings.model)

    t = torch.ones(2, 3, 128, 128) * 0.3
    x = torch.ones(2, 3, 256, 256) * 0.3
    gt = torch.rand(2, 4)
    l = torch.rand(2, 10, 768)

    in_dict = {
        'template': t,
        'search': x,
        'target': gt,
        'language': l,
        'lang_mask': torch.rand(2, 10, 1),
        'training': True,
    }

    out = net(in_dict)
    print(out)
    out = net(in_dict)
    print(out)

    from ptflops import get_model_complexity_info


    def prepare_input(resolution):
        x1 = torch.rand(*resolution[0])
        x2 = torch.rand(*resolution[1])
        x3 = torch.rand(*resolution[2])
        x4 = torch.rand(1, 768)

        input_dict = {
            'template': t,
            'search': x,
            'target': gt,
            'language': l,
            'lang_mask': torch.rand(2, 10, 1),
            'training': True,
        }

        return dict(input_dict=input_dict)


    flops, params = get_model_complexity_info(net,
                                              input_res=((1, 3, 128, 128), (1, 3, 256, 256), (1, 4)),
                                              input_constructor=prepare_input,
                                              as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)
    #       - Flops:  7.22 GMac
    #       - Params: 23.63 M
