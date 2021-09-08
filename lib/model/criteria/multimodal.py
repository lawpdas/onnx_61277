import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou, generalized_box_iou
import math
from typing import List


def multimodal_criterion(predictions: List[torch.Tensor], targets: List[torch.Tensor],
                   alpha_giou=None, alpha_l1=None, alpha_conf=None):
    assert len(predictions) == 2, "predictions: must be normalized [pred_box, pred_conf]"
    assert len(targets) == 2, "targets: must be normalized [target_box, target_iou, loss_mask]"

    pred_box, pred_conf = predictions
    target_box, target_iou = targets

    loss_giou, loss_l1, iou = loss_box(pred_box, target_box.detach(), reduction='mean')
    loss_conf = F.mse_loss(pred_conf.reshape(-1), target_iou.reshape(-1).detach(), reduction='mean')

    loss = (alpha_giou * loss_giou
            + alpha_l1 * loss_l1
            + alpha_conf * loss_conf)

    return [loss, loss_giou, loss_l1, loss_conf], [iou.mean()]


def loss_box(pred, target, reduction='mean'):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    params:
        pred_box: [B 4]  [x y x y]
        target_box: [B 4]  [x y x y]
    return:
        loss_giou
        loss_bbox
    """

    if reduction == 'mean':
        loss_l1 = F.l1_loss(pred, target, reduction='mean')

        try:
            loss_iou = (1 - torch.diag(generalized_box_iou(pred, target))).mean()
            miou = torch.diag(box_iou(pred, target))
        except:
            loss_iou, miou = torch.tensor(0.0).to(pred.device), torch.zeros(pred.shape[0]).to(pred.device)

    else:
        loss_l1 = F.l1_loss(pred, target, reduction='none')

        try:
            loss_iou = (1 - torch.diag(generalized_box_iou(pred, target)))
            miou = torch.diag(box_iou(pred, target))
        except:
            loss_iou, miou = torch.zeros(pred.shape[0]).to(pred.device), torch.zeros(pred.shape[0]).to(pred.device)

    return loss_iou, loss_l1, miou
