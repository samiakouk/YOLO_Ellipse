import torch
import config
from torch import nn as nn
from torch.nn import functional as F
from utils import get_yolo_iou

BOX_CONF = 0
BOX_X = 1
BOX_Y = 2
BOX_W = 3
BOX_H = 4
BOX_SIN = 5


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::6]

class SumSquaredErrorLoss(nn.Module):
    def __init__(self, l_coord=5, l_noobj=0.5):
        super().__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, p, a):
        iou = get_yolo_iou(p, a)                     # (batch, S, S, B)
        best_box = torch.argmax(iou, dim=-1, keepdim=True).long()

        bbox_mask = (bbox_attr(a, BOX_CONF) > 0.0).float()
        p_template = bbox_attr(p, BOX_CONF)

        responsible = torch.zeros_like(p_template)
        responsible.scatter_(-1, best_box, 1.0)

        obj_i = bbox_mask[..., 0:1]         # 1 if grid I has any object at all
        obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
        noobj_ij = 1.0 - obj_ij             # Otherwise, confidence should be 0

        # XY position losses
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, BOX_X),
            obj_ij * bbox_attr(a, BOX_X)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, BOX_Y),
            obj_ij * bbox_attr(a, BOX_Y)
        )
        pos_losses = x_losses + y_losses
        # print('pos_losses', pos_losses.item())

        # Bbox dimension losses
        p_width = bbox_attr(p, BOX_W)
        a_width = bbox_attr(a, BOX_W)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + config.EPSILON),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, BOX_H)
        a_height = bbox_attr(a, BOX_H)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + config.EPSILON),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses
        # print('dim_losses', dim_losses.item())

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, BOX_CONF),
            obj_ij * torch.ones_like(bbox_attr(p, BOX_CONF))
        )
        # print('obj_confidence_losses', obj_confidence_losses.item())
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, BOX_CONF),
            torch.zeros_like(bbox_attr(p, BOX_CONF))
        )
        # print('noobj_confidence_losses', noobj_confidence_losses.item())

        # Classification losses
        class_losses = mse_loss(
            obj_i * p[..., :config.C],
            obj_i * a[..., :config.C]
        )
        # print('class_losses', class_losses.item())

        # Angle Losses
        angle_losses = mse_loss(
            obj_ij * bbox_attr(p, BOX_SIN),
            obj_ij * bbox_attr(a, BOX_SIN)
        )
        # print('angle_losses', angle_losses.item())

        total = self.l_coord * (pos_losses + dim_losses + angle_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses \
                
        # Divide by the actual batch size so the loss remains correctly scaled
        # when training uses a smaller GPU-safe batch or a partial final batch.
        return total / p.shape[0]


def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )
