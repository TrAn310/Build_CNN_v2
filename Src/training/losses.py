"""
losses.py
Detection loss = Box (Smooth L1) + Objectness (BCE) + Class (CE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3,
                 lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 pos_weight=1.0, neg_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, preds, targets):
        device = preds[0].device
        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        n_pos = 0

        for pred, tgt in zip(preds, targets):
            p_box = pred[:, :4]
            p_obj = pred[:, 4]
            p_cls = pred[:, 5:]

            t_box = tgt[:, :4]
            t_obj = tgt[:, 4]
            t_cls = tgt[:, 5:]

            pos_mask = t_obj > 0.5

            # Box loss (chỉ positive)
            if pos_mask.sum() > 0:
                pb = p_box.permute(0, 2, 3, 1)[pos_mask]
                tb = t_box.permute(0, 2, 3, 1)[pos_mask]
                total_box = total_box + F.smooth_l1_loss(pb, tb, reduction='sum')
                n_pos += pos_mask.sum().item()

            # Objectness loss (tất cả cells, weight imbalance)
            obj_loss_map = self.bce(p_obj, t_obj)
            w = torch.where(pos_mask,
                            torch.full_like(obj_loss_map, self.pos_weight),
                            torch.full_like(obj_loss_map, self.neg_weight))
            total_obj = total_obj + (obj_loss_map * w).sum()

            # Class loss (chỉ positive)
            if pos_mask.sum() > 0:
                pc = p_cls.permute(0, 2, 3, 1)[pos_mask]
                tc = t_cls.permute(0, 2, 3, 1)[pos_mask]
                tc_idx = tc.argmax(dim=1)
                total_cls = total_cls + self.ce(pc, tc_idx)

        n_pos = max(n_pos, 1)
        box_loss = total_box / n_pos
        obj_loss = total_obj / (n_pos * 100.0)
        cls_loss = total_cls / n_pos
        total = (self.lambda_box * box_loss +
                 self.lambda_obj * obj_loss +
                 self.lambda_cls * cls_loss)
        return total, box_loss, obj_loss, cls_loss
    