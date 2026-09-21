"""
losses.py
Detection loss = Box (CIoU) + Objectness (BCE weighted) + Class (Focal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             weight=self.class_weights)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.sum()


def ciou_loss_from_offsets(pred_off, target_off, eps=1e-7):
    """CIoU loss trên offset (tx, ty, tw, th) — cùng cell."""
    px1 = pred_off[:, 0] - torch.exp(pred_off[:, 2]) / 2
    py1 = pred_off[:, 1] - torch.exp(pred_off[:, 3]) / 2
    px2 = pred_off[:, 0] + torch.exp(pred_off[:, 2]) / 2
    py2 = pred_off[:, 1] + torch.exp(pred_off[:, 3]) / 2

    tx1 = target_off[:, 0] - torch.exp(target_off[:, 2]) / 2
    ty1 = target_off[:, 1] - torch.exp(target_off[:, 3]) / 2
    tx2 = target_off[:, 0] + torch.exp(target_off[:, 2]) / 2
    ty2 = target_off[:, 1] + torch.exp(target_off[:, 3]) / 2

    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    c_area = (cx2 - cx1) * (cy2 - cy1) + eps

    rho2 = (pred_off[:, 0] - target_off[:, 0]) ** 2 + \
           (pred_off[:, 1] - target_off[:, 1]) ** 2

    w_p = torch.exp(pred_off[:, 2]); h_p = torch.exp(pred_off[:, 3])
    w_t = torch.exp(target_off[:, 2]); h_t = torch.exp(target_off[:, 3])
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w_t / (h_t + eps)) - torch.atan(w_p / (h_p + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c_area - alpha * v
    return (1 - ciou).mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=10,
                 lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 pos_weight=10.0, neg_weight=0.05):   # ← ĐÃ FIX
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = FocalLoss(alpha=0.75, gamma=2.0)

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

            # ---- Box loss (CIoU) ----
            if pos_mask.sum() > 0:
                pb = p_box.permute(0, 2, 3, 1)[pos_mask]
                tb = t_box.permute(0, 2, 3, 1)[pos_mask]
                total_box = total_box + ciou_loss_from_offsets(pb, tb) * pos_mask.sum()
                n_pos += pos_mask.sum().item()

            # ---- Objectness loss ----
            obj_loss_map = self.bce(p_obj, t_obj)
            w = torch.where(pos_mask,
                            torch.full_like(obj_loss_map, self.pos_weight),
                            torch.full_like(obj_loss_map, self.neg_weight))
            total_obj = total_obj + (obj_loss_map * w).mean()

            # ---- Class loss (Focal) ----
            if pos_mask.sum() > 0:
                pc = p_cls.permute(0, 2, 3, 1)[pos_mask]
                tc = t_cls.permute(0, 2, 3, 1)[pos_mask]
                tc_idx = tc.argmax(dim=1)
                total_cls = total_cls + self.ce(pc, tc_idx)

        n_pos = max(n_pos, 1)

        box_loss = total_box / n_pos
        obj_loss = total_obj   # ← ĐÃ FIX: bỏ /n_scales
        cls_loss = total_cls / n_pos

        total = (self.lambda_box * box_loss +
                 self.lambda_obj * obj_loss +
                 self.lambda_cls * cls_loss)
        return total, box_loss, obj_loss, cls_loss