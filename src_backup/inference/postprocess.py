"""
postprocess.py
NMS tự code + pipeline decode -> NMS.
"""

import torch
from .decoder import decode_predictions


def iou_batch(boxes, box):
    """
    boxes: [N,4], box: [4]
    return: [N]
    """
    x1 = torch.max(boxes[:, 0], box[0])
    y1 = torch.max(boxes[:, 1], box[1])
    x2 = torch.min(boxes[:, 2], box[2])
    y2 = torch.min(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(dets, iou_thresh=0.5):
    """
    dets: [N,6] = [x1,y1,x2,y2, score, class]
    return: [M,6]
    """
    if dets.shape[0] == 0:
        return dets
    keep = []
    for c in dets[:, 5].unique():
        cls_mask = dets[:, 5] == c
        cls_dets = dets[cls_mask]
        order = cls_dets[:, 4].argsort(descending=True)
        cls_dets = cls_dets[order]
        while cls_dets.shape[0] > 0:
            best = cls_dets[0]
            keep.append(best)
            if cls_dets.shape[0] == 1:
                break
            ious = iou_batch(cls_dets[1:, :4], best[:4])
            cls_dets = cls_dets[1:][ious < iou_thresh]
    if len(keep) == 0:
        return torch.zeros(0, 6, device=dets.device)
    return torch.stack(keep, dim=0)


def postprocess(raw_outputs, strides, num_classes, img_size=640,
                conf_thresh=0.3, iou_thresh=0.5):
    decoded = decode_predictions(raw_outputs, strides, num_classes,
                                 img_size, conf_thresh)
    return [nms(d, iou_thresh) for d in decoded]
