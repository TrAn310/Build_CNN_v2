"""
postprocess.py
NMS tự code + pipeline decode -> NMS.
Đã fix: clamp tọa độ, lọc box invalid, NMS class-aware, max detections.
"""

import torch
from Src.inference.decoder import decode_predictions


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
    area_a = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
             (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_b = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(dets, iou_thresh=0.5, img_size=640, min_box_size=2):
    """
    dets: [N,6] = [x1,y1,x2,y2, score, class]
    return: [M,6]
    """
    if dets.shape[0] == 0:
        return dets

    # ---- 1. Clamp tọa độ ----
    dets = dets.clone()
    dets[:, 0] = dets[:, 0].clamp(0, img_size)
    dets[:, 1] = dets[:, 1].clamp(0, img_size)
    dets[:, 2] = dets[:, 2].clamp(0, img_size)
    dets[:, 3] = dets[:, 3].clamp(0, img_size)

    # ---- 2. Lọc box quá nhỏ ----
    w = dets[:, 2] - dets[:, 0]
    h = dets[:, 3] - dets[:, 1]
    valid = (w > min_box_size) & (h > min_box_size)
    dets = dets[valid]

    if dets.shape[0] == 0:
        return dets

    # ---- 3. NMS class-aware ----
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
                conf_thresh=0.3, iou_thresh=0.5,
                max_detections=300, obj_thresh=0.5):
    """
    Pipeline: decode -> NMS -> giới hạn số box.
    """
    decoded = decode_predictions(raw_outputs, strides, num_classes,
                                 img_size, conf_thresh, obj_thresh)
    results = []
    for d in decoded:
        d_nms = nms(d, iou_thresh, img_size)
        if d_nms.shape[0] > max_detections:
            order = d_nms[:, 4].argsort(descending=True)
            d_nms = d_nms[order[:max_detections]]
        results.append(d_nms)
    return results