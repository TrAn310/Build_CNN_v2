"""
evaluator.py
Precision/Recall/F1/AP/mAP tự code.
"""

import numpy as np
import torch
from ..inference.postprocess import iou_batch, postprocess


def _gt_to_xyxy(gts):
    """gts: [M,5] = [cls, xc, yc, w, h] pixel -> [M,5] = [cls, x1, y1, x2, y2]"""
    if gts.shape[0] == 0:
        return gts
    x1 = gts[:, 1] - gts[:, 3] / 2
    y1 = gts[:, 2] - gts[:, 4] / 2
    x2 = gts[:, 1] + gts[:, 3] / 2
    y2 = gts[:, 2] + gts[:, 4] / 2
    return torch.stack([gts[:, 0], x1, y1, x2, y2], dim=1)


def compute_precision_recall(tp, fp, fn):
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return p, r, f1


def compute_ap_11point(precisions, recalls):
    """11-point interpolation AP."""
    if len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_max = 0.0
        for p, r in zip(precisions, recalls):
            if r >= t and p > p_max:
                p_max = p
        ap += p_max / 11.0
    return ap


def compute_ap_from_curve(scores, tps, num_gt):
    """
    scores: list float (score của từng prediction, đã sort)
    tps:    list int (1 nếu TP, 0 nếu FP)
    num_gt: tổng số GT của class
    """
    if num_gt == 0 or len(scores) == 0:
        return 0.0
    scores = np.array(scores)
    tps = np.array(tps)
    order = np.argsort(-scores)
    tp_cum = np.cumsum(tps[order])
    fp_cum = np.cumsum(1 - tps[order])
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / (num_gt + 1e-6)
    return compute_ap_11point(precisions.tolist(), recalls.tolist())


@torch.no_grad()
def evaluate(model, loader, device, num_classes, img_size,
             strides, conf_thresh=0.3, iou_thresh=0.5,
             class_names=None):
    """
    Trả về dict metrics tổng + per-class.
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    model.eval()
    per_class_preds = {c: [] for c in range(num_classes)}
    per_class_num_gt = {c: 0 for c in range(num_classes)}

    for imgs, targets in loader:
        imgs = imgs.to(device)
        raw = model(imgs)
        dets = postprocess(raw, strides, num_classes, img_size,
                           conf_thresh, iou_thresh)

        for b in range(imgs.shape[0]):
            gt = targets[b].clone().to(device)
            if gt.shape[0] > 0:
                gt[:, 1:] = gt[:, 1:] * img_size
                gt_xyxy = _gt_to_xyxy(gt)
            else:
                gt_xyxy = torch.zeros(0, 5, device=device)

            preds = dets[b]

            for c in range(num_classes):
                p_c = preds[preds[:, 5] == c] if preds.shape[0] > 0 else preds
                g_c = gt_xyxy[gt_xyxy[:, 0] == c] if gt_xyxy.shape[0] > 0 else gt_xyxy

                per_class_num_gt[c] += g_c.shape[0]

                # Sort predictions theo score giảm dần
                if p_c.shape[0] > 0:
                    order = p_c[:, 4].argsort(descending=True)
                    p_c = p_c[order]

                matched_gt = set()
                for p in p_c:
                    best_iou = 0.0
                    best_gi = -1
                    for gi in range(g_c.shape[0]):
                        if gi in matched_gt:
                            continue
                        iou = iou_batch(g_c[gi:gi+1, 1:], p[:4])[0].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    is_tp = 1 if (best_iou >= iou_thresh and best_gi >= 0) else 0
                    if is_tp:
                        matched_gt.add(best_gi)
                    per_class_preds[c].append((p[4].item(), is_tp))

    metrics = {}
    ap_list = []
    for c in range(num_classes):
        num_gt = per_class_num_gt[c]
        preds_c = per_class_preds[c]
        if len(preds_c) > 0:
            scores = [x[0] for x in preds_c]
            tps = [x[1] for x in preds_c]
        else:
            scores, tps = [], []

        ap = compute_ap_from_curve(scores, tps, num_gt)
        tp_total = sum(tps)
        fp_total = len(tps) - tp_total
        fn_total = num_gt - tp_total
        p, r, f1 = compute_precision_recall(tp_total, fp_total, fn_total)

        metrics[class_names[c]] = {
            'precision': p, 'recall': r, 'f1': f1, 'ap': ap,
            'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
            'num_gt': num_gt,
        }
        ap_list.append(ap)

    metrics['mAP'] = sum(ap_list) / max(len(ap_list), 1)
    return metrics


def print_metrics(metrics, class_names=None):
    print('\n' + '=' * 70)
    print(f'{"Class":<12}{"P":>8}{"R":>8}{"F1":>8}{"AP":>8}'
          f'{"TP":>6}{"FP":>6}{"FN":>6}{"GT":>6}')
    print('-' * 70)
    for k, v in metrics.items():
        if k == 'mAP':
            continue
        print(f'{k:<12}{v["precision"]:>8.3f}{v["recall"]:>8.3f}'
              f'{v["f1"]:>8.3f}{v["ap"]:>8.3f}'
              f'{v["tp"]:>6d}{v["fp"]:>6d}{v["fn"]:>6d}{v["num_gt"]:>6d}')
    print('-' * 70)
    print(f'{"mAP":<12}{"":>8}{"":>8}{"":>8}{metrics["mAP"]:>8.3f}')
    print('=' * 70 + '\n')