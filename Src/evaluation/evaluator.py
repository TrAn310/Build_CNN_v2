"""
evaluator.py

BUOC 1 (Bai 9 - Evaluator):
    Detection matching -- ghep tung prediction voi ground truth box
    tuong ung (neu co) de xac dinh TP/FP. Day la nen tang de tinh
    precision/recall/AP/mAP o cac buoc sau.

BUOC 2 (lam o buoc sau, CHUA co trong file nay):
    calculate_precision_recall() (duong cong PR theo confidence ranking)
    -> calculate_ap() -> calculate_map()

Ban da biet IoU va precision/recall roi nen khong nhac lai ly thuyet do.
Phan MOI so voi nhung gi ban da hoc: trong detection, 1 ground-truth box
CHI duoc tinh la "da phat hien" (matched) DUNG 1 LAN. Neu 2 prediction
cung IoU cao voi 1 GT, chi prediction co score cao hon duoc tinh TP --
prediction con lai la FP du no khoanh dung vi tri, vi no la BAN SAO
(duplicate) cua mot phat hien da co roi, khong phai vi no khoanh sai.

CAU TRUC THU MUC: file nay o Src/evaluation/, con postprocess.py
(chua ham compute_iou) o Src/inference/ -- 2 thu muc ANH EM, nen phai
tu them Src/inference/ vao sys.path TRUOC KHI import, giong het cach
lam trong train.py.
"""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # Src/evaluation
_SRC_DIR = os.path.dirname(_CURRENT_DIR)                     # Src/

_INFERENCE_DIR = os.path.normpath(os.path.join(_SRC_DIR, "inference"))
if _INFERENCE_DIR not in sys.path:
    sys.path.append(_INFERENCE_DIR)

import torch

from postprocess import compute_iou   # tai su dung ham IoU da viet o Bai 8


def match_predictions_to_ground_truth(pred_boxes, pred_scores, pred_classes,
                                        gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Ghep prediction (cua 1 ANH) voi ground truth (cua CUNG anh do).

    Args:
        pred_boxes:   [N,4]  box SAU NMS (output tu postprocess.py)
        pred_scores:  [N]
        pred_classes: [N]
        gt_boxes:     [M,4]  box that (tu annotation)
        gt_classes:   [M]
        iou_threshold: IoU toi thieu de tinh la match dung

    Returns:
        order:  index cua pred, sap xep theo SCORE GIAM DAN.
        is_tp:  bool tensor [N], THEO DUNG THU TU cua `order`.
        num_gt: int, tong so ground-truth box (mau so cua Recall).
    """
    num_gt = gt_boxes.shape[0]

    order = torch.argsort(pred_scores, descending=True)

    matched_gt = torch.zeros(num_gt, dtype=torch.bool)
    is_tp = torch.zeros(len(order), dtype=torch.bool)

    for i, pred_idx in enumerate(order.tolist()):
        pred_box = pred_boxes[pred_idx]
        pred_cls = pred_classes[pred_idx].item()

        candidate_mask = (gt_classes == pred_cls) & (~matched_gt)

        if candidate_mask.sum() == 0:
            continue

        candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
        ious = compute_iou(pred_box, gt_boxes[candidate_idx])
        best_iou, best_local = ious.max(dim=0)

        if best_iou.item() >= iou_threshold:
            gt_idx = candidate_idx[best_local].item()
            matched_gt[gt_idx] = True
            is_tp[i] = True

    return order, is_tp, num_gt


def summarize(is_tp, num_gt):
    """
    Ham phu de xem nhanh ket qua Buoc 1 bang cach CONG DON tat ca
    prediction lai thanh 1 con so Precision/Recall duy nhat.
    """
    tp = is_tp.sum().item()
    fp = (~is_tp).sum().item()
    fn = num_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall}


if __name__ == "__main__":
    gt_boxes = torch.tensor([
        [100., 100., 200., 200.],  # GT1
        [400., 400., 500., 500.],  # GT2
        [700., 700., 800., 800.],  # GT3 -- se bi MISS
    ])
    gt_classes = torch.tensor([0, 0, 0])

    pred_boxes = torch.tensor([
        [105., 105., 205., 205.],  # P1 -> gan GT1
        [110., 110., 210., 210.],  # P2 -> cung gan GT1 (duplicate cua P1)
        [408., 408., 508., 508.],  # P3 -> gan GT2
        [250., 250., 350., 350.],  # P4 -> khong gan GT nao
    ])
    pred_scores = torch.tensor([0.95, 0.85, 0.90, 0.60])
    pred_classes = torch.tensor([0, 0, 0, 0])

    order, is_tp, num_gt = match_predictions_to_ground_truth(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5
    )

    print("Thu tu xu ly (confidence ranking), index prediction:", order.tolist())
    print("Score tuong ung                                    :", pred_scores[order].tolist())
    print("Ket qua TP/FP theo dung thu tu tren                :", is_tp.tolist())

    result = summarize(is_tp, num_gt)
    print("\nTong hop:", result)

    print("\nExpected: is_tp = [True, True, False, False]")
    print("Expected TP=2, FP=2, FN=1, precision=0.5, recall=0.667")