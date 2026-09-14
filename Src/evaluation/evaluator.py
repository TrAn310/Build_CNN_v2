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


# ======================================================================
# BUOC 2 - Precision/Recall curve -> AP -> mAP
#
# Khac voi summarize() o tren (cong don TP/FP thanh 1 con so duy nhat),
# o day ta can DUONG CONG precision/recall theo tung nguong confidence
# (moi diem tren duong cong ung voi "neu chi lay N prediction diem cao
# nhat"), roi lay DIEN TICH duoi duong cong do lam AP (Average Precision)
# cho 1 class. mAP = trung binh AP cua tat ca class.
# ======================================================================

def calculate_precision_recall(is_tp_sorted, num_gt):
    """
    Tinh duong cong precision/recall TU 1 DANH SACH is_tp DA SAP XEP
    SAN theo confidence GIAM DAN (khac voi summarize(): o day tinh
    CUMULATIVE tai tung vi tri, khong chi 1 con so tong).

    Args:
        is_tp_sorted: bool tensor [N], da sap theo score giam dan
        num_gt: int, tong so ground-truth box (mau so cua recall)

    Returns:
        precisions: tensor [N]
        recalls:    tensor [N]
    """
    tp_cum = torch.cumsum(is_tp_sorted.float(), dim=0)
    fp_cum = torch.cumsum((~is_tp_sorted).float(), dim=0)

    recalls = tp_cum / max(num_gt, 1)
    precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)

    return precisions, recalls


def calculate_ap(precisions, recalls):
    """
    AP = dien tich duoi duong cong precision-recall, dung phuong phap
    "all-point interpolation" (giong COCO/VOC2012): tai moi muc recall,
    precision duoc thay bang MAX precision cua moi muc recall >= no
    (lam duong cong don dieu giam -> khu dao dong zic-zac tu nhien cua
    precision khi them dan prediction).

    Args:
        precisions, recalls: list hoac 1D array cung do dai, DA sap
            theo confidence giam dan (recall tang dan)

    Returns:
        ap: float
    """
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Lam precision don dieu giam tu phai sang trai
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Cong dien tich cac hinh chu nhat tai moi cho recall THAY DOI
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def calculate_map(all_predictions, all_ground_truths, num_classes, iou_threshold=0.5):
    """
    Tinh mAP tren CA TAP DU LIEU (nhieu anh), khong phai 1 anh don le.

    Buoc quan trong khac voi match_predictions_to_ground_truth() (chi
    xu ly 1 anh): o day phai GOP prediction CUNG 1 class TU TAT CA anh
    lai, sap xep chung theo confidence, roi moi tinh precision/recall -
    khong duoc tinh AP rieng tung anh roi trung binh (sai ve mat toan
    hoc, se ra ket qua khac AP chuan).

    Args:
        all_predictions: list, moi phan tu la tuple (boxes, scores, classes)
            cho 1 anh - DUNG output cua postprocess.decode_and_filter()
        all_ground_truths: list, moi phan tu la tuple (gt_boxes, gt_classes)
            cho CUNG anh do (dung thu tu voi all_predictions)
        num_classes: int
        iou_threshold: nguong IoU de tinh la match dung (mac dinh mAP@0.5)

    Returns:
        map_score: float, trung binh AP tren cac class CO xuat hien
            trong ground truth cua tap du lieu nay
        ap_per_class: dict {class_id: ap}, de debug/log rieng tung class
    """
    per_class_scores = {c: [] for c in range(num_classes)}
    per_class_tp = {c: [] for c in range(num_classes)}
    per_class_num_gt = {c: 0 for c in range(num_classes)}

    for (pred_boxes, pred_scores, pred_classes), (gt_boxes, gt_classes) in zip(
        all_predictions, all_ground_truths
    ):
        for c in range(num_classes):
            per_class_num_gt[c] += int((gt_classes == c).sum().item())

        if pred_boxes.shape[0] == 0:
            continue

        # Tai su dung matching 1-anh da co san o Buoc 1 -- KHONG viet lai
        # logic IoU-matching, chi gom ket qua lai theo class.
        order, is_tp, _ = match_predictions_to_ground_truth(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold
        )

        scores_sorted = pred_scores[order]
        classes_sorted = pred_classes[order]

        for i in range(len(order)):
            c = int(classes_sorted[i].item())
            per_class_scores[c].append(scores_sorted[i].item())
            per_class_tp[c].append(bool(is_tp[i].item()))

    ap_per_class = {}
    for c in range(num_classes):
        num_gt_c = per_class_num_gt[c]
        if num_gt_c == 0:
            # Class nay khong xuat hien trong ground truth cua tap du
            # lieu nay -> khong co gi de tinh recall, bo qua (giong
            # quy uoc chuan cua COCO/VOC).
            continue

        scores_c = per_class_scores[c]
        tp_c = per_class_tp[c]

        if len(scores_c) == 0:
            # Co GT nhung model khong predict duoc gi cho class nay -> AP=0
            ap_per_class[c] = 0.0
            continue

        # GOM prediction tu nhieu anh lai -> phai sap xep lai theo
        # confidence GIAM DAN tren TOAN BO tap du lieu (khong chi trong
        # tung anh rieng le nhu order cua match_predictions... o tren).
        order_idx = sorted(range(len(scores_c)), key=lambda i: scores_c[i], reverse=True)
        tp_sorted = torch.tensor([tp_c[i] for i in order_idx], dtype=torch.bool)

        precisions, recalls = calculate_precision_recall(tp_sorted, num_gt_c)
        ap_per_class[c] = calculate_ap(precisions.tolist(), recalls.tolist())

    if len(ap_per_class) == 0:
        return 0.0, {}

    map_score = sum(ap_per_class.values()) / len(ap_per_class)
    return map_score, ap_per_class


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

    # ------------------------------------------------------------
    # TEST BUOC 2: mAP tren 1 tap du lieu GIA gom 2 anh, 2 class
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST calculate_map() tren tap du lieu gia (2 anh, 2 class)")
    print("=" * 60)

    # Anh 1: giong vi du Buoc 1 o tren (class 0 co 3 GT, 1 bi miss)
    img1_pred = (pred_boxes, pred_scores, pred_classes)
    img1_gt = (gt_boxes, gt_classes)

    # Anh 2: them class 1, prediction hoan hao (AP=1.0 cho class 1)
    img2_gt_boxes = torch.tensor([[50., 50., 150., 150.]])
    img2_gt_classes = torch.tensor([1])
    img2_pred_boxes = torch.tensor([[52., 52., 148., 148.]])
    img2_pred_scores = torch.tensor([0.99])
    img2_pred_classes = torch.tensor([1])
    img2_pred = (img2_pred_boxes, img2_pred_scores, img2_pred_classes)
    img2_gt = (img2_gt_boxes, img2_gt_classes)

    all_predictions = [img1_pred, img2_pred]
    all_ground_truths = [img1_gt, img2_gt]

    map_score, ap_per_class = calculate_map(
        all_predictions, all_ground_truths, num_classes=2, iou_threshold=0.5
    )

    print("AP tung class:", {k: round(v, 4) for k, v in ap_per_class.items()})
    print("mAP@0.5      :", round(map_score, 4))
    print("\nExpected: class 1 AP = 1.0 (predict hoan hao), class 0 AP < 1.0")
    print("(vi GT3 cua class 0 bi miss va co 1 duplicate prediction bi FP)")