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
"""

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
        order:  index cua pred, sap xep theo SCORE GIAM DAN. Day chinh la
                "confidence ranking" -- se dung lai o Buoc 2 de ve duong
                cong Precision-Recall.
        is_tp:  bool tensor [N], THEO DUNG THU TU cua `order`.
                True  = prediction nay la True Positive.
                False = prediction nay la False Positive.
        num_gt: int, tong so ground-truth box (mau so cua Recall).
    """
    num_gt = gt_boxes.shape[0]

    # Sap xep prediction theo score GIAM DAN: prediction diem cao hon
    # duoc "quyen uu tien" chon GT truoc.
    order = torch.argsort(pred_scores, descending=True)

    matched_gt = torch.zeros(num_gt, dtype=torch.bool)  # GT nao da "co chu" roi
    is_tp = torch.zeros(len(order), dtype=torch.bool)

    for i, pred_idx in enumerate(order.tolist()):
        pred_box = pred_boxes[pred_idx]
        pred_cls = pred_classes[pred_idx].item()

        # chi duoc match voi GT CUNG CLASS va CHUA co prediction nao nhan
        candidate_mask = (gt_classes == pred_cls) & (~matched_gt)

        if candidate_mask.sum() == 0:
            continue   # het GT cung class de match -> FP (is_tp[i] giu False)

        candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
        ious = compute_iou(pred_box, gt_boxes[candidate_idx])
        best_iou, best_local = ious.max(dim=0)

        if best_iou.item() >= iou_threshold:
            gt_idx = candidate_idx[best_local].item()
            matched_gt[gt_idx] = True   # GT nay het hang, pred sau khong duoc nhan nua
            is_tp[i] = True
        # nguoc lai: IoU khong du -> FP, is_tp[i] giu nguyen False

    return order, is_tp, num_gt


def summarize(is_tp, num_gt):
    """
    Ham phu de xem nhanh ket qua Buoc 1 bang cach CONG DON tat ca
    prediction lai thanh 1 con so Precision/Recall duy nhat.

    LUU Y: day CHUA phai duong cong Precision-Recall that su (do can
    tinh precision/recall LUY KE theo TUNG MUC confidence -- se lam o
    Buoc 2). Ham nay chi de kiem tra nhanh logic match co dung khong.
    """
    tp = is_tp.sum().item()
    fp = (~is_tp).sum().item()
    fn = num_gt - tp   # GT nao khong duoc match la False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0

    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall}


if __name__ == "__main__":
    # ---- TEST BUOC 1 BANG 1 ANH GIA: 3 ground truth, 4 prediction ----
    #
    #        GT1                 GT2                 GT3 (KHONG co pred nao khop)
    #   [100,100,200,200]   [400,400,500,500]   [700,700,800,800]
    #
    #   P1 (score .95) khop sat GT1  -> TP, GT1 "het hang"
    #   P3 (score .90) khop sat GT2  -> TP, GT2 "het hang"
    #   P2 (score .85) cung nham GT1 nhung GT1 da co chu -> FP (duplicate)
    #   P4 (score .60) o giua, khong trung GT nao         -> FP (sai vi tri)
    #   GT3 khong prediction nao nhan                     -> FN
    gt_boxes = torch.tensor([
        [100., 100., 200., 200.],  # GT1
        [400., 400., 500., 500.],  # GT2
        [700., 700., 800., 800.],  # GT3 -- se bi MISS
    ])
    gt_classes = torch.tensor([0, 0, 0])  # deu la Person

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
    print("  (thu tu la P1, P3, P2, P4 vi sap theo score giam dan 0.95>0.90>0.85>0.60)")
    print("  P1 -> TP (khop GT1)")
    print("  P3 -> TP (khop GT2)")
    print("  P2 -> FP (GT1 da bi P1 lay mat, du P2 cung khoanh dung vung do)")
    print("  P4 -> FP (khong khop GT nao)")
    print("Expected TP=2, FP=2, FN=1 (GT3 bi mat hoan toan), precision=0.5, recall=0.667")