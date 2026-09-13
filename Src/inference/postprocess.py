"""
postprocess.py

Pipeline day du: decode -> objectness*class = score -> threshold -> NMS
-> final detections.

Ban da hoc IoU/NMS ly thuyet roi nen o day chi giai thich phan MOI:
cach ghep 3 scale (P3,P4,P5) thanh 1 danh sach chung, va vi sao phai
chay NMS RIENG cho tung class (khong tron chung tat ca class lai).
"""

import torch

from decoder import decode_predictions

# QUAN TRONG: config nay PHAI khop voi SCALE_CONFIGS trong target_assigner.py
# (dung thu tu P3, P4, P5 vi detector.py tra ve preds theo dung thu tu do)
SCALE_CONFIGS = [
    {"name": "P3", "stride": 8},
    {"name": "P4", "stride": 16},
    {"name": "P5", "stride": 32},
]


def _flatten_scale(boxes, objectness, class_probs):
    """
    Gop chieu (H, W) thanh 1 chieu N = H*W, de sau nay concat 3 scale
    lai voi nhau tren cung 1 chieu (flatten SPATIAL, khong phai flatten
    CHANNEL nhu trong classification).

    boxes:       [B, H, W, 4]  -> [B, H*W, 4]
    objectness:  [B, H, W]     -> [B, H*W]
    class_probs: [B, H, W, C]  -> [B, H*W, C]
    """
    B, H, W, _ = boxes.shape
    C = class_probs.shape[-1]

    boxes = boxes.reshape(B, H * W, 4)
    objectness = objectness.reshape(B, H * W)
    class_probs = class_probs.reshape(B, H * W, C)

    return boxes, objectness, class_probs


def compute_iou(box, boxes):
    """
    Tinh IoU giua 1 box va MOT DANH SACH box khac. Ham nay dung chung
    cho ca NMS (trong file nay) va evaluator.py (Bai 9) - viet 1 lan,
    dung lai 2 cho, tranh code trung lap 2 cong thuc IoU khac nhau.

    Args:
        box:  tensor [4]    (x1,y1,x2,y2) - 1 box duy nhat
        boxes: tensor [N,4] (x1,y1,x2,y2) - N box can so sanh

    Returns:
        ious: tensor [N] - IoU giua box va tung box trong boxes
    """
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    union_area = area_box + area_boxes - inter_area
    union_area = union_area.clamp(min=1e-6)  # tranh chia cho 0

    return inter_area / union_area


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression tu code (KHONG dung torchvision.ops.nms).

    Y tuong: lap lai nhieu lan viec "chon box diem cao nhat con lai,
    loai bo moi box khac chong lan qua nhieu (IoU cao) voi no", cho den
    khi khong con box nao.

    Args:
        boxes:  tensor [N,4]  (x1,y1,x2,y2)
        scores: tensor [N]
        iou_threshold: box bi loai neu IoU voi box da chon >= nguong nay

    Returns:
        keep: list index (trong boxes/scores goc) cua cac box duoc GIU LAI
    """
    if boxes.shape[0] == 0:
        return []

    # Sap xep index theo score giam dan
    order = torch.argsort(scores, descending=True).tolist()

    keep = []
    while len(order) > 0:
        current = order[0]      # box diem cao nhat CON LAI -> chac chan duoc giu
        keep.append(current)

        if len(order) == 1:
            break

        rest_idx = order[1:]
        rest_boxes = boxes[rest_idx]

        ious = compute_iou(boxes[current], rest_boxes)   # [len(rest_idx)]

        # Chi giu lai nhung box co IoU THAP hon nguong (khong bi coi la
        # "trung lap" voi box vua chon)
        remaining_mask = ious < iou_threshold
        order = [rest_idx[i] for i in range(len(rest_idx)) if remaining_mask[i].item()]

    return keep


def decode_and_filter(preds, conf_threshold=0.3, nms_iou_threshold=0.5):
    """
    Args:
        preds: tuple (pred_p3, pred_p4, pred_p5), moi tensor [B, 5+C, H, W]
               -- output tho tu MiniPPEDetector (CHUA qua decode)
        conf_threshold: box bi loai neu score <= nguong nay
        nms_iou_threshold: nguong IoU dung trong buoc NMS

    Returns:
        list co B phan tu (B = batch size). Moi phan tu la tuple:
            boxes:   tensor [N, 4]   (x1,y1,x2,y2) tren anh 640x640
            scores:  tensor [N]      confidence score = objectness * class_prob
            classes: tensor [N]      class index co score cao nhat

        N khac nhau tuy anh -- day la ket qua SAU CUNG (da qua NMS),
        san sang de ve len anh hoac dua vao evaluator.py.
    """
    all_boxes, all_obj, all_cls = [], [], []

    for raw_pred, cfg in zip(preds, SCALE_CONFIGS):
        stride = cfg["stride"]

        # decode_predictions da tu sigmoid(objectness) va softmax(class) roi
        # -- xem lai decoder.py, khong can lam lai o day
        boxes, objectness, class_probs = decode_predictions(
            raw_pred, stride=stride, base_w=stride, base_h=stride
        )

        boxes, objectness, class_probs = _flatten_scale(boxes, objectness, class_probs)

        all_boxes.append(boxes)
        all_obj.append(objectness)
        all_cls.append(class_probs)

    # Ghep 3 scale lai theo chieu N (dim=1):
    # P3: 80*80=6400, P4: 40*40=1600, P5: 20*20=400 -> tong 8400
    boxes_all = torch.cat(all_boxes, dim=1)   # [B, 8400, 4]
    obj_all = torch.cat(all_obj, dim=1)       # [B, 8400]
    cls_all = torch.cat(all_cls, dim=1)       # [B, 8400, C]

    # ---- CONG THUC QUAN TRONG NHAT CUA BUOC NAY ----
    # score cho TUNG class = objectness * class_probability
    scores_all = obj_all.unsqueeze(-1) * cls_all   # [B, 8400, C]

    # moi vi tri chi giu lai class co score cao nhat
    final_scores, final_classes = scores_all.max(dim=-1)  # [B,8400], [B,8400]

    results = []
    B = boxes_all.shape[0]
    for b in range(B):
        mask = final_scores[b] > conf_threshold
        boxes_b = boxes_all[b][mask]
        scores_b = final_scores[b][mask]
        classes_b = final_classes[b][mask]

        # ---- NMS RIENG CHO TUNG CLASS ----
        # Vi sao khong NMS chung tat ca class 1 luot: neu 1 vi tri co
        # Person va Helmet chong len nhau (VD nguoi dang doi mu), NMS
        # gop chung se lam mat 1 trong 2 box, du chung la 2 OBJECT
        # THAT KHAC NHAU, khong phai box trung lap cua CUNG 1 object.
        keep_indices = []
        for cls_id in classes_b.unique().tolist():
            cls_mask = (classes_b == cls_id).nonzero(as_tuple=True)[0]
            cls_boxes = boxes_b[cls_mask]
            cls_scores = scores_b[cls_mask]

            keep_local = nms(cls_boxes, cls_scores, iou_threshold=nms_iou_threshold)
            keep_indices.extend([cls_mask[i].item() for i in keep_local])

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)

        results.append((
            boxes_b[keep_indices],
            scores_b[keep_indices],
            classes_b[keep_indices],
        ))

    return results


if __name__ == "__main__":
    # ---- TEST 1: NMS co loai duoc box TRUNG LAP khong ----
    test_boxes = torch.tensor([
        [100., 100., 200., 200.],   # box A - score cao nhat
        [105., 105., 205., 205.],   # box B - gan trung box A (IoU cao) -> phai bi loai
        [500., 500., 600., 600.],   # box C - o xa, khong lien quan -> phai duoc giu
    ])
    test_scores = torch.tensor([0.9, 0.8, 0.7])

    keep = nms(test_boxes, test_scores, iou_threshold=0.5)
    print("NMS keep index:", keep)
    print("Expected: [0, 2]  (giu box A va box C, loai box B vi trung box A)\n")

    # ---- TEST 2: decode_and_filter DAY DU (decode + threshold + NMS) ----
    B, C = 1, 3

    pred_p3 = torch.zeros(B, 5 + C, 80, 80)
    pred_p4 = torch.zeros(B, 5 + C, 40, 40)
    pred_p5 = torch.zeros(B, 5 + C, 20, 20)

    gy, gx = 5, 7
    pred_p5[0, 4, gy, gx] = 5.0   # objectness raw cao -> sigmoid ~ 0.99
    pred_p5[0, 5, gy, gx] = 5.0   # class 0 (Person) logit cao -> softmax ~ 0.98

    # Them 1 o LAN CAN de test NMS thuc su co hoat dong trong pipeline day du
    pred_p5[0, 4, gy, gx + 1] = 5.0
    pred_p5[0, 5, gy, gx + 1] = 5.0

    preds = (pred_p3, pred_p4, pred_p5)

    results = decode_and_filter(preds, conf_threshold=0.3, nms_iou_threshold=0.5)
    boxes, scores, classes = results[0]

    print("So box con lai sau threshold + NMS:", boxes.shape[0])
    print("Box   :", boxes.tolist())
    print("Score :", scores.tolist())
    print("Class :", classes.tolist())
    print("\n(Neu 2 o lan can nay tao ra box qua gan nhau, NMS se chi giu 1;")
    print(" neu box cua chung du xa nhau (do decode ra vi tri khac han),")
    print(" ca 2 co the duoc giu - phu thuoc IoU thuc te giua chung.)")