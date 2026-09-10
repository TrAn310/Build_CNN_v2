"""
postprocess.py

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


def decode_and_filter(preds, conf_threshold=0.3):
    """
    Args:
        preds: tuple (pred_p3, pred_p4, pred_p5), moi tensor [B, 5+C, H, W]
               -- output tho tu MiniPPEDetector (CHUA qua decode)
        conf_threshold: box bi loai neu score <= nguong nay

    Returns:
        list co B phan tu (B = batch size). Moi phan tu la tuple:
            boxes:   tensor [N, 4]   (x1,y1,x2,y2) tren anh 640x640
            scores:  tensor [N]      confidence score = objectness * class_prob
            classes: tensor [N]      class index co score cao nhat

        N khac nhau tuy anh -- giong dung style boxes/classes trong
        dataset.py va collate.py, de sau nay ghep chung 1 pipeline de dang.
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
    # (objectness tra loi "co object khong", class_prob tra loi "neu co
    #  thi la class gi" -- phai nhan lai voi nhau moi ra confidence that
    #  su cho tung class, mot minh objectness cao khong co nghia gi neu
    #  model khong chac la class nao)
    scores_all = obj_all.unsqueeze(-1) * cls_all   # [B, 8400, C]

    # moi vi tri chi giu lai class co score cao nhat
    final_scores, final_classes = scores_all.max(dim=-1)  # [B,8400], [B,8400]

    results = []
    B = boxes_all.shape[0]
    for b in range(B):
        mask = final_scores[b] > conf_threshold
        results.append((
            boxes_all[b][mask],
            final_scores[b][mask],
            final_classes[b][mask],
        ))

    return results


if __name__ == "__main__":
    # ---- TEST BUOC 1 BANG PREDICTION GIA ----
    # Dung lai dung cach test cua decoder.py: dat 1 o "co object ro rang"
    # tai P5, gy=5 gx=7, con lai de 0 het (nghia la "nen", khong co object)
    B, C = 1, 3

    pred_p3 = torch.zeros(B, 5 + C, 80, 80)
    pred_p4 = torch.zeros(B, 5 + C, 40, 40)
    pred_p5 = torch.zeros(B, 5 + C, 20, 20)

    gy, gx = 5, 7
    pred_p5[0, 4, gy, gx] = 5.0   # objectness raw cao -> sigmoid ~ 0.99
    pred_p5[0, 5, gy, gx] = 5.0   # class 0 (Person) logit cao -> softmax ~ 0.98

    preds = (pred_p3, pred_p4, pred_p5)

    results = decode_and_filter(preds, conf_threshold=0.3)
    boxes, scores, classes = results[0]

    print("Tong so vi tri dense prediction:", 80 * 80 + 40 * 40 + 20 * 20)
    print("So box con lai sau threshold  :", boxes.shape[0])
    print("Box   :", boxes.tolist())
    print("Score :", scores.tolist())
    print("Class :", classes.tolist())

    print("\nExpected: 1 box duy nhat, class=0 (Person), score xap xi 0.98")
    print("(8399 vi tri con lai co objectness=sigmoid(0)=0.5 va")
    print(" class_prob=1/3 deu nhau -> score=0.5/3=0.1667 < 0.3 nguong -> bi loc)")