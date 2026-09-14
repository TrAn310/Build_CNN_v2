"""
losses.py

Detection Loss cho single-scale detector (P5, grid 20x20, stride 32).

Nhận:
    raw_pred:   [B, 5+C, H, W]   output thô từ Detection Head (CHƯA decode)
    target_obj: [B, H, W]        từ target_assigner.build_targets()
    target_box: [B, H, W, 4]     (tx*, ty*, tw*, th*) - đã encode sẵn
    target_cls: [B, H, W]        class index (long)

Trả về:
    total_loss, và 3 loss thành phần để theo dõi riêng khi train.
"""

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
        super().__init__()

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        # reduction="sum" vì ta sẽ tự chia số lượng positive cell sau,
        # để loss không phụ thuộc batch size / số cell một cách khó kiểm soát
        self.box_loss_fn = nn.SmoothL1Loss(reduction="sum")
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        # cls dùng reduction="sum" tương tự, chia số positive cell thủ công bên dưới
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, raw_pred, target_obj, target_box, target_cls):
        B, ch, H, W = raw_pred.shape
        num_classes = ch - 5

        # Đưa channel về cuối, giống hệt cách làm trong decoder.py
        # để tách tx,ty,tw,th,obj,cls một cách nhất quán với phần decode.
        pred = raw_pred.permute(0, 2, 3, 1)  # [B, H, W, 5+C]

        pred_box_raw = pred[..., 0:4]   # [B,H,W,4]  = (tx,ty,tw,th) RAW, chưa sigmoid/exp
        pred_obj_raw = pred[..., 4]     # [B,H,W]    RAW, chưa sigmoid
        pred_cls_raw = pred[..., 5:]    # [B,H,W,C]  RAW logits, chưa softmax

        # ------------------------------------------------------------
        # MASK: xác định vị trí positive (có object) trong toàn bộ B*H*W cell
        # ------------------------------------------------------------
        positive_mask = target_obj == 1.0   # [B,H,W] kiểu bool

        num_positive = positive_mask.sum().item()
        # Nếu ảnh không có object nào (num_positive=0), tránh chia 0 ở dưới
        num_positive = max(num_positive, 1)

        # ==============================================================
        # 1. BOX LOSS - chỉ tính tại positive cell
        # ==============================================================
        # positive_mask có shape [B,H,W], còn pred_box_raw/target_box có
        # thêm chiều cuối = 4 -> dùng positive_mask để "lọc" ra đúng các
        # cell positive, kết quả là tensor phẳng [num_positive, 4]
        pred_box_pos = pred_box_raw[positive_mask]     # [num_positive, 4]
        target_box_pos = target_box[positive_mask]     # [num_positive, 4]

        loss_box = self.box_loss_fn(pred_box_pos, target_box_pos) / num_positive

        # ==============================================================
        # 2. OBJECTNESS LOSS - Cân bằng tỷ lệ (Fix Loss Drowning)
        # ==============================================================
        import torch.nn.functional as F
        
        # Tính xem số lượng ô nền (Negative) đang áp đảo bao nhiêu lần
        num_negative = (B * H * W) - num_positive
        
        # Tạo trọng số để "bơm" giá trị cho Positive. 
        # Cắt ngọn (min) ở mức 100.0 để tránh gradient phát nổ nếu ảnh quá trống.
        weight_ratio = min(num_negative / num_positive, 100.0)
        pos_weight = torch.tensor([weight_ratio], device=raw_pred.device)
        
        # Sử dụng F.binary_cross_entropy_with_logits hỗ trợ truyền pos_weight
        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj_raw, 
            target_obj, 
            reduction="mean", 
            pos_weight=pos_weight
        )

        # ==============================================================
        # 3. CLASSIFICATION LOSS - chỉ tính tại positive cell
        # ==============================================================
        pred_cls_pos = pred_cls_raw[positive_mask]      # [num_positive, C]
        target_cls_pos = target_cls[positive_mask]      # [num_positive]

        # CrossEntropyLoss cần ít nhất 1 sample, nếu không có positive cell
        # nào thì loss_cls = 0 (không có gì để học ở batch này)
        if num_positive > 0 and pred_cls_pos.shape[0] > 0:
            loss_cls = self.cls_loss_fn(pred_cls_pos, target_cls_pos) / num_positive
        else:
            loss_cls = torch.tensor(0.0, device=raw_pred.device)

        # ==============================================================
        # TỔNG HỢP
        # ==============================================================
        total_loss = (
            self.lambda_box * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_cls * loss_cls
        )

        return total_loss, loss_box, loss_obj, loss_cls


def compute_multiscale_loss(preds, gt_boxes, gt_classes, num_classes, loss_fn):
    """
    Mở rộng của DetectionLoss: tính loss cho CẢ 3 SCALE (P3, P4, P5) cùng lúc.

    DetectionLoss ở trên chỉ biết tính loss cho 1 scale. Hàm này gọi lại
    DetectionLoss 3 lần (1 lần/scale) rồi cộng kết quả - không định nghĩa
    công thức toán mới, chỉ điều phối.

    Input:
        preds: tuple (pred_p3, pred_p4, pred_p5) - output từ detector.py
        gt_boxes, gt_classes: xem target_assigner.build_targets_multiscale
        num_classes: int
        loss_fn: instance của DetectionLoss (tạo 1 lần, tái sử dụng)

    Output:
        total_loss: tổng loss của cả 3 scale (dùng để .backward())
        loss_dict: dict chứa từng thành phần loss của từng scale, để debug
    """
    from target_assigner import build_targets_multiscale

    pred_p3, pred_p4, pred_p5 = preds
    targets = build_targets_multiscale(gt_boxes, gt_classes, num_classes)

    preds_per_scale = [pred_p3, pred_p4, pred_p5]
    scale_names = ["P3", "P4", "P5"]

    total_loss = 0.0
    loss_dict = {}

    for i in range(3):
        raw_pred = preds_per_scale[i]
        target_obj, target_box, target_cls = targets[i]

        target_obj = target_obj.to(raw_pred.device)
        target_box = target_box.to(raw_pred.device)
        target_cls = target_cls.to(raw_pred.device)

        loss, l_box, l_obj, l_cls = loss_fn(raw_pred, target_obj, target_box, target_cls)
        total_loss = total_loss + loss

        name = scale_names[i]
        loss_dict[f"loss_{name}"] = loss.item()
        loss_dict[f"loss_{name}_box"] = l_box.item()
        loss_dict[f"loss_{name}_obj"] = l_obj.item()
        loss_dict[f"loss_{name}_cls"] = l_cls.item()

    loss_dict["total_loss"] = total_loss.item()
    return total_loss, loss_dict


if __name__ == "__main__":
    # ---- TEST 1: DetectionLoss cho 1 scale riêng lẻ (giữ nguyên như cũ) ----
    import sys
    import os

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(CURRENT_DIR)  # để tìm target_assigner.py (cùng thư mục training/)

    # detector.py nằm ở Model/src/, còn file này ở Model/training/
    # -> phải thêm đường dẫn tới src/ thì Python mới "thấy" được detector.py
    SRC_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "src"))
    sys.path.append(SRC_DIR)

    from target_assigner import build_targets_multiscale

    num_classes = 3
    gt_boxes = [torch.tensor([[210.0, 150.0, 270.0, 210.0]])]  # w=h=60 -> P5
    gt_classes = [torch.tensor([0])]

    targets = build_targets_multiscale(gt_boxes, gt_classes, num_classes)
    target_obj, target_box, target_cls = targets[2]  # lấy P5 để test 1-scale

    print("target_obj shape (P5):", target_obj.shape)

    B, C, H, W = 1, num_classes, 20, 20
    raw_pred_bad = torch.randn(B, 5 + C, H, W)

    loss_fn = DetectionLoss()
    total, l_box, l_obj, l_cls = loss_fn(raw_pred_bad, target_obj, target_box, target_cls)

    print("\n--- Loss 1 scale (P5) với prediction NGẪU NHIÊN ---")
    print(f"total_loss = {total.item():.4f}")

    # ---- TEST 2: compute_multiscale_loss cho CẢ 3 SCALE cùng lúc ----
    from detector import MiniPPEDetector

    model = MiniPPEDetector(num_classes=num_classes)
    x = torch.randn(1, 3, 640, 640)
    preds = model(x)

    gt_boxes_multi = [
        torch.tensor([
            [100.0, 100.0, 130.0, 130.0],   # nhỏ -> P3
            [300.0, 200.0, 380.0, 280.0],   # vừa -> P4
            [50.0, 50.0, 250.0, 250.0],     # lớn -> P5
        ])
    ]
    gt_classes_multi = [torch.tensor([1, 0, 0])]

    total_loss, loss_dict = compute_multiscale_loss(
        preds, gt_boxes_multi, gt_classes_multi, num_classes, loss_fn
    )

    print("\n--- Loss breakdown (3 scale) ---")
    for k, v in loss_dict.items():
        print(f"{k}: {v:.4f}")

    total_loss.backward()
    print("\nbackward() chạy thành công.")