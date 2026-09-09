"""
head.py

Detection Head — PHIÊN BẢN MULTI-SCALE (P3, P4, P5)

Khác với V0.1 (chỉ 1 head áp cho P5):
  - Giờ có 3 head RIÊNG BIỆT, mỗi head phụ trách 1 scale.
  - Lý do bắt buộc phải tách riêng: P3/P4/P5 có SỐ CHANNEL khác nhau
    (128 / 256 / 512) -> 1 Conv2d chỉ nhận đúng 1 số channel input cố định,
    không thể dùng chung 1 head cho cả 3 scale có channel khác nhau.
  - Mỗi scale cũng "chuyên trách" object cỡ khác nhau (P3 nhỏ, P4 vừa,
    P5 lớn) nên để mỗi head tự học trọng số riêng là hợp lý, không chỉ
    là vấn đề kỹ thuật channel.
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Head cho MỘT scale — logic giữ nguyên y hệt bản V0.1, không đổi gì bên
    trong. Điểm khác là từ giờ sẽ có 3 INSTANCE của class này (P3, P4, P5)
    thay vì chỉ 1.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        out_channels = 5 + num_classes   # tx,ty,tw,th,obj + C class

        # kernel_size=1: chỉ trộn channel, KHÔNG trộn thông tin không gian
        # -> mỗi vị trí (h,w) trên scale này vẫn tự dự đoán độc lập
        self.pred = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.pred(x)


class MultiScaleDetectionHead(nn.Module):
    """
    Bọc 3 DetectionHead độc lập cho P3, P4, P5.

    Input:
        p3 [B,128,80,80]  (object nhỏ, ví dụ helmet)
        p4 [B,256,40,40]  (object trung bình)
        p5 [B,512,20,20]  (object lớn, ví dụ person)

    Output: tuple 3 tensor RIÊNG BIỆT (không gộp làm 1 vì H,W khác nhau
    giữa các scale -> không thể torch.cat trực tiếp):
        pred_p3 [B, 5+num_classes, 80, 80]
        pred_p4 [B, 5+num_classes, 40, 40]
        pred_p5 [B, 5+num_classes, 20, 20]

    LƯU Ý: 3 số in_channels (128/256/512) PHẢI khớp chính xác với channel
    output thật của backbone.py. Nếu sau này bạn đổi kiến trúc backbone,
    phải sửa lại 3 số này theo.
    """
    def __init__(self, num_classes,
                 in_channels_p3=128, in_channels_p4=256, in_channels_p5=512):
        super().__init__()
        self.head_p3 = DetectionHead(in_channels_p3, num_classes)
        self.head_p4 = DetectionHead(in_channels_p4, num_classes)
        self.head_p5 = DetectionHead(in_channels_p5, num_classes)

    def forward(self, p3, p4, p5):
        pred_p3 = self.head_p3(p3)
        pred_p4 = self.head_p4(p4)
        pred_p5 = self.head_p5(p5)
        return pred_p3, pred_p4, pred_p5


if __name__ == "__main__":
    # ---- TEST RIÊNG HEAD, CHƯA GHÉP VÀO DETECTOR ----
    num_classes = 3
    head = MultiScaleDetectionHead(num_classes)

    # Giả lập output backbone (đúng shape đã xác nhận ở Bài 1)
    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)

    pred_p3, pred_p4, pred_p5 = head(p3, p4, p5)

    print("pred_p3:", pred_p3.shape)   # expect [2, 8, 80, 80]
    print("pred_p4:", pred_p4.shape)   # expect [2, 8, 40, 40]
    print("pred_p5:", pred_p5.shape)   # expect [2, 8, 20, 20]