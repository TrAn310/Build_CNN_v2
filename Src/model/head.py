"""
head.py

Detection Head -- 1 head DÙNG CHUNG CHO MỌI SCALE trong kiến trúc
multi-scale (Version 0.2). MiniPPEDetector (detector.py) tự tạo 3
INSTANCE RIÊNG của class này (head_p3, head_p4, head_p5), mỗi instance
nhận feature map đã qua Neck (cùng số channel = neck_out_channels).

Nhiệm vụ: dùng Conv2d kernel=1 để biến đổi số channel từ C_in ->
(5 + num_classes), giữ nguyên kích thước không gian H, W.
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Input:  [B, in_channels, H, W]
    Output: [B, 5 + num_classes, H, W]

    5 kênh đầu:  tx, ty, tw, th, objectness
    num_classes kênh sau: class scores (Person, Helmet, Vest, ...)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        out_channels = 5 + num_classes

        # kernel_size=1: chỉ trộn channel, KHÔNG trộn thông tin không gian
        # -> giữ nguyên nguyên tắc "mỗi vị trí (h,w) tự dự đoán độc lập"
        self.pred = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        return self.pred(x)


if __name__ == "__main__":
    # ---- TEST HEAD DUNG DUNG CACH detector.py THUC SU SU DUNG ----
    # Neck luon xuat ra out_channels=256 cho ca 3 scale (xem neck.py),
    # nen head nhan in_channels=256 -- KHONG PHAI 512 nhu output tho
    # cua Backbone P5. detector.py tao 3 INSTANCE rieng, moi instance
    # dung cho 1 scale (H,W khac nhau, nhung cung in_channels=256).
    num_classes = 3  # Person, Helmet, Vest
    neck_out_channels = 256

    head_p3 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)
    head_p4 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)
    head_p5 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)

    # Gia lap output cua Neck (f3, f4, f5) -- xem lai neck.py
    f3 = torch.randn(2, neck_out_channels, 80, 80)
    f4 = torch.randn(2, neck_out_channels, 40, 40)
    f5 = torch.randn(2, neck_out_channels, 20, 20)

    pred_p3 = head_p3(f3)
    pred_p4 = head_p4(f4)
    pred_p5 = head_p5(f5)

    print("Pred P3 shape:", pred_p3.shape)  # Expected: [2, 8, 80, 80]
    print("Pred P4 shape:", pred_p4.shape)  # Expected: [2, 8, 40, 40]
    print("Pred P5 shape:", pred_p5.shape)  # Expected: [2, 8, 20, 20]
    print("\nExpected out_channels (moi scale) =", 5 + num_classes)