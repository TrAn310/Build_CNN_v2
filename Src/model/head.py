"""
head.py

Detection Head cho Version 0.1 (single-scale).

Nhiệm vụ: nhận feature map từ backbone, dùng Conv2d kernel=1
để biến đổi số channel từ C_in -> (5 + num_classes),
Giữ nguyên kích thước không gian H, W.
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
    # ---- TEST RIÊNG HEAD TRƯỚC KHI GHÉP VÀO DETECTOR ----
    num_classes = 3  # Person, Helmet, Vest
    head = DetectionHead(in_channels=512, num_classes=num_classes)

    # Giả lập P5 output từ backbone
    p5 = torch.randn(2, 512, 20, 20)
    out = head(p5)

    print("Input shape (P5):", p5.shape)
    print("Output shape    :", out.shape)

    # Expected: Output shape: torch.Size([2, 8, 20, 20])
    print("Expected out_channels =", 5 + num_classes)