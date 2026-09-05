"""

Ghép Backbone + Detection Head, CHƯA có Neck, CHƯA multi-scale.
Chỉ dùng P5 (feature map nhỏ nhất, nhiều semantic nhất) để dự đoán.

"""

import torch
import torch.nn as nn

from Backbone import Backbone
from head import DetectionHead


class MiniPPEDetector(nn.Module):
    """
    Version 0.1: Single-scale detector.

    Input:  [B, 3, 640, 640]
    Output: [B, 5+num_classes, 20, 20]   (dự đoán dựa trên P5)
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = Backbone()

        # P5 có 512 channels (xem lại backbone.py)
        # -> Head phải khai in_channels=512 để khớp
        self.head = DetectionHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        # Version 0.1: CHỈ dùng P5, bỏ qua P3, P4
        # (P3, P4 sẽ được dùng ở Version 0.2 khi có Neck + multi-scale head)
        pred = self.head(p5)

        return pred


if __name__ == "__main__":
    # ---- TEST TOÀN BỘ MODEL VERSION 0.1 ----
    model = MiniPPEDetector(num_classes=3)

    x = torch.randn(2, 3, 640, 640)
    pred = model(x)

    print("Input shape :", x.shape)
    print("Pred shape  :", pred.shape)

    # Expected: Pred shape  : torch.Size([2, 8, 20, 20])

    num_params = sum(p.numel() for p in model.parameters())
    print("Total params:", num_params)
    print("Batch size            :", pred.shape[0])
print("Số kênh (5+classes)    :", pred.shape[1])
print("Grid height             :", pred.shape[2])
print("Grid width              :", pred.shape[3])