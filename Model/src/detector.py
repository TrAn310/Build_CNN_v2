"""
detector.py

Version 0.2: Multi-scale detector.

Backbone -> Neck -> 3 Head riêng (P3, P4, P5) -> 3 raw prediction.
"""

import torch
import torch.nn as nn

from Backbone import Backbone
from neck import Neck
from head import DetectionHead
from utils.config_utils import load_dataset_config


class MiniPPEDetector(nn.Module):
    """
    Version 0.2: Multi-scale detector.

    Input:  [B, 3, 640, 640]
    Output: 3 tensor (pred_p3, pred_p4, pred_p5)
        pred_p3: [B, 5+num_classes, 80, 80]
        pred_p4: [B, 5+num_classes, 40, 40]
        pred_p5: [B, 5+num_classes, 20, 20]
    """

    def __init__(self, num_classes=3, neck_out_channels=256):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = Backbone()
        self.neck = Neck(c3=128, c4=256, c5=512, out_channels=neck_out_channels)

        # 3 HEAD RIÊNG - không dùng chung 1 head cho cả 3 scale
        self.head_p3 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)
        self.head_p4 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)
        self.head_p5 = DetectionHead(in_channels=neck_out_channels, num_classes=num_classes)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)

        pred_p3 = self.head_p3(f3)  # [B,8,80,80]
        pred_p4 = self.head_p4(f4)  # [B,8,40,40]
        pred_p5 = self.head_p5(f5)  # [B,8,20,20]

        return pred_p3, pred_p4, pred_p5


if __name__ == "__main__":
    import os
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.normpath(os.path.join(CURRENT_DIR, "..", "Data", "data.yaml"))
 
    num_classes, class_names = load_dataset_config(yaml_path)
    print("num_classes lấy từ data.yaml:", num_classes)
    print("class_names:", class_names)
    # ---- TEST TOÀN BỘ MODEL VERSION 0.2 ----
    model = MiniPPEDetector(num_classes=3)

    x = torch.randn(2, 3, 640, 640)
    pred_p3, pred_p4, pred_p5 = model(x)

    print("Input shape :", x.shape)
    print("Pred P3 shape:", pred_p3.shape)  # Expected: [2, 8, 80, 80]
    print("Pred P4 shape:", pred_p4.shape)  # Expected: [2, 8, 40, 40]
    print("Pred P5 shape:", pred_p5.shape)  # Expected: [2, 8, 20, 20]

    total_predictions = (
        pred_p3.shape[2] * pred_p3.shape[3]
        + pred_p4.shape[2] * pred_p4.shape[3]
        + pred_p5.shape[2] * pred_p5.shape[3]
    )
    print("Tổng số vị trí dự đoán (dense prediction):", total_predictions)
    # Expected: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400

    num_params = sum(p.numel() for p in model.parameters())
    print("Total params:", num_params)