"""
detector.py
Custom PPE Detector hoÃ n chá»‰nh.

Pipeline:
    Image [B,3,640,640]
        -> Backbone
        -> P3, P4, P5
        -> Neck
        -> F3, F4, F5
        -> 3 Detection Heads
        -> raw outputs
"""

import torch.nn as nn
from Src.models.backbone import Backbone
from Src.models.neck import Neck
from Src.models.head import DetectionHead


class CustomPPEDetector(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0, neck_channels=128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Backbone(width_mult=width_mult)
        self.neck = Neck(self.backbone.out_channels, out_channels=neck_channels)

        self.head_p3 = DetectionHead(neck_channels, num_classes)
        self.head_p4 = DetectionHead(neck_channels, num_classes)
        self.head_p5 = DetectionHead(neck_channels, num_classes)

        self.strides = [8, 16, 32]

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return [self.head_p3(f3), self.head_p4(f4), self.head_p5(f5)]
    
