"""
head.py
Anchor-free detection head.

Output: [B, 5 + num_classes, H, W]
Trong đó: (tx, ty, tw, th, objectness, cls_0..cls_{C-1})
"""

import torch
import torch.nn as nn
from .blocks import ConvBlock


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=3, hidden=128):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = 5 + num_classes

        self.stem = nn.Sequential(
            ConvBlock(in_channels, hidden, k=3),
            ConvBlock(hidden, hidden, k=3),
        )
        self.pred_box = nn.Conv2d(hidden, 5, 1)
        self.pred_cls = nn.Conv2d(hidden, num_classes, 1)

        # Init bias objectness âm để tránh positive ban đầu
        nn.init.constant_(self.pred_box.bias[4], -4.0)

    def forward(self, x):
        feat = self.stem(x)
        box = self.pred_box(feat)
        cls = self.pred_cls(feat)
        return torch.cat([box, cls], dim=1)