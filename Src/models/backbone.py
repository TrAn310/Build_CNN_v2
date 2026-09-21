"""
backbone.py
Custom CNN backbone, output P3/P4/P5.

Input:  [B, 3, 640, 640]
Output: P3 [B, c3, 80, 80]
        P4 [B, c4, 40, 40]
        P5 [B, c5, 20, 20]
"""

import torch.nn as nn
from Src.models.blocks import ConvBlock, ResidualBlock


class Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        c5 = int(512 * width_mult)

        # Stem: 640 -> 320
        self.stem = ConvBlock(3, c1, k=3, s=2)

        # Stage 1: 320 -> 160
        self.stage1 = nn.Sequential(
            ConvBlock(c1, c2, k=3, s=2),
            ResidualBlock(c2, c2),
        )

        # Stage 2: 160 -> 80 (P3)
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2),
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )

        # Stage 3: 80 -> 40 (P4)
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2),
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )

        # Stage 4: 40 -> 20 (P5)
        self.stage4 = nn.Sequential(
            ConvBlock(c4, c5, k=3, s=2),
            ResidualBlock(c5, c5),
            ResidualBlock(c5, c5),
        )

        self.out_channels = [c3, c4, c5]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
