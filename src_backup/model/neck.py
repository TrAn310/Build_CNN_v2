"""
neck.py
FPN-like top-down fusion.

P5 -> Conv -> Upsample x2 -> Concat P4 -> Conv -> F4
F4 -> Conv -> Upsample x2 -> Concat P3 -> Conv -> F3
F4 -> Conv stride 2 -> Concat P5 -> Conv -> F5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock


class Neck(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat5 = nn.Conv2d(c5, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.lat3 = nn.Conv2d(c3, out_channels, 1)

        self.smooth4 = ConvBlock(out_channels * 2, out_channels, k=3)
        self.smooth3 = ConvBlock(out_channels * 2, out_channels, k=3)

        self.down5 = ConvBlock(out_channels, out_channels, k=3, s=2)
        self.out_channels = out_channels

    def forward(self, p3, p4, p5):
        l5 = self.lat5(p5)
        l4 = self.lat4(p4)
        l3 = self.lat3(p3)

        up5 = F.interpolate(l5, scale_factor=2, mode='nearest')
        cat4 = torch.cat([up5, l4], dim=1)
        f4 = self.smooth4(cat4)

        up4 = F.interpolate(f4, scale_factor=2, mode='nearest')
        cat3 = torch.cat([up4, l3], dim=1)
        f3 = self.smooth3(cat3)

        f5 = self.down5(f4)
        return f3, f4, f5