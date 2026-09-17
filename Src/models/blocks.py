"""
blocks.py
Building blocks cơ bản cho detector.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block đơn giản."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(c_in, c_out, k=3, s=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)
    