"""
neck.py

Feature Pyramid-style Neck đơn giản (top-down + lateral connection).

Nhiệm vụ: nhận P3, P4, P5 từ Backbone (khác channel, khác semantic level),
    fuse thông tin theo hướng top-down (P5 -> P4 -> P3) để mỗi output
    (F3, F4, F5) vừa có semantic sâu (từ P5) vừa giữ chi tiết không gian
    (từ P3/P4).

Input:
    p3: [B, 128, 80, 80]
    p4: [B, 256, 40, 40]
    p5: [B, 512, 20, 20]

Output:
    f3: [B, 256, 80, 80]
    f4: [B, 256, 40, 40]
    f5: [B, 256, 20, 20]
"""

import torch
import torch.nn as nn

from block import ConvBlock


class Neck(nn.Module):
    def __init__(self, c3=128, c4=256, c5=512, out_channels=256):
        super().__init__()

        # ---- Lateral conv: chỉ đổi số channel về out_channels, giữ nguyên H,W ----
        # kernel_size=1 -> không trộn thông tin không gian, chỉ trộn channel
        self.reduce5 = ConvBlock(c5, out_channels, kernel_size=1, padding=0)
        self.reduce4 = ConvBlock(c4, out_channels, kernel_size=1, padding=0)
        self.reduce3 = ConvBlock(c3, out_channels, kernel_size=1, padding=0)

        # ---- Upsample: nhân đôi H, W bằng nearest interpolation ----
        # (không có tham số học được, chỉ là phép nội suy)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # ---- Fuse conv: sau khi concat (out_channels + out_channels),
        #      dùng conv 3x3 để trộn thông tin không gian + giảm về out_channels ----
        self.fuse4 = ConvBlock(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.fuse3 = ConvBlock(out_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, p3, p4, p5):
        # F5: chỉ cần giảm channel, không cần fuse gì thêm (đã là cấp sâu nhất)
        f5 = self.reduce5(p5)                     # [B,256,20,20]

        # ---- Fuse F5 -> P4 ----
        f5_up = self.upsample(f5)                 # [B,256,40,40]
        p4_lat = self.reduce4(p4)                 # [B,256,40,40]
        cat4 = torch.cat([f5_up, p4_lat], dim=1)  # [B,512,40,40]
        f4 = self.fuse4(cat4)                     # [B,256,40,40]

        # ---- Fuse F4 -> P3 ----
        f4_up = self.upsample(f4)                 # [B,256,80,80]
        p3_lat = self.reduce3(p3)                 # [B,256,80,80]
        cat3 = torch.cat([f4_up, p3_lat], dim=1)  # [B,512,80,80]
        f3 = self.fuse3(cat3)                     # [B,256,80,80]

        return f3, f4, f5


if __name__ == "__main__":
    # ---- TEST RIÊNG NECK bằng feature map GIẢ (chưa cần chạy Backbone thật) ----
    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)

    neck = Neck(c3=128, c4=256, c5=512, out_channels=256)
    f3, f4, f5 = neck(p3, p4, p5)

    print("F3 shape:", f3.shape)  # Expected: [2, 256, 80, 80]
    print("F4 shape:", f4.shape)  # Expected: [2, 256, 40, 40]
    print("F5 shape:", f5.shape)  # Expected: [2, 256, 20, 20]

    num_params = sum(p.numel() for p in neck.parameters())
    print("Neck params:", num_params)