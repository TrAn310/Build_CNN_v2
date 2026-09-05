import torch
import torch.nn as nn

from block import ConvBlock


class Backbone(nn.Module):
    """
    Input:  [B, 3, 640, 640]

    Stage 1: 3   -> 32   stride 2   -> [B, 32,  320, 320]
    Stage 2: 32  -> 64   stride 2   -> [B, 64,  160, 160]
    Stage 3: 64  -> 128  stride 2   -> [B, 128, 80,  80]   => P3
    Stage 4: 128 -> 256  stride 2   -> [B, 256, 40,  40]   => P4
    Stage 5: 256 -> 512  stride 2   -> [B, 512, 20,  20]   => P5
    """

    def __init__(self):
        super().__init__()

        self.stage1 = ConvBlock(in_channels=3,   out_channels=32,  stride=2)
        self.stage2 = ConvBlock(in_channels=32,  out_channels=64,  stride=2)
        self.stage3 = ConvBlock(in_channels=64,  out_channels=128, stride=2)
        self.stage4 = ConvBlock(in_channels=128, out_channels=256, stride=2)
        self.stage5 = ConvBlock(in_channels=256, out_channels=512, stride=2)

    def forward(self, x):
        x = self.stage1(x)   # [B, 32,  320, 320]
        x = self.stage2(x)   # [B, 64,  160, 160]

        p3 = self.stage3(x)  # [B, 128, 80, 80]
        p4 = self.stage4(p3)  # [B, 256, 40, 40]
        p5 = self.stage5(p4)  # [B, 512, 20, 20]

        return p3, p4, p5


if __name__ == "__main__":
    # ---- TEST RIÊNG BACKBONE TRƯỚC KHI GHÉP VÀO DETECTOR ----
    model = Backbone()

    x = torch.randn(2, 3, 640, 640)
    p3, p4, p5 = model(x)

    print("Input shape:", x.shape)
    print("P3 shape   :", p3.shape)
    print("P4 shape   :", p4.shape)
    print("P5 shape   :", p5.shape)

    # Expected:
    # P3 shape   : torch.Size([2, 128, 80, 80])
    # P4 shape   : torch.Size([2, 256, 40, 40])
    # P5 shape   : torch.Size([2, 512, 20, 20])

    num_params = sum(p.numel() for p in model.parameters())
    print("Total params:", num_params)