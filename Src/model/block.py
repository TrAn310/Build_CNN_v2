import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # bias=False vì BatchNorm ngay sau đã có phần "shift" riêng
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    # ---- TEST RIÊNG BLOCK NÀY TRƯỚC KHI DÙNG TRONG BACKBONE ----
    block = ConvBlock(in_channels=3, out_channels=32, stride=2)

    x = torch.randn(2, 3, 640, 640)
    out = block(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)

    # Expected: Output shape: torch.Size([2, 32, 320, 320])
    block2 = ConvBlock(in_channels=32, out_channels=64, stride=2)
    x2 = torch.randn(2, 32, 320, 320)
    out2 = block2(x2)
    print("Input shape :", x2.shape)
    print("Output shape:", out2.shape)