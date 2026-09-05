"""
=========================================
LAB 4.2 - QUAN SÁT FEATURE MAP CỦA CONV2D
=========================================

Mục tiêu:
1. Đọc ảnh bằng OpenCV
2. Chuyển ảnh sang Tensor PyTorch
3. Tạo lớp Conv2D
4. Quan sát Feature Maps
5. Quan sát ReLU
6. Đếm số tham số
"""

import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==========================================================
# BƯỚC 1. ĐỌC ẢNH
# ==========================================================

img = cv2.imread("testlab4.2.jpg")      # Đổi thành ảnh của bạn

if img is None:
    raise FileNotFoundError("Không tìm thấy ảnh!")

# OpenCV đọc theo BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img = cv2.resize(img, (224, 224))

print("=" * 50)
print("Image shape:", img.shape)

# Hiển thị ảnh
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# ==========================================================
# BƯỚC 2. CHUYỂN THÀNH TENSOR
# ==========================================================

x = torch.tensor(img, dtype=torch.float32)

print("\nTensor ban đầu:", x.shape)

# [224,224,3]
# ->
# [3,224,224]

x = x.permute(2,0,1)

print("Sau permute:", x.shape)

# Thêm Batch Dimension

x = x.unsqueeze(0)

print("Sau unsqueeze:", x.shape)

# Kết quả:
# [1,3,224,224]

# ==========================================================
# BƯỚC 3. TẠO CONV2D
# ==========================================================

conv = nn.Conv2d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)

print("\nConv2D")
print(conv)

# ==========================================================
# BƯỚC 4. FORWARD
# ==========================================================

feature = conv(x)

print("\nInput Shape :", x.shape)
print("Output Shape:", feature.shape)

# ==========================================================
# BƯỚC 5. HIỂN THỊ FEATURE MAP
# ==========================================================

fig, axes = plt.subplots(2,4, figsize=(10,5))

for i in range(8):

    fmap = feature[0,i].detach().numpy()

    axes[i//4, i%4].imshow(fmap, cmap="gray")
    axes[i//4, i%4].set_title(f"Feature {i+1}")
    axes[i//4, i%4].axis("off")

plt.suptitle("Feature Maps Before ReLU")
plt.tight_layout()
plt.show()

# ==========================================================
# BƯỚC 6. RELU
# ==========================================================

relu = nn.ReLU()

feature_relu = relu(feature)

fig, axes = plt.subplots(2,4, figsize=(10,5))

for i in range(8):

    fmap = feature_relu[0,i].detach().numpy()

    axes[i//4, i%4].imshow(fmap, cmap="gray")
    axes[i//4, i%4].set_title(f"ReLU {i+1}")
    axes[i//4, i%4].axis("off")

plt.suptitle("Feature Maps After ReLU")
plt.tight_layout()
plt.show()

# ==========================================================
# BƯỚC 7. ĐẾM THAM SỐ
# ==========================================================

params = sum(p.numel() for p in conv.parameters())

print("\nSố tham số của Conv2D =", params)

# Công thức:
#
# out_channels × (in_channels × kernel × kernel + bias)
#
# = 8 × (3×3×3 + 1)
#
# = 224

# ==========================================================
# BƯỚC 8. THỬ TĂNG SỐ FILTER
# ==========================================================

conv32 = nn.Conv2d(
    3,
    32,
    kernel_size=3,
    padding=1
)

feature32 = conv32(x)

print("\nOutput với 32 filter")

print(feature32.shape)

params32 = sum(p.numel() for p in conv32.parameters())

print("Số tham số =", params32)

# ==========================================================
# BƯỚC 9. HIỂN THỊ 16 FEATURE MAP ĐẦU
# ==========================================================

fig, axes = plt.subplots(4,4, figsize=(10,10))

for i in range(16):

    fmap = feature32[0,i].detach().numpy()

    axes[i//4, i%4].imshow(fmap, cmap="gray")
    axes[i//4, i%4].set_title(f"Map {i+1}")
    axes[i//4, i%4].axis("off")

plt.suptitle("16 / 32 Feature Maps")
plt.tight_layout()
plt.show()

# ==========================================================
# BƯỚC 10. THÔNG TIN CUỐI
# ==========================================================

print("\n========== KẾT LUẬN ==========")

print("Input :", x.shape)
print("Output (8 filters):", feature.shape)
print("Output (32 filters):", feature32.shape)

print("\n8 Filters  -> 224 tham số")
print("32 Filters -> 896 tham số")