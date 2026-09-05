"""import numpy as np

img = np.array([
    [0,0,0,0,0,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
], dtype=np.float32)

kernel = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
], dtype=np.float32)

H, W = img.shape
K = 3

out = np.zeros((H-K+1, W-K+1))

for i in range(H-K+1):
    for j in range(W-K+1):
        patch = img[i:i+K, j:j+K]
        out[i,j] = np.sum(patch * kernel)

print(out)"""


"""import numpy as np

# Ảnh đầu vào
img = np.array([
    [0,0,0,0,0,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
], dtype=np.int32)

# Kernel
kernel = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
], dtype=np.int32)

stride = 2
K = kernel.shape[0]
H, W = img.shape

# Tính kích thước output
H_out = (H - K) // stride + 1
W_out = (W - K) // stride + 1

output = np.zeros((H_out, W_out), dtype=np.int32)

print("Ảnh đầu vào:")
print(img)

print("\nKernel:")
print(kernel)

print("\n" + "="*60)

for out_i, i in enumerate(range(0, H-K+1, stride)):
    for out_j, j in enumerate(range(0, W-K+1, stride)):

        patch = img[i:i+K, j:j+K]

        value = np.sum(patch * kernel)
        output[out_i, out_j] = value

        print(f"\nOutput({out_i},{out_j})")
        print(f"Patch lấy từ ảnh: hàng {i}-{i+2}, cột {j}-{j+2}")

        print("\nPatch:")
        print(patch)

        print("\nPatch * Kernel:")
        print(patch * kernel)

        print(f"\nGiá trị Output = {value}")

        print("-"*60)

print("\nFeature Map:")
print(output)"""

"""import numpy as np

# ==========================
# INPUT IMAGE
# ==========================
img = np.array([
    [0,0,0,0,0,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,1,1,1,1,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
], dtype=np.int32)

# ==========================
# KERNEL
# ==========================
kernel = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
], dtype=np.int32)

# ==========================
# THAM SỐ
# ==========================
padding = 1
stride = 2

# ==========================
# THÊM PADDING
# ==========================
img_pad = np.pad(
    img,
    pad_width=padding,
    mode="constant",
    constant_values=0
)

print("="*60)
print("Ảnh sau khi Padding = 1")
print("="*60)
print(img_pad)

# ==========================
# TÍNH OUTPUT SIZE
# ==========================
H, W = img_pad.shape
K = kernel.shape[0]

H_out = (H - K) // stride + 1
W_out = (W - K) // stride + 1

output = np.zeros((H_out, W_out), dtype=np.int32)

print("\nOutput size:", output.shape)

print("\nKernel")
print(kernel)

print("\n" + "="*60)

# ==========================
# CONVOLUTION
# ==========================
for out_i, i in enumerate(range(0, H-K+1, stride)):
    for out_j, j in enumerate(range(0, W-K+1, stride)):

        patch = img_pad[i:i+K, j:j+K]

        mul = patch * kernel
        value = np.sum(mul)

        output[out_i, out_j] = value

        print(f"\nOutput({out_i},{out_j})")
        print(f"Lấy patch: hàng {i}-{i+2}, cột {j}-{j+2}")

        print("\nPatch:")
        print(patch)

        print("\nPatch * Kernel:")
        print(mul)

        print("\nPhép tính:")

        expression = []

        for r in range(K):
            for c in range(K):
                expression.append(
                    f"{patch[r,c]}×({kernel[r,c]})"
                )

        print(" + ".join(expression))

        print(f"\nOutput = {value}")

        print("-"*60)

print("\nFeature Map")
print(output)"""


import numpy as np
img = np.array([
 [0,0,0,0,0,0],
 [0,1,1,1,1,0],
 [0,1,1,1,1,0],
 [0,1,1,1,1,0],
 [0,0,0,0,0,0],
 [0,0,0,0,0,0]], dtype=np.float32)
kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
H, W = img.shape
K = 3
out = np.zeros((H-K+1, W-K+1), dtype=np.float32)
for i in range(H-K+1):
    for j in range(W-K+1):
        patch = img[i:i+K, j:j+K]
        out[i,j] = np.sum(patch * kernel)
print(out)
