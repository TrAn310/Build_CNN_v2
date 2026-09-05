import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # Thêm thư viện hiển thị ảnh


# CẤU HÌNH & PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

CLASSES = ['co_khau_trang', 'khong_khau_trang']

# Sử dụng dấu / để không bị lỗi đường dẫn
IMAGE_PATH = "F:/NCKH_2026/lapTrinhPy/CNN_v2/Chuong5/test2.jpg"  


# ĐỊNH NGHĨA MODEL & LOAD WEIGHTS[cite: 1]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Khởi tạo model và load trọng số
model = SimpleCNN(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()  # Chuyển sang chế độ đánh giá[cite: 1]


# TIỀN XỬ LÝ ẢNH & DỰ ĐOÁN

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Mở ảnh
image = Image.open(IMAGE_PATH).convert('RGB')

# Biến đổi thành tensor đưa vào model[cite: 1]
img_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(img_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    pred_idx = logits.argmax(dim=1).item()
    confidence = probabilities[pred_idx].item() * 100

predicted_label = CLASSES[pred_idx]


# IN KẾT QUẢ OUT CONSOLE

print(f"Ảnh test    : {IMAGE_PATH}")
print(f"Dự đoán     : {predicted_label}")
print(f"Độ tin cậy  : {confidence:.2f}%")


# HIỂN THỊ CỬA SỔ ẢNH TRỰC QUAN

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Predict: {predicted_label} ({confidence:.1f}%)", fontsize=14, color="green")
plt.axis("off")  # Tắt trục tọa độ px
plt.show()      # Bật cửa sổ hiển thị ảnh