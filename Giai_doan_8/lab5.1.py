# IMPORT THƯ VIỆN
import torch                                   # thư viện chính của PyTorch (tensor, autograd...)
import torch.nn as nn                          # chứa các lớp mạng: Conv2d, Linear, ReLU...
import torch.optim as optim                    # chứa các thuật toán tối ưu: Adam, SGD...
from torch.utils.data import DataLoader        # công cụ chia dữ liệu thành từng batch
import torchvision                             # thư viện phụ trợ cho ảnh: dataset, model có sẵn
import torchvision.transforms as transforms    # các phép biến đổi ảnh: resize, augmentation...
import matplotlib.pyplot as plt                # vẽ biểu đồ

# CẤU HÌNH CHUNG

torch.manual_seed(42)
# Cố định "hạt giống ngẫu nhiên" -> mỗi lần chạy lại, các phần có random
# (shuffle dữ liệu, khởi tạo trọng số, dropout...) sẽ cho kết quả gần giống nhau,
# giúp dễ so sánh khi bạn thử nghiệm thay đổi hyperparameter.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Kiểm tra máy có GPU (CUDA) không.
# Có GPU -> DEVICE = "cuda" (train nhanh hơn nhiều).
# Không có -> DEVICE = "cpu" (vẫn chạy được, chỉ chậm hơn).

DATA_DIR = "F:/NCKH_2026/lapTrinhPy/CNN_v2/Chuong5/Dataset"
# Đường dẫn tới thư mục Dataset, bên trong có 3 folder con: train / val / test


BATCH_SIZE = 32     # số ảnh xử lý cùng lúc trong 1 lần cập nhật trọng số
NUM_EPOCHS = 15      # số lần model học qua TOÀN BỘ tập train
LR = 1e-3            # learning rate - độ lớn bước nhảy mỗi lần cập nhật trọng số


# TRANSFORM (tiền xử lý + augmentation ảnh)

train_tf = transforms.Compose([
    # Compose = gộp nhiều phép biến đổi lại, áp dụng theo đúng thứ tự liệt kê

    transforms.Resize((128, 128)),
    # Resize mọi ảnh về đúng 128x128, vì ảnh gốc trong dataset có thể
    # kích thước khác nhau, nhưng CNN yêu cầu input cùng 1 kích thước cố định.

    transforms.RandomHorizontalFlip(p=0.5),
    # Augmentation: với xác suất 50%, lật ảnh theo chiều ngang.
    # Giúp model không học "vị trí trái/phải" một cách máy móc, tăng khả năng
    # tổng quát hóa. Đây là augmentation hợp lý vì người đeo/không đeo khẩu
    # trang khi lật ngang vẫn còn hợp lý về mặt hình ảnh.

    transforms.ToTensor(),
    # Chuyển ảnh (đang ở dạng PIL Image, pixel 0-255) thành Tensor PyTorch,
    # đồng thời tự động chia pixel về khoảng [0, 1] và đổi thứ tự chiều
    # từ HWC (Height-Width-Channel) sang CHW (Channel-Height-Width) mà
    # PyTorch yêu cầu.
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),   # vẫn phải resize giống hệt train
    transforms.ToTensor(),           # vẫn phải chuyển tensor giống hệt train
    # KHÔNG có RandomHorizontalFlip ở đây.
    # Lý do: tập val/test dùng để ĐÁNH GIÁ khách quan, phải giữ nguyên ảnh
    # gốc, không được augmentation ngẫu nhiên - nếu không, mỗi lần đánh giá
    # sẽ ra kết quả khác nhau, không công bằng và không phản ánh đúng model.
])


# TẠO DATASET + DATALOADER
# Xét qua từng ảnh xong gán nhãn tương ứng
train_ds = torchvision.datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds   = torchvision.datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_tf)
test_ds  = torchvision.datasets.ImageFolder(f"{DATA_DIR}/test",  transform=val_tf)
# ImageFolder tự động:
#  - Quét các folder con trong DATA_DIR/train (co_khau_trang, khong_khau_trang)
#  - Coi mỗi folder là 1 class, gán nhãn số nguyên theo thứ tự alphabet
#  - Mỗi lần lấy 1 ảnh, tự áp dụng "transform" tương ứng đã truyền vào

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# shuffle=True: xáo trộn thứ tự ảnh mỗi epoch.
# Quan trọng cho tập TRAIN: tránh model học theo thứ tự ảnh cố định,
# giúp gradient mỗi batch đa dạng hơn -> học tốt hơn.

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
# shuffle=False cho val/test: không cần xáo trộn vì không dùng để train,
# giữ thứ tự cố định giúp dễ debug/so sánh giữa các lần chạy.

print("Số ảnh train:", len(train_ds))
print("Số ảnh val:", len(val_ds))
print("Số ảnh test:", len(test_ds))
print("Class mapping:", train_ds.class_to_idx)
# class_to_idx trả về dict kiểu {'co_khau_trang': 0, 'khong_khau_trang': 1}
# Đây là mapping QUAN TRỌNG - dùng để suy ngược từ số (0/1) ra tên lớp thật.

# KIỂM TRA SHAPE 1 BATCH (sanity check trước khi train)

sample_images, sample_labels = next(iter(train_loader))
# Lấy thử 1 batch đầu tiên ra để kiểm tra, không ảnh hưởng gì tới training
# loop phía dưới (vòng for sau này sẽ tự lấy batch mới từ đầu).

print("images shape:", sample_images.shape)   
print("labels shape:", sample_labels.shape)  
# [32, 3, 128, 128]:
#   32  = batch size
#   3   = số channel màu (RGB)
#   128, 128 = chiều cao, chiều rộng ảnh sau resize

# ĐỊNH NGHĨA KIẾN TRÚC SimpleCNN

class SimpleCNN(nn.Module):
    # Mọi model PyTorch đều kế thừa từ nn.Module

    def __init__(self, num_classes=2):
        # __init__: nơi KHAI BÁO các layer sẽ dùng (chưa chạy dữ liệu qua)
        super().__init__()
        # Bắt buộc gọi để nn.Module khởi tạo cơ chế nội bộ (theo dõi tham số, gradient...) trước khi bạn tự thêm layer của mình.

        self.features = nn.Sequential(
            # nn.Sequential: xếp các layer chạy NỐI TIẾP theo đúng thứ tự viết

            nn.Conv2d(3, 16, 3, padding=1),
            # Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            #   in=3    : ảnh RGB đầu vào có 3 channel
            #   out=16  : dùng 16 bộ lọc (filter) -> output có 16 channel
            #   kernel=3: mỗi filter kích thước 3x3
            #   padding=1: thêm viền 1 pixel quanh ảnh để giữ nguyên
            #              kích thước không gian sau conv (128 vẫn ra 128)
            nn.ReLU(),
            # ReLU(x) = max(0, x). Thêm tính phi tuyến; nếu không có, xếp
            # nhiều Conv2D liên tiếp cũng chỉ tương đương 1 phép tuyến tính.
            nn.MaxPool2d(2),
            # Lấy giá trị lớn nhất trong mỗi ô 2x2 -> giảm kích thước ảnh
            # còn một nửa: 128x128 -> 64x64. Giúp giảm tính toán và tăng
            # receptive field (vùng ảnh gốc mà 1 neuron "nhìn thấy").

            nn.Conv2d(16, 32, 3, padding=1),
            # Nhận 16 channel từ block trước, tạo ra 32 channel mới.
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 64x64 -> 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            # Nhận 32 channel, tạo ra 64 channel.
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32x32 -> 16x16
        )
        # Sau 3 block: tensor có shape [batch, 64, 16, 16]
        # Ý tưởng: càng vào sâu, số channel càng TĂNG (học đặc trưng phức
        # tạp hơn), còn kích thước không gian càng GIẢM (thu gọn thông tin).

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # "Duỗi thẳng" tensor [batch, 64, 16, 16] thành [batch, 64*16*16] = [batch, 16384], vì lớp Linear chỉ nhận vector 1 chiều.

            nn.Linear(64 * 16 * 16, 128),
            # Fully-connected: nén 16384 giá trị xuống còn 128, học tổ hợp các đặc trưng để phục vụ phân loại.

            nn.ReLU(),
            # Phi tuyến lần nữa cho lớp fully-connected.

            nn.Dropout(0.3),
            # Trong lúc TRAIN: ngẫu nhiên "tắt" 30% neuron mỗi lần forward,
            # ép model không phụ thuộc quá mức vào vài neuron cụ thể
            # -> chống overfitting.
            # Trong lúc EVAL (model.eval()): Dropout tự động tắt, dùng
            # 100% neuron.

            nn.Linear(128, num_classes),
            # Lớp cuối: ra đúng num_classes=2 con số (logits), tương ứng
            # 2 lớp co_khau_trang / khong_khau_trang. Đây CHƯA phải xác
            # suất, chỉ là điểm số thô.
        )

    def forward(self, x):
        # forward: định nghĩa dữ liệu CHẢY QUA model như thế nào.
        # PyTorch tự động gọi hàm này khi bạn viết model(x).
        x = self.features(x)      # trích xuất đặc trưng (feature extraction)
        return self.classifier(x)  # phân loại dựa trên đặc trưng đã trích


# KHỞI TẠO MODEL, LOSS, OPTIMIZER

model = SimpleCNN(num_classes=2).to(DEVICE)
# Tạo model, rồi đẩy toàn bộ trọng số lên DEVICE (GPU nếu có, không thì CPU).

criterion = nn.CrossEntropyLoss()
# Hàm loss chuẩn cho bài toán phân loại nhiều lớp (kể cả 2 lớp).
# Nhận logits thô (chưa qua softmax) + label thật (số nguyên 0/1),
# tự làm log-softmax + negative-log-likelihood bên trong theo cách ổn
# định số học.

optimizer = optim.Adam(model.parameters(), lr=LR)
# Adam: thuật toán cập nhật trọng số dựa trên gradient, thường hội tụ
# nhanh và ổn định hơn SGD thường trong nhiều bài toán.
# model.parameters(): lấy toàn bộ trọng số của model để optimizer biết
# cần cập nhật những gì.

total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số: {total_params:,}")
# p.numel(): đếm số phần tử (số trọng số) trong 1 tensor tham số.
# Cộng dồn tất cả layer -> tổng số tham số toàn model.


# TRAINING LOOP (huấn luyện + validate + lưu checkpoint tốt nhất)

best_val = 0.0
# Biến ghi nhớ val_accuracy CAO NHẤT đạt được tính tới hiện tại.
# Bắt đầu = 0 vì model chưa học gì cả.

history = {"train_loss": [], "train_acc": [], "val_acc": []}
# Dict lưu lại lịch sử qua từng epoch, dùng để vẽ biểu đồ ở bước sau.

for epoch in range(1, NUM_EPOCHS + 1):
    # Lặp NUM_EPOCHS=15 lần. range(1, 16) để in số đẹp từ 1 đến 15.

    # -------------------- PHẦN TRAIN (dạy model) --------------------
    model.train()
    # Bật chế độ HỌC: Dropout hoạt động ngẫu nhiên tắt neuron.

    running_loss, correct, total = 0.0, 0, 0
    # Reset bộ đếm về 0 vào đầu MỖI epoch:
    #   running_loss: tổng loss cộng dồn qua các batch trong epoch này
    #   correct     : tổng số ảnh đoán đúng trong epoch này
    #   total       : tổng số ảnh đã xử lý trong epoch này

    for images, labels in train_loader:
        # train_loader chia 5286 ảnh thành từng batch 32 ảnh.
        # Vòng lặp này chạy khoảng 5286/32 ~ 165 lần cho tới khi hết ảnh.

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Đưa batch ảnh/nhãn hiện tại lên cùng DEVICE với model
        # (bắt buộc, nếu không sẽ lỗi "tensor không cùng device").

        optimizer.zero_grad()
        # Xóa gradient CŨ đang lưu trong model.
        # PyTorch mặc định CỘNG DỒN gradient qua mỗi lần backward(),
        # nếu không xóa, gradient batch này sẽ cộng nhầm với batch trước.

        logits = model(images)
        # Forward: đưa 32 ảnh qua model, nhận về logits shape [32, 2].

        loss = criterion(logits, labels)
        # So sánh dự đoán (logits) với đáp án thật (labels)
        # -> ra 1 con số duy nhất đại diện "model đang sai bao nhiêu".

        loss.backward()
        # Backpropagation: PyTorch tự tính gradient - tức là "mỗi trọng số
        # cần thay đổi theo hướng nào, bao nhiêu" để giảm loss.
        # Chỉ TÍNH TOÁN, chưa thay đổi trọng số.

        optimizer.step()
        # Thực sự CẬP NHẬT trọng số, dựa theo gradient vừa tính,
        # theo công thức của thuật toán Adam.

        # Ghi sổ điểm (không liên quan tới việc học) 
        running_loss += loss.item() * images.size(0)
        # loss.item(): lấy loss trung bình của batch này ra số Python thường.
        # Nhân với images.size(0) (=32, số ảnh trong batch) để ra TỔNG loss,
        # vì sẽ tự chia lại đúng cách ở cuối epoch.

        correct += (logits.argmax(1) == labels).sum().item()
        # logits.argmax(1): với mỗi ảnh, chọn class có điểm số cao hơn
        #                    (0 hoặc 1) -> đó là dự đoán của model.
        # == labels: so với nhãn thật, ra tensor True/False.
        # .sum().item(): đếm số True (đoán đúng) trong batch này.

        total += labels.size(0)
        # Cộng thêm số ảnh trong batch (32) vào tổng số ảnh đã xử lý.

    train_loss = running_loss / total
    # Loss trung bình MỖI ẢNH trong cả epoch (tổng loss / tổng số ảnh).
    train_acc = correct / total
    # Tỉ lệ % đoán đúng trên tập TRAIN của epoch này.

    # -------------------- PHẦN VALIDATE (kiểm tra, không dạy) --------------------
    model.eval()
    # Chuyển sang chế độ KIỂM TRA: Dropout tự động tắt, dùng 100% neuron.

    v_correct, v_total = 0, 0
    # Reset bộ đếm riêng cho validation (đặt tên v_ để phân biệt với train).

    with torch.no_grad():
        # Tắt hẳn việc TÍNH GRADIENT trong khối lệnh này.
        # Vì đây chỉ là kiểm tra, không sửa trọng số -> tắt đi giúp chạy
        # nhanh hơn và tốn ít bộ nhớ hơn.

        for images, labels in val_loader:
            # val_loader chứa 1132 ảnh model CHƯA TỪNG THẤY lúc train.
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            # Chỉ forward để lấy dự đoán, KHÔNG có zero_grad/backward/step
            # -> không học gì trong bước này, chỉ đánh giá.

            v_correct += (logits.argmax(1) == labels).sum().item()
            v_total += labels.size(0)

    val_acc = v_correct / v_total
    # Tỉ lệ % đoán đúng trên tập VALIDATION (ảnh lạ, chưa học qua)
    # -> con số này phản ánh khả năng model học "thật" hay học "vẹt".

    # -------------------- Lưu lịch sử + in kết quả --------------------
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    # Lưu 3 con số vừa tính vào history, dùng vẽ biểu đồ đường sau này.

    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} "
          f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    # {epoch:2d}: canh số epoch đều 2 chữ số cho đẹp.
    # {train_loss:.4f}: làm tròn 4 chữ số thập phân.

    # -------------------- Lưu checkpoint tốt nhất --------------------
    if val_acc > best_val:
        # Nếu val_acc epoch này CAO HƠN kỷ lục cũ:
        best_val = val_acc
        # Cập nhật kỷ lục mới.

        torch.save(model.state_dict(), "best_model.pth")
        # model.state_dict(): lấy toàn bộ trọng số hiện tại của model
        #                      (dạng dict tên_layer -> tensor trọng số).
        # torch.save(...): ghi trọng số đó ra file trên ổ đĩa.
        # Lý do phải lưu kiểu này: model có thể học giỏi ở epoch 8 nhưng
        # đến epoch 15 lại overfit (val giảm xuống). Lưu theo cách này
        # đảm bảo LUÔN giữ được phiên bản tốt nhất từng đạt được, không
        # phải model của epoch cuối cùng.

        print(f"  -> Lưu checkpoint mới, val_acc={val_acc:.4f}")

print(f"\nBest validation accuracy: {best_val:.4f}")
# Sau khi chạy hết 15 epoch, in ra con số val_accuracy tốt nhất đạt được
# trong suốt quá trình train.

# VẼ BIỂU ĐỒ TRAINING CURVES

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Tạo 1 figure với 2 ô đồ thị (subplot) đặt cạnh nhau (1 hàng, 2 cột).

axes[0].plot(history["train_loss"], marker="o")
# Vẽ đường loss qua từng epoch, marker="o" để đánh dấu tròn tại mỗi điểm.
axes[0].set_title("Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)
# Thêm lưới mờ (alpha=0.3) cho dễ đọc giá trị.

axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(history["val_acc"], label="Val Acc", marker="o")
# Vẽ 2 đường trên cùng 1 biểu đồ để so sánh trực tiếp train vs val.
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
# Hiện chú thích (label) để phân biệt 2 đường.
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# Tự động canh khoảng cách giữa các subplot cho không bị chồng chữ.
plt.savefig("training_curves.png", dpi=150)
# Lưu ảnh ra file, dpi=150 cho ảnh nét.
plt.show()
# Hiện cửa sổ biểu đồ lên màn hình.

# LOAD LẠI BEST CHECKPOINT + INFERENCE 20 ẢNH TEST

model.load_state_dict(torch.load("best_model.pth"))
# QUAN TRỌNG: sau vòng lặp 15 epoch, biến "model" đang giữ trọng số của
# EPOCH CUỐI CÙNG, không phải trọng số tốt nhất. Phải load lại từ file
# best_model.pth để đảm bảo dùng đúng phiên bản tốt nhất khi inference.

model.eval()
# Chuyển sang chế độ kiểm tra (tắt Dropout) trước khi dự đoán thật.

idx_to_class = {v: k for k, v in test_ds.class_to_idx.items()}
# Đảo ngược dict class_to_idx: từ {'co_khau_trang': 0, ...}
# thành {0: 'co_khau_trang', ...} để tiện tra cứu tên lớp từ số dự đoán.

fig, axes = plt.subplots(4, 5, figsize=(13, 11))
# Lưới 4 hàng x 5 cột = 20 ô, mỗi ô hiển thị 1 ảnh.
axes = axes.flatten()
# Chuyển mảng 2 chiều (4x5) thành mảng 1 chiều (20 phần tử) cho dễ lặp qua.

with torch.no_grad():
    # Không cần tính gradient khi chỉ inference (dự đoán).
    for i, ax in enumerate(axes):
        # Lặp qua 20 ô đồ thị, lấy luôn 20 ảnh đầu tiên trong test_ds.
        img, true_label = test_ds[i]
        # test_ds[i]: lấy ảnh thứ i (đã qua transform) + nhãn thật của nó.

        pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1).item()
        # img.unsqueeze(0): thêm 1 chiều batch giả ở đầu, vì model luôn
        #                    kỳ vọng input dạng [batch, C, H, W], còn img
        #                    hiện tại chỉ là [C, H, W] (1 ảnh đơn lẻ).
        # model(...): forward ra logits shape [1, 2].
        # .argmax(1): chọn class có điểm cao hơn -> ra số 0 hoặc 1.
        # .item(): chuyển tensor 1 phần tử thành số Python thường.

        ax.imshow(img.permute(1, 2, 0))
        # img đang ở dạng [C, H, W] (chuẩn PyTorch), nhưng matplotlib cần
        # dạng [H, W, C] để hiển thị ảnh đúng -> permute đổi lại thứ tự trục.

        color = "green" if pred == true_label else "red"
        # Đoán đúng -> tiêu đề màu xanh; đoán sai -> màu đỏ, dễ nhìn ra
        # ngay ảnh nào model bị sai.

        ax.set_title(f"GT:{idx_to_class[true_label]}\nPred:{idx_to_class[pred]}",
                     fontsize=8, color=color)
        # GT = Ground Truth (nhãn thật), Pred = nhãn model dự đoán.

        ax.axis("off")
        # Ẩn trục tọa độ (số 0,1,2...) quanh ảnh cho gọn.

plt.tight_layout()
plt.savefig("inference_grid.png", dpi=150)
plt.show()

print("\nHoàn thành Lab 5.1. Các file đã tạo:")
print("  - best_model.pth        (checkpoint tốt nhất)")
print("  - training_curves.png   (biểu đồ loss + accuracy)")
print("  - inference_grid.png    (20 ảnh test + dự đoán)")