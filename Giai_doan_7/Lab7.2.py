"""
Lab 7.2 - Overfitting co chu dinh
==================================
Muc tieu: Co tinh tao overfitting bang cach chi dung 10-20% du lieu train,
sau do them augmentation + dropout + weight decay de "chua" no, va so sanh
gap train-val truoc/sau.

Yeu cau du lieu: dataset dang ImageFolder theo cau truc giai doan 6:
dataset/
    train/helmet/  train/no_helmet/
    val/helmet/    val/no_helmet/

Sua bien DATA_DIR ben duoi cho dung duong dan cua ban.
"""

import os
import random
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. Cau hinh chung
# ----------------------------------------------------------------------
DATA_DIR = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Chương6\Dataset"          # thu muc goc chua train/ va val/
BATCH_SIZE = 16
NUM_EPOCHS = 50
SUBSET_RATIO = 0.15           # chi lay 15% train set de ep overfitting
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 1. Transform: co augmentation (dung cho ban "sau") va khong (ban "truoc")
# ----------------------------------------------------------------------
no_aug_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

aug_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------------------------------------------------
# 2. Model SimpleCNN (giong Giai doan 5), them tham so dropout_p de
#    de dang bat/tat dropout giua 2 thi nghiem
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.0):
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
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ----------------------------------------------------------------------
# 3. Ham tao subset 15% tu train set goc (giu seed de tai lap duoc)
# ----------------------------------------------------------------------
def make_small_subset(full_dataset, ratio=SUBSET_RATIO):
    n = len(full_dataset)
    subset_size = int(ratio * n)
    indices = random.sample(range(n), subset_size)
    return Subset(full_dataset, indices)


# ----------------------------------------------------------------------
# 4. Ham training loop dung chung cho ca 2 thi nghiem, tra ve history
# ----------------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer,
                 num_epochs=NUM_EPOCHS, tag="experiment"):
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"[{tag}] Epoch {epoch+1:02d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    torch.save(best_state, f"best_model_{tag}.pth")
    return history, best_val_acc


# ----------------------------------------------------------------------
# 5. Load du lieu goc (khong augmentation) de tao subset dung chung
# ----------------------------------------------------------------------
full_train_no_aug = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=no_aug_tf)
full_train_aug = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=aug_tf)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"), transform=val_tf)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Dung CUNG mot bo indices cho ca 2 thi nghiem de dam bao cong bang
random.seed(SEED)
n = len(full_train_no_aug)
subset_size = int(SUBSET_RATIO * n)
shared_indices = random.sample(range(n), subset_size)

small_train_no_aug = Subset(full_train_no_aug, shared_indices)
small_train_aug = Subset(full_train_aug, shared_indices)

train_loader_before = DataLoader(small_train_no_aug, batch_size=BATCH_SIZE, shuffle=True)
train_loader_after = DataLoader(small_train_aug, batch_size=BATCH_SIZE, shuffle=True)

print(f"So anh subset train: {len(small_train_no_aug)} / tong {n} anh goc")

# ----------------------------------------------------------------------
# 6. THI NGHIEM "TRUOC" - khong augmentation, khong dropout, khong weight decay
# ----------------------------------------------------------------------
torch.manual_seed(SEED)
model_before = SimpleCNN(num_classes=2, dropout_p=0.0).to(DEVICE)
optimizer_before = torch.optim.Adam(model_before.parameters(), lr=1e-3)  # KHONG weight_decay

print("\n===== THI NGHIEM TRUOC (khong regularization) =====")
history_before, best_val_before = train_model(
    model_before, train_loader_before, val_loader,
    optimizer_before, num_epochs=NUM_EPOCHS, tag="before")

# ----------------------------------------------------------------------
# 7. THI NGHIEM "SAU" - co augmentation + dropout + weight decay
# ----------------------------------------------------------------------
torch.manual_seed(SEED)
model_after = SimpleCNN(num_classes=2, dropout_p=0.3).to(DEVICE)
optimizer_after = torch.optim.Adam(
    model_after.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n===== THI NGHIEM SAU (co augmentation + dropout + weight decay) =====")
history_after, best_val_after = train_model(
    model_after, train_loader_after, val_loader,
    optimizer_after, num_epochs=NUM_EPOCHS, tag="after")

# ----------------------------------------------------------------------
# 8. Ve 4 duong training curves (2 loss + 2 accuracy, before vs after)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(history_before["train_loss"], label="train_loss")
axes[0, 0].plot(history_before["val_loss"], label="val_loss")
axes[0, 0].set_title("Loss - TRUOC (khong regularization)")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()

axes[0, 1].plot(history_before["train_acc"], label="train_acc")
axes[0, 1].plot(history_before["val_acc"], label="val_acc")
axes[0, 1].set_title("Accuracy - TRUOC (khong regularization)")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()

axes[1, 0].plot(history_after["train_loss"], label="train_loss")
axes[1, 0].plot(history_after["val_loss"], label="val_loss")
axes[1, 0].set_title("Loss - SAU (co regularization)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend()

axes[1, 1].plot(history_after["train_acc"], label="train_acc")
axes[1, 1].plot(history_after["val_acc"], label="val_acc")
axes[1, 1].set_title("Accuracy - SAU (co regularization)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("lab7_2_training_curves.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# 9. Bang so sanh before/after: gap train-val o epoch cuoi
# ----------------------------------------------------------------------
gap_before = history_before["train_acc"][-1] - history_before["val_acc"][-1]
gap_after = history_after["train_acc"][-1] - history_after["val_acc"][-1]

print("\n===== BANG SO SANH BEFORE / AFTER =====")
print(f"{'':25s}{'TRUOC':>15s}{'SAU':>15s}")
print(f"{'train_acc (epoch cuoi)':25s}{history_before['train_acc'][-1]:>15.4f}"
      f"{history_after['train_acc'][-1]:>15.4f}")
print(f"{'val_acc (epoch cuoi)':25s}{history_before['val_acc'][-1]:>15.4f}"
      f"{history_after['val_acc'][-1]:>15.4f}")
print(f"{'gap train-val':25s}{gap_before:>15.4f}{gap_after:>15.4f}")
print(f"{'best_val_acc':25s}{best_val_before:>15.4f}{best_val_after:>15.4f}")

# Xac dinh epoch val_loss bat dau tang (dau hieu overfitting ro nhat)
def find_overfit_epoch(history):
    val_loss = history["val_loss"]
    min_loss = val_loss[0]
    min_epoch = 0
    for i, vl in enumerate(val_loss):
        if vl < min_loss:
            min_loss = vl
            min_epoch = i
    return min_epoch + 1  # epoch tinh tu 1

overfit_epoch_before = find_overfit_epoch(history_before)
print(f"\nEpoch co val_loss thap nhat (TRUOC): {overfit_epoch_before}"
      f" -> sau epoch nay val_loss co xu huong tang = dau hieu overfitting")

# ----------------------------------------------------------------------
# 10. Luu ket qua ra file text de dua vao bao cao
# ----------------------------------------------------------------------
with open("lab7_2_report.txt", "w", encoding="utf-8") as f:
    f.write("LAB 7.2 - OVERFITTING CO CHU DINH - KET QUA\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"So anh subset train: {len(small_train_no_aug)} / {n}\n")
    f.write(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}\n\n")
    f.write("BANG SO SANH:\n")
    f.write(f"{'':25s}{'TRUOC':>15s}{'SAU':>15s}\n")
    f.write(f"{'train_acc cuoi':25s}{history_before['train_acc'][-1]:>15.4f}"
            f"{history_after['train_acc'][-1]:>15.4f}\n")
    f.write(f"{'val_acc cuoi':25s}{history_before['val_acc'][-1]:>15.4f}"
            f"{history_after['val_acc'][-1]:>15.4f}\n")
    f.write(f"{'gap train-val':25s}{gap_before:>15.4f}{gap_after:>15.4f}\n")
    f.write(f"{'best_val_acc':25s}{best_val_before:>15.4f}{best_val_after:>15.4f}\n\n")
    f.write(f"Epoch val_loss thap nhat (TRUOC): {overfit_epoch_before}\n")

print("\nDa luu: lab7_2_training_curves.png, lab7_2_report.txt, "
      "best_model_before.pth, best_model_after.pth")