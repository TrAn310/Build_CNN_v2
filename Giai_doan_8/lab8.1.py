import os
import csv
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Giai_doan_8\best_model.pth"

TEST_DIR = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Giai_doan_5\Dataset\test"

OUTPUT_DIR = "outputs/lab8_1"

BATCH_SIZE = 32

CLASSES = [
    "co_khau_trang",
    "khong_khau_trang"
]

NUM_CLASSES = 2


# Tạo thư mục output
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("LAB 8.1 - CONFUSION MATRIX VA METRIC")
print("=" * 70)

print(f"Device     : {DEVICE}")
print(f"Model      : {MODEL_PATH}")
print(f"Test set   : {TEST_DIR}")
print(f"Output dir : {OUTPUT_DIR}")


# ============================================================
# 2. ĐỊNH NGHĨA MODEL
# ============================================================

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

            nn.Linear(
                64 * 16 * 16,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            ),
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# 3. LOAD BEST MODEL
# ============================================================

model = SimpleCNN(
    num_classes=NUM_CLASSES
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("\nBest model loaded successfully.")


# ============================================================
# 4. TEST DATASET
# ============================================================

transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),
])


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\n" + "=" * 70)
print("TEST DATASET INFORMATION")
print("=" * 70)

print(f"Test samples : {len(test_dataset)}")
print(f"Classes      : {test_dataset.classes}")
print(f"Batch size   : {BATCH_SIZE}")


# ============================================================
# 5. CHẠY MODEL TRÊN TOÀN BỘ TEST SET
# ============================================================

all_labels = []
all_preds = []
all_paths = []


model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_preds.extend(
            predictions.cpu().numpy().tolist()
        )


# Lấy đường dẫn ảnh theo đúng thứ tự của ImageFolder
all_paths = [
    path
    for path, label in test_dataset.samples
]


print("\nTest evaluation completed.")


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ------------------------------------------------------------
# Vẽ confusion matrix
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Lab 8.1 - Confusion Matrix")

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


print(f"\nConfusion matrix saved to:")
print(cm_path)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASSES,
    digits=4
)


accuracy = accuracy_score(
    all_labels,
    all_preds
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

print(f"Accuracy: {accuracy:.4f}")


report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "LAB 8.1 - CLASSIFICATION REPORT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(report)

    f.write(
        f"\nAccuracy: {accuracy:.4f}\n"
    )


# ============================================================
# 8. XÁC ĐỊNH TP / TN / FP / FN
# ============================================================

tn, fp, fn, tp = cm.ravel()


print("\n" + "=" * 70)
print("TP / TN / FP / FN")
print("=" * 70)

print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")


# ============================================================
# 9. LIỆT KÊ FALSE NEGATIVE
# ============================================================

# Positive = khong_khau_trang = class 1
#
# FN:
# Ground Truth = khong_khau_trang
# Prediction    = co_khau_trang

false_negatives = []

for i in range(len(all_labels)):

    true_label = all_labels[i]
    pred_label = all_preds[i]

    if true_label == 1 and pred_label == 0:

        false_negatives.append({
            "index": i,
            "path": all_paths[i],
            "true_label": CLASSES[true_label],
            "pred_label": CLASSES[pred_label]
        })


# ============================================================
# 10. LIỆT KÊ FALSE POSITIVE
# ============================================================

# FP:
# Ground Truth = co_khau_trang
# Prediction    = khong_khau_trang

false_positives = []

for i in range(len(all_labels)):

    true_label = all_labels[i]
    pred_label = all_preds[i]

    if true_label == 0 and pred_label == 1:

        false_positives.append({
            "index": i,
            "path": all_paths[i],
            "true_label": CLASSES[true_label],
            "pred_label": CLASSES[pred_label]
        })


print("\n" + "=" * 70)
print("ERROR CASES")
print("=" * 70)

print(
    f"False Negative : {len(false_negatives)}"
)

print(
    f"False Positive : {len(false_positives)}"
)


# ============================================================
# 11. LƯU DANH SÁCH FN + FP
# ============================================================

error_csv_path = os.path.join(
    OUTPUT_DIR,
    "error_cases.csv"
)


with open(
    error_csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "type",
        "index",
        "image_path",
        "true_label",
        "pred_label",
        "failure_reason"
    ])


    for item in false_negatives:

        writer.writerow([
            "FN",
            item["index"],
            item["path"],
            item["true_label"],
            item["pred_label"],
            ""
        ])


    for item in false_positives:

        writer.writerow([
            "FP",
            item["index"],
            item["path"],
            item["true_label"],
            item["pred_label"],
            ""
        ])


print(
    f"\nError cases saved to:"
)

print(error_csv_path)


# ============================================================
# 12. HÀM TẠO CONTACT SHEET
# ============================================================

def create_contact_sheet(
    error_cases,
    output_path,
    title,
    columns=5,
    image_size=160
):

    if len(error_cases) == 0:

        print(
            f"\nNo error cases for: {title}"
        )

        return


    rows = (
        len(error_cases) + columns - 1
    ) // columns


    margin = 20

    title_height = 50

    cell_width = image_size + margin

    cell_height = image_size + 50


    sheet_width = (
        columns * cell_width
    )

    sheet_height = (
        title_height +
        rows * cell_height
    )


    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )


    draw = ImageDraw.Draw(sheet)


    # Tiêu đề
    draw.text(
        (10, 10),
        title,
        fill="black"
    )


    for i, item in enumerate(error_cases):

        path = item["path"]

        try:

            image = Image.open(
                path
            ).convert("RGB")

            image.thumbnail(
                (
                    image_size,
                    image_size
                )
            )

            x = (
                (i % columns)
                * cell_width
                + margin // 2
            )

            y = (
                title_height
                + (i // columns)
                * cell_height
            )


            sheet.paste(
                image,
                (
                    x,
                    y
                )
            )


            text = (
                f"{i}: "
                f"True={item['true_label']}\n"
                f"Pred={item['pred_label']}"
            )


            draw.multiline_text(
                (
                    x,
                    y + image_size + 5
                ),
                text,
                fill="black"
            )


        except Exception as e:

            print(
                f"Cannot open image: {path}"
            )

            print(e)


    sheet.save(
        output_path
    )


# ============================================================
# 13. CONTACT SHEET FALSE NEGATIVE
# ============================================================

fn_sheet_path = os.path.join(
    OUTPUT_DIR,
    "false_negative_contact_sheet.png"
)


create_contact_sheet(
    false_negatives,
    fn_sheet_path,
    "False Negative - No Helmet equivalent: khong_khau_trang"
)


# ============================================================
# 14. CONTACT SHEET FALSE POSITIVE
# ============================================================

fp_sheet_path = os.path.join(
    OUTPUT_DIR,
    "false_positive_contact_sheet.png"
)


create_contact_sheet(
    false_positives,
    fp_sheet_path,
    "False Positive"
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LAB 8.1 COMPLETED")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")

print(f"TN              : {tn}")
print(f"FP              : {fp}")
print(f"FN              : {fn}")
print(f"TP              : {tp}")

print(
    f"\nFalse Negative : {len(false_negatives)}"
)

print(
    f"False Positive : {len(false_positives)}"
)

print("\nOutput files:")

print(
    f"- {cm_path}"
)

print(
    f"- {report_path}"
)

print(
    f"- {error_csv_path}"
)

print(
    f"- {fn_sheet_path}"
)

print(
    f"- {fp_sheet_path}"
)

print("=" * 70)