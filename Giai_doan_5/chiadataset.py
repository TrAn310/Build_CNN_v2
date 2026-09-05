import random
import shutil
from pathlib import Path


# =========================
# CẤU HÌNH
# =========================

SOURCE_DIR = Path(__file__).parent / "Dataset"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

CLASSES = [
    "co_khau_trang",
    "khong_khau_trang"
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]


# =========================
# KIỂM TRA TỶ LỆ
# =========================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0


# =========================
# RANDOM CỐ ĐỊNH
# =========================

random.seed(SEED)


# =========================
# TẠO FOLDER
# =========================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = SOURCE_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


# =========================
# CHIA DATASET
# =========================

for class_name in CLASSES:

    class_dir = SOURCE_DIR / class_name

    images = [
        file for file in class_dir.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Trộn ảnh
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")

    # Copy ảnh
    for image in train_images:
        shutil.copy2(
            image,
            SOURCE_DIR / "train" / class_name / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            SOURCE_DIR / "val" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            SOURCE_DIR / "test" / class_name / image.name
        )


print("\nChia dataset hoàn tất!")