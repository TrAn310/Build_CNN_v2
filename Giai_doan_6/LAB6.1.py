from pathlib import Path
import matplotlib.pyplot as plt


# =========================================================
# 1. ĐƯỜNG DẪN DATASET
# =========================================================

# Lấy thư mục chứa file lab6_1.py
BASE_DIR = Path(__file__).parent

# Dataset nằm cùng cấp với lab6_1.py
DATASET_DIR = BASE_DIR / "Dataset"


# =========================================================
# 2. CẤU HÌNH DATASET
# =========================================================

splits = [
    "train",
    "val",
    "test"
]

classes = [
    "co_khau_trang",
    "khong_khau_trang"
]


# Các định dạng ảnh được chấp nhận
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# 3. HÀM ĐẾM ẢNH
# =========================================================

def count_images(folder):

    count = 0

    for file in folder.iterdir():

        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ):
            count += 1

    return count


# =========================================================
# 4. THỐNG KÊ DATASET
# =========================================================

stats = {}


for split in splits:

    stats[split] = {}

    for class_name in classes:

        folder = DATASET_DIR / split / class_name

        if not folder.exists():

            print(
                f"[WARNING] Không tìm thấy: {folder}"
            )

            stats[split][class_name] = 0

            continue

        count = count_images(folder)

        stats[split][class_name] = count


# =========================================================
# 5. IN KẾT QUẢ
# =========================================================

print("\n")
print("=" * 60)
print("           DATASET AUDIT - LAB 6.1")
print("=" * 60)

print(f"Dataset path: {DATASET_DIR}")

print("\n")


for split in splits:

    print(f"--- {split.upper()} ---")

    total = 0

    for class_name in classes:

        count = stats[split][class_name]

        total += count

        print(
            f"{class_name:<20}: {count:>6} ảnh"
        )

    print(
        f"{'TỔNG':<20}: {total:>6} ảnh"
    )

    print()


# =========================================================
# 6. TỔNG DATASET
# =========================================================

print("=" * 60)

grand_total = 0

for split in splits:

    for class_name in classes:

        grand_total += stats[split][class_name]


print(
    f"TỔNG TOÀN BỘ DATASET: {grand_total} ảnh"
)

print("=" * 60)


# =========================================================
# 7. KIỂM TRA CÂN BẰNG CLASS
# =========================================================

print("\n")
print("=" * 60)
print("              CLASS BALANCE")
print("=" * 60)


for split in splits:

    with_mask = stats[split]["co_khau_trang"]

    without_mask = stats[split]["khong_khau_trang"]

    total = with_mask + without_mask

    print(f"\n{split.upper()}")

    if total == 0:

        print("Không có ảnh.")

        continue

    with_percent = with_mask / total * 100
    without_percent = without_mask / total * 100

    print(
        f"co_khau_trang    : "
        f"{with_mask} ({with_percent:.2f}%)"
    )

    print(
        f"khong_khau_trang : "
        f"{without_mask} ({without_percent:.2f}%)"
    )


# =========================================================
# 8. VẼ BIỂU ĐỒ PHÂN BỐ DATASET
# =========================================================

x = range(len(classes))

width = 0.25


train_values = [
    stats["train"][class_name]
    for class_name in classes
]


val_values = [
    stats["val"][class_name]
    for class_name in classes
]


test_values = [
    stats["test"][class_name]
    for class_name in classes
]


plt.figure(figsize=(9, 6))


plt.bar(
    [i - width for i in x],
    train_values,
    width=width,
    label="Train"
)


plt.bar(
    x,
    val_values,
    width=width,
    label="Validation"
)


plt.bar(
    [i + width for i in x],
    test_values,
    width=width,
    label="Test"
)


plt.xticks(
    list(x),
    classes
)


plt.xlabel("Class")

plt.ylabel("Number of images")

plt.title(
    "Dataset Distribution - Mask Classification"
)

plt.legend()

plt.tight_layout()


# =========================================================
# 9. LƯU BIỂU ĐỒ
# =========================================================

output_path = BASE_DIR / "class_distribution.png"

plt.savefig(
    output_path,
    dpi=300
)

print("\n")
print(
    f"Đã lưu biểu đồ tại:\n{output_path}"
)


plt.show()






from PIL import Image


def check_corrupted_images():

    print("\n")
    print("=" * 60)
    print("           KIỂM TRA ẢNH HỎNG")
    print("=" * 60)

    corrupted = []

    for split in splits:

        for class_name in classes:

            folder = DATASET_DIR / split / class_name

            for image_path in folder.iterdir():

                if (
                    not image_path.is_file()
                    or image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                try:

                    with Image.open(image_path) as img:
                        img.verify()

                except Exception:

                    corrupted.append(
                        image_path
                    )

    print(
        f"\nTổng số ảnh lỗi: {len(corrupted)}"
    )

    if len(corrupted) == 0:

        print("Không phát hiện ảnh bị lỗi.")

    else:

        print("\nDanh sách ảnh lỗi:")

        for image_path in corrupted:

            print(image_path)

    return corrupted
corrupted_images = check_corrupted_images()



