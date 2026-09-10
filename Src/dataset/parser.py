# parser.py
#
# Tu viet ham doc file label dang YOLO format (.txt):
#     class x_center y_center width height   (toa do normalize [0,1])

# LUU Y: file nay CHI doc va tach chuoi, KHONG doi normalize -> pixel.
# Viec doi pixel la nhiem vu cua dataset.py (vi chi dataset.py moi biet
# kich thuoc anh goc va kich thuoc anh sau resize).


def parse_yolo_label(label_path):
    """
    Doc 1 file .txt label dang YOLO format.

    Args:
        label_path: duong dan file .txt

    Returns:
        list cac tuple: (class_id, x_center, y_center, width, height)
        Tat ca toa do VAN o dang normalize [0,1] -- CHUA doi sang pixel.
    """
    objects = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()  # bo ky tu xuong dong \n va khoang trang thua
        if not line:
            continue  # bo qua dong trong

        parts = line.split()
        if len(parts) != 5:
            print(f"[WARNING] Dong khong hop le trong {label_path}: '{line}'")
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        objects.append((class_id, x_center, y_center, width, height))

    return objects


if __name__ == "__main__":
    # ---- TEST BANG FILE LABEL GIA (khong can dataset that) ----
    import tempfile
    import os

    test_content = (
        "5 0.375 0.28125 0.09375 0.09375\n"   # Person
        "0 0.1 0.1 0.05 0.05\n"                # Hardhat
        "\n"                                     # dong trong -- phai bi bo qua
        "invalid line here\n"                    # dong loi -- phai bi bo qua, in WARNING
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(test_content)
        temp_path = f.name

    objects = parse_yolo_label(temp_path)

    print("So object doc duoc:", len(objects))
    for obj in objects:
        print(" ", obj)

    print("\nExpected: 2 object (dong trong va dong loi phai bi bo qua)")

    os.remove(temp_path)