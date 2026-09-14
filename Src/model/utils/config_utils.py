"""
config_utils.py

Đọc data.yaml của dataset để lấy:
    - num_classes, class_names (load_dataset_config)
    - duong dan anh/label cua tap train + valid, kieu file data.yaml
      CHUAN cua YOLO (resolve_dataset_paths)

Viec nay giup num_classes KHONG bi hard-code trong code model, ma lay
dong tu chinh dataset dang dung. Tuong tu, run_train.py (entry point
kieu YOLO) chi can 1 tham so --data la du, khong phai tu go 4 duong
dan img/label rieng le.
"""

import os

import yaml


def load_dataset_config(yaml_path):
    """
    Input:
        yaml_path: đường dẫn tới file data.yaml

    Output:
        num_classes: int
        class_names: list[str], ví dụ ['Person', 'Helmet', 'Vest']

    File data.yaml kỳ vọng có dạng:
        nc: 3
        names: ['Person', 'Helmet', 'Vest']
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 'nc' là key chuẩn kiểu YOLO annotation, nhưng vẫn có fallback
    # phòng trường hợp bạn đặt tên khác trong file thật
    if "nc" in data:
        num_classes = data["nc"]
    elif "num_classes" in data:
        num_classes = data["num_classes"]
    else:
        raise KeyError(
            f"Không tìm thấy key 'nc' hoặc 'num_classes' trong {yaml_path}. "
            f"Các key hiện có: {list(data.keys())}"
        )

    if "names" in data:
        class_names = data["names"]
    elif "classes" in data:
        class_names = data["classes"]
    else:
        raise KeyError(
            f"Không tìm thấy key 'names' hoặc 'classes' trong {yaml_path}. "
            f"Các key hiện có: {list(data.keys())}"
        )

    # Kiểm tra chéo: số lượng tên class phải khớp với nc
    assert len(class_names) == num_classes, (
        f"nc={num_classes} nhưng names có {len(class_names)} phần tử "
        f"({class_names}). Kiểm tra lại data.yaml."
    )

    return num_classes, class_names


def _images_dir_to_label_dir(images_dir):
    """
    Suy ra thu muc label tu thu muc anh, dung DUNG QUY UOC cua YOLO:
    thay THANH PHAN thu muc "images" (KHONG phai chuoi con bat ky) bang
    "labels" trong duong dan, vi du:

        .../train/images  ->  .../train/labels
        .../images/train   ->  .../labels/train

    Neu duong dan KHONG chua thanh phan "images" nao (cau truc khong
    theo quy uoc YOLO), raise loi ro rang thay vi doan sai.
    """
    parts = images_dir.replace("\\", "/").split("/")
    if "images" not in parts:
        raise ValueError(
            f"Khong suy ra duoc thu muc label tu '{images_dir}': duong dan "
            f"khong chua thanh phan thu muc ten 'images'. Hay dat anh trong "
            f"1 thu muc ten 'images' (quy uoc chuan YOLO), vi du "
            f".../train/images/, de tu dong suy ra .../train/labels/."
        )
    idx = len(parts) - 1 - parts[::-1].index("images")  # thay lan xuat hien CUOI CUNG
    parts[idx] = "labels"
    return os.path.normpath("/".join(parts))


def resolve_dataset_paths(yaml_path):
    """
    Doc data.yaml theo DUNG chuan cua YOLO (path/train/val + nc/names),
    tra ve san 4 duong dan anh/label cua tap train + valid -- de
    run_train.py chi can 1 tham so --data la du, giong het cach dung
    "--data data.yaml" cua YOLO.

    File data.yaml ky vong:
        path: ../datasets/ppe          # (tuy chon) thu muc goc dataset,
                                        # cac duong dan train/val se noi
                                        # TUONG DOI voi thu muc chua chinh
                                        # file data.yaml nay neu "path" la
                                        # duong dan tuong doi
        train: train/images            # thu muc ANH cua tap train
        val: valid/images              # thu muc ANH cua tap valid
        nc: 3
        names: ['Person', 'Helmet', 'Vest']

    Thu muc LABEL duoc TU DONG suy ra tu thu muc anh bang quy uoc YOLO
    (xem _images_dir_to_label_dir) -- KHONG can khai bao rieng trong
    data.yaml, dung giong YOLO that.

    Output: dict gom
        num_classes, class_names,
        train_img_dir, train_label_dir,
        val_img_dir, val_label_dir
    (tat ca duong dan da la ABSOLUTE, resolve san tu vi tri data.yaml)
    """
    yaml_path = os.path.abspath(yaml_path)
    yaml_dir = os.path.dirname(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    num_classes, class_names = load_dataset_config(yaml_path)

    if "train" not in data:
        raise KeyError(f"Khong tim thay key 'train' trong {yaml_path}.")
    if "val" not in data:
        raise KeyError(f"Khong tim thay key 'val' trong {yaml_path}.")

    # "path" (tuy chon, giong YOLO): thu muc goc de noi voi train/val
    # neu 2 cai do la duong dan TUONG DOI. Neu khong co "path", coi
    # train/val la tuong doi ngay voi thu muc chua data.yaml.
    base_dir = yaml_dir
    if "path" in data and data["path"]:
        base_dir = data["path"]
        if not os.path.isabs(base_dir):
            base_dir = os.path.normpath(os.path.join(yaml_dir, base_dir))

    def _resolve(rel_path):
        if os.path.isabs(rel_path):
            return os.path.normpath(rel_path)
        return os.path.normpath(os.path.join(base_dir, rel_path))

    train_img_dir = _resolve(data["train"])
    val_img_dir = _resolve(data["val"])

    train_label_dir = _images_dir_to_label_dir(train_img_dir)
    val_label_dir = _images_dir_to_label_dir(val_img_dir)

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "train_img_dir": train_img_dir,
        "train_label_dir": train_label_dir,
        "val_img_dir": val_img_dir,
        "val_label_dir": val_label_dir,
    }


if __name__ == "__main__":
    # ---- TEST bằng file data.yaml THẬT của dataset ----

    # File này nằm ở: Model/src/utils/config_utils.py
    # data.yaml nằm ở: Model/Data/data.yaml
    # -> từ utils/ đi lên 2 cấp (utils -> src -> Model) rồi vào Data/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(CURRENT_DIR, "..", "..", "Data", "data.yaml")
    yaml_path = os.path.normpath(yaml_path)

    print("Đang đọc file:", yaml_path)

    num_classes, class_names = load_dataset_config(yaml_path)
    print("num_classes:", num_classes)
    print("class_names:", class_names)

    print("\n--- TEST resolve_dataset_paths() ---")
    try:
        paths = resolve_dataset_paths(yaml_path)
        for k, v in paths.items():
            print(f"{k}: {v}")
    except (KeyError, ValueError) as e:
        print(f"(Bo qua: {e})")