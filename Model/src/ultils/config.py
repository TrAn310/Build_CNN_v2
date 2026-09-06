"""
config_utils.py

Đọc data.yaml của dataset để lấy:
    - num_classes
    - class_names (danh sách tên class, đúng thứ tự index)

Việc này giúp num_classes KHÔNG bị hard-code trong code model, mà lấy động từ chính dataset đang dùng.
"""

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

if __name__ == "__main__":
    # ---- TEST bằng file data.yaml THẬT của dataset ----
    import os
 
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
 
 