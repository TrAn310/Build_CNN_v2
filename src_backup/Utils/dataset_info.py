"""
dataset_info.py
Đọc data.yaml (Roboflow/YOLO format) để lấy số class và tên class.

Format data.yaml:
    nc: 10
    names: ['Hardhat', 'Mask', 'NO-Hardhat', ...]
"""

import os


def load_class_names(yaml_path):
    """
    Đọc file data.yaml, trả về (num_classes, list_of_names).
    Không dùng PyYAML để tránh cài thêm lib — parse thủ công.
    
    Trả về (None, None) nếu không đọc được.
    """
    if not os.path.exists(yaml_path):
        return None, None

    nc = None
    names = None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse 'nc: <số>'
    for line in content.splitlines():
        line = line.strip()
        if line.startswith('nc:'):
            try:
                nc = int(line.split(':', 1)[1].strip())
            except ValueError:
                pass
            break

    # Parse 'names: [...]' hoặc names dạng list nhiều dòng
    if 'names:' in content:
        # Trường hợp 1 dòng: names: ['a', 'b', 'c']
        idx = content.find('names:')
        rest = content[idx + len('names:'):].strip()

        # Bỏ phần mở đầu list
        if rest.startswith('['):
            end = rest.find(']')
            if end != -1:
                inner = rest[1:end]
                names = [s.strip().strip("'\"").strip() for s in inner.split(',')]

    return nc, names


def get_dataset_info(yaml_path=None, fallback_classes=3):
    """
    Public API:
        - Nếu tìm thấy data.yaml → trả về (nc, names) từ đó
        - Nếu không → trả về (fallback_classes, list generic)
    """
    if yaml_path and os.path.exists(yaml_path):
        nc, names = load_class_names(yaml_path)
        if nc is not None:
            if names is None or len(names) != nc:
                # Nếu names thiếu hoặc sai số lượng → tạo generic
                names = [f'class_{i}' for i in range(nc)]
            return nc, names

    # Fallback
    names = [f'class_{i}' for i in range(fallback_classes)]
    return fallback_classes, names
