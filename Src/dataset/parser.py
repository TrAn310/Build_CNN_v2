"""
parser.py
Đọc annotation YOLO format: class xc yc w h (normalized 0..1)
"""

import os
import torch


def parse_yolo_label(label_path):
    """
    return: tensor [N, 5] = [class, xc, yc, w, h] normalized
    """
    if not os.path.exists(label_path):
        return torch.zeros(0, 5)
    rows = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            rows.append([cls, xc, yc, w, h])
    if len(rows) == 0:
        return torch.zeros(0, 5)
    return torch.tensor(rows, dtype=torch.float32)
