"""
visualization.py
Vẽ bounding box lên ảnh.
"""

import cv2
import numpy as np

COLORS = {
    0: (0, 255, 0),    # person - xanh lá
    1: (255, 0, 0),    # helmet - xanh dương
    2: (0, 165, 255),  # vest - cam
}
NAMES = {0: 'Person', 1: 'Helmet', 2: 'Vest'}


def draw_boxes(img, dets):
    """
    img: numpy HxWx3 (RGB hoặc BGR, sẽ convert)
    dets: [N,6] tensor [x1,y1,x2,y2, score, class]
    return: img với box vẽ lên
    """
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().numpy()
    img = img.copy()
    for d in dets:
        x1, y1, x2, y2, score, cls = d
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{NAMES.get(cls, str(cls))} {score:.2f}"
        cv2.putText(img, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


import torch  # noqa