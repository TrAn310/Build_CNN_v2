"""
visualization.py
Vẽ bounding box + label lên ảnh.
"""

import cv2
import numpy as np
import torch

COLORS = {
    0: (0, 255, 0),     # person - xanh lá
    1: (255, 100, 0),   # helmet - xanh dương
    2: (0, 165, 255),   # vest - cam
}
NAMES = {0: 'Person', 1: 'Helmet', 2: 'Vest'}


def draw_boxes(img_rgb, dets, class_names=None):
    """
    img_rgb: numpy HxWx3 RGB
    dets: [N,6] tensor [x1,y1,x2,y2,score,class]
    return: ảnh RGB đã vẽ
    """
    if isinstance(dets, torch.Tensor):
        dets = dets.detach().cpu().numpy()

    img = img_rgb.copy()
    if dets is None or len(dets) == 0:
        return img

    for d in dets:
        x1, y1, x2, y2, score, cls = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = (class_names[cls] if class_names else
                NAMES.get(cls, f'cls{cls}'))
        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img
