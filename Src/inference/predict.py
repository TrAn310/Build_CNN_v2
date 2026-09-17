"""
predict.py
Inference trên 1 ảnh.
"""

import cv2
import torch


def preprocess(img_rgb, img_size=640):
    img = cv2.resize(img_rgb, (img_size, img_size))
    img = img.astype('float32') / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def predict_image(model, img_path, device='cpu', img_size=640,
                  conf_thresh=0.3, iou_thresh=0.5, num_classes=3):
    """
    return: (img_rgb, dets) với dets [N,6] toạ độ pixel trên ảnh gốc.
    """
    from .postprocess import postprocess
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    tensor = preprocess(img_rgb, img_size).to(device)
    raw = model(tensor)
    dets = postprocess(raw, model.strides, num_classes, img_size,
                       conf_thresh, iou_thresh)[0]

    # Rescale boxes về kích thước ảnh gốc
    if dets.shape[0] > 0:
        sx = orig_w / img_size
        sy = orig_h / img_size
        dets[:, 0] *= sx
        dets[:, 2] *= sx
        dets[:, 1] *= sy
        dets[:, 3] *= sy

    return img_rgb, dets