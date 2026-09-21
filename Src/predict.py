"""
predict.py
Test model với 1 ảnh bất kỳ.
Cách dùng:
    python -m Src.predict --weights "Src\outputs\checkpoints\best_model.pth" --image "test.jpg" --output "result.jpg"
"""

import argparse
import cv2
import torch
from Src.models.detector import CustomPPEDetector
from Src.inference.postprocess import postprocess

CLASS_NAMES = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

# Màu cho từng class (BGR)
COLORS = [
    (255, 0, 0),      # Hardhat - xanh dương
    (0, 255, 0),      # Mask - xanh lá
    (0, 0, 255),      # NO-Hardhat - đỏ
    (255, 255, 0),    # NO-Mask - cyan
    (255, 0, 255),    # NO-Safety Vest - tím
    (0, 255, 255),    # Person - vàng
    (128, 0, 128),    # Safety Cone - tím đậm
    (255, 165, 0),    # Safety Vest - cam
    (0, 128, 0),      # machinery - xanh đậm
    (128, 128, 0),    # vehicle - olive
]


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize giữ tỉ lệ, thêm viền xám (YOLOv5 style)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw //= 2
    dh //= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    img_resized = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)
    return img_resized, r, (dw, dh)


def detect_image(model, img_path, conf=0.5, nms_iou=0.4, img_size=640, device='cuda'):
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        raise FileNotFoundError(f'Không đọc được ảnh: {img_path}')
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_orig.shape[:2]

    # Letterbox thay vì resize trực tiếp
    img_letterbox, ratio, (dw, dh) = letterbox(img_rgb, img_size)
    img_tensor = torch.from_numpy(img_letterbox).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        raw = model(img_tensor)
        dets = postprocess(raw, model.strides, len(CLASS_NAMES),
                           img_size, conf, nms_iou)

    dets = dets[0]

    # Scale box về ảnh gốc (bỏ padding)
    if dets.shape[0] > 0:
        dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / ratio
        dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / ratio
        dets[:, [0, 2]] = dets[:, [0, 2]].clamp(0, w_orig)
        dets[:, [1, 3]] = dets[:, [1, 3]].clamp(0, h_orig)

    img_draw = img_orig.copy()
    for d in dets:
        x1, y1, x2, y2, score, cls = d.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f'{CLASS_NAMES[cls]} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_draw, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_draw, dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='result.jpg')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (khuyến nghị 0.5-0.6)')
    parser.add_argument('--nms_iou', type=float, default=0.4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[Predict] Device: {device}')

    # Load model
    model = CustomPPEDetector(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f'[Predict] Loaded weights: {args.weights}')

    # Detect
    img_draw, dets = detect_image(model, args.image, args.conf, args.nms_iou, device=device)

    # Lưu ảnh
    cv2.imwrite(args.output, img_draw)
    print(f'[Predict] Đã lưu ảnh: {args.output}')
    print(f'[Predict] Số object phát hiện: {dets.shape[0]}')

    # In chi tiết
    for d in dets:
        x1, y1, x2, y2, score, cls = d.tolist()
        print(f'  {CLASS_NAMES[int(cls)]:15s} score={score:.3f} '
              f'box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')


if __name__ == '__main__':
    main()