
"""
test_one_image.py
Test PPE detector tren mot anh, ve bounding box va luu ket qua.

Chay tu thu muc goc CNN_v2:
python test_one_image.py --image "test3.jpg" --weights "Src/outputs/checkpoints/best_model.pth"
"""

import argparse
import os
import sys

# Thu muc project
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "Src")

# Them cac thu muc de tuong thich voi import trong project
for folder in [
    "model",
    os.path.join("model", "utils"),
    "inference",
]:
    path = os.path.join(SRC, folder)
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2
import numpy as np
import torch

from detector import MiniPPEDetector
from postprocess import decode_and_filter


COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 0),
    (0, 128, 255),
]


def load_model(weights, device):
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Khong tim thay checkpoint: {weights}")

    # Checkpoint phai co metadata theo format cua project
    checkpoint = torch.load(weights, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint khong co model_state_dict. "
            "Can doi chieu format file checkpoint."
        )

    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 640)

    model = MiniPPEDetector(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Checkpoint : {weights}")
    print(f"So class   : {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Image size : {img_size}")
    print(f"Device     : {device}")

    return model, class_names, img_size


def preprocess(image_path, img_size):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    original_h, original_w = image.shape[:2]

    # Resize theo cung cach don gian cua inference trong project
    resized = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        np.transpose(rgb, (2, 0, 1)).copy()
    ).unsqueeze(0)

    return tensor, image, original_w, original_h


def draw_predictions(image, boxes, scores, classes, class_names):
    output = image.copy()
    h, w = output.shape[:2]

    for box, score, class_id in zip(
        boxes.tolist(), scores.tolist(), classes.tolist()
    ):
        x1, y1, x2, y2 = map(int, box)

        # Gioi han box trong khung hinh
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        color = COLORS[int(class_id) % len(COLORS)]
        name = (
            class_names[int(class_id)]
            if 0 <= int(class_id) < len(class_names)
            else f"class_{int(class_id)}"
        )
        label = f"{name} {float(score):.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )

        label_y = max(y1, th + 6)
        cv2.rectangle(
            output,
            (x1, label_y - th - 6),
            (x1 + tw + 4, label_y),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 2, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Test custom PPE detector tren mot anh"
    )
    parser.add_argument("--image", required=True, help="Duong dan anh can test")
    parser.add_argument("--weights", required=True, help="Checkpoint .pth")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "result_test.jpg"),
        help="Duong dan luu anh ket qua",
    )
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    weights_path = os.path.abspath(args.weights)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Ban chon CUDA nhung PyTorch khong nhan GPU.")

    model, class_names, img_size = load_model(weights_path, device)
    tensor, original, original_w, original_h = preprocess(
        image_path, img_size
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        predictions = model(tensor)

    # Su dung decoder/NMS cua chinh project
    results = decode_and_filter(
        predictions,
        conf_threshold=args.conf,
        nms_iou_threshold=args.iou,
    )

    boxes, scores, classes = results[0]

    # Box decoder tra ve theo toa do anh da resize
    if boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= original_w / img_size
        boxes[:, [1, 3]] *= original_h / img_size

    result_image = draw_predictions(
        original, boxes, scores, classes, class_names
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not cv2.imwrite(output_path, result_image):
        raise RuntimeError(f"Khong luu duoc anh ket qua: {output_path}")

    print(f"\nSo object phat hien: {len(boxes)}")
    for box, score, class_id in zip(
        boxes.tolist(), scores.tolist(), classes.tolist()
    ):
        name = class_names[int(class_id)]
        print(
            f"{name:20s} confidence={float(score):.3f} "
            f"box={[round(v, 1) for v in box]}"
        )

    print(f"Anh ket qua: {output_path}")

    # Mo cua so xem anh neu may co giao dien
    cv2.imshow("PPE Detection - Press Q to close", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
