"""
Entry point predict 1 ảnh.
"""

import argparse
import cv2
import torch

from src.models.detector import CustomPPEDetector
from src.inference.predict import predict_image
from src.utils.visualization import draw_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--out', type=str, default='outputs/predictions/result.jpg')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CustomPPEDetector(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img, dets = predict_image(model, args.image, device,
                              args.img_size, args.conf, args.iou,
                              args.num_classes)
    out = draw_boxes(img, dets)
    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f'Saved to {args.out}')
    print(f'Detections: {dets.shape[0]}')


if __name__ == '__main__':
    main()