"""
Entry point video inference.
"""

import argparse
import torch

from Src.models.detector import CustomPPEDetector
from Src.inference.video import run_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True,
                        help='video path or 0 for webcam')
    parser.add_argument('--output', type=str, default='outputs/videos/result.mp4')
    parser.add_argument('--save_csv', type=str, default='outputs/logs/violations.csv')
    parser.add_argument('--save_snapshots', type=str, default='outputs/snapshots')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_show', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = CustomPPEDetector(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    source = int(args.source) if args.source.isdigit() else args.source

    run_video(
        model, source,
        output_path=args.output,
        device=device,
        img_size=args.img_size,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        num_classes=args.num_classes,
        save_csv=args.save_csv,
        save_snapshots=args.save_snapshots,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
    