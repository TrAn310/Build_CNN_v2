"""
Entry point đánh giá model trên test set.
"""

import argparse
import torch
from torch.utils.data import DataLoader

from src.models.detector import CustomPPEDetector
from src.dataset.dataset import PPEDetectionDataset
from src.dataset.collate import collate_fn
from src.evaluation.evaluator import evaluate, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_img', type=str, required=True)
    parser.add_argument('--test_lbl', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--class_names', type=str,
                        default='person,helmet,vest')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    class_names = args.class_names.split(',')

    model = CustomPPEDetector(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f'[Eval] Loaded weights: {args.weights}')

    test_ds = PPEDetectionDataset(args.test_img, args.test_lbl,
                                  img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    metrics = evaluate(model, test_loader, device, args.num_classes,
                       args.img_size, model.strides,
                       conf_thresh=args.conf, iou_thresh=args.iou,
                       class_names=class_names)
    print_metrics(metrics, class_names)


if __name__ == '__main__':
    main()
    