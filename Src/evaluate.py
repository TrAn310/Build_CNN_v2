"""
Entry point đánh giá model trên test set.
Tách biệt nms_iou (cho NMS) và match_iou (cho tính TP/FP).
"""

import argparse
import torch
from torch.utils.data import DataLoader

from Src.models.detector import CustomPPEDetector
from Src.dataset.dataset import PPEDetectionDataset
from Src.dataset.collate import collate_fn
from Src.evaluation.evaluator import evaluate, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_img', type=str, required=True)
    parser.add_argument('--test_lbl', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--conf', type=float, default=0.3)
    # Tách 2 tham số IoU
    parser.add_argument('--nms_iou', type=float, default=0.5,
                        help='IoU threshold cho NMS')
    parser.add_argument('--match_iou', type=float, default=0.5,
                        help='IoU threshold để match TP/FP (mAP@match_iou)')
    # Giữ --iou làm alias (backward compatible)
    parser.add_argument('--iou', type=float, default=None,
                        help='(deprecated) alias cho --nms_iou')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--class_names', type=str,
                        default='person,helmet,vest')
    args = parser.parse_args()

    # Backward compat: nếu --iou được truyền, dùng cho cả 2
    nms_iou = args.nms_iou
    match_iou = args.match_iou
    if args.iou is not None:
        nms_iou = args.iou
        match_iou = args.iou
        print(f'[Eval] WARNING: --iou deprecated, dùng --nms_iou và --match_iou')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    class_names = args.class_names.split(',')

    model = CustomPPEDetector(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f'[Eval] Loaded weights: {args.weights}')
    print(f'[Eval] nms_iou={nms_iou}, match_iou={match_iou}, conf={args.conf}')

    test_ds = PPEDetectionDataset(args.test_img, args.test_lbl,
                                  img_size=args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    metrics = evaluate(model, test_loader, device, args.num_classes,
                       args.img_size, model.strides,
                       conf_thresh=args.conf,
                       nms_iou=nms_iou,
                       match_iou=match_iou,
                       class_names=class_names)
    print_metrics(metrics, class_names)


if __name__ == '__main__':
    main()