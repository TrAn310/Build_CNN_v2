"""
overfit_test.py
Test bắt buộc theo prompt: train model overfit 1 ảnh duy nhất.
Nếu không overfit được 1 ảnh -> KHÔNG được train dataset lớn.
"""

import os
import argparse
import torch
import cv2
from torch.utils.data import DataLoader, Dataset

from Src.models.detector import CustomPPEDetector
from Src.dataset.parser import parse_yolo_label
from Src.dataset.collate import collate_fn
from Src.training.target_assigner import build_targets
from Src.training.losses import DetectionLoss
from Src.inference.postprocess import postprocess
from Src.Utils.Visualization import draw_boxes


class OneImageDataset(Dataset):
    def __init__(self, img_path, lbl_path, img_size):
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.img_size = img_size
        self.target = parse_yolo_label(lbl_path)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        img = cv2.imread(self.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img, self.target.clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--lbl', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default='outputs/figures/overfit')
    parser.add_argument('--log_every', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    ds = OneImageDataset(args.img, args.lbl, args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    print(f'[Overfit] GT: {ds.target.shape[0]} objects')
    print(f'[Overfit] device = {device}')

    model = CustomPPEDetector(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DetectionLoss(num_classes=args.num_classes)
    strides = model.strides
    feat_sizes = [(args.img_size // s, args.img_size // s) for s in strides]

    model.train()
    for epoch in range(1, args.epochs + 1):
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets]

            raw = model(imgs)
            tgt_list = build_targets(targets, args.num_classes,
                                     args.img_size, strides, feat_sizes)
            loss, bl, ol, cl = criterion(raw, tgt_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.log_every == 0 or epoch == 1:
            print(f'Epoch {epoch:4d} | loss={loss.item():.4f} '
                  f'box={bl.item():.4f} obj={ol.item():.4f} cls={cl.item():.4f}')

    # Visualization cuối
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            raw = model(imgs)
            dets = postprocess(raw, strides, args.num_classes,
                               args.img_size, conf_thresh=0.1,
                               iou_thresh=0.5)[0]
            break

    img = cv2.imread(args.img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (args.img_size, args.img_size))
    vis = draw_boxes(img_rgb, dets)
    out_path = os.path.join(args.out, 'overfit_result.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f'[Overfit] Saved visualization to {out_path}')
    print(f'[Overfit] Detections: {dets.shape[0]}')
    print('[Overfit] Nếu model không detect đúng -> debug trước khi train dataset lớn!')

    torch.save(model.state_dict(),
               os.path.join(args.out, 'overfit_model.pth'))


if __name__ == '__main__':
    main()
    