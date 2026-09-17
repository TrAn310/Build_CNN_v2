# Custom PPE Detector — Toàn bộ mã nguồn dự án

## Cấu trúc thư mục đầy đủ

```
custom_ppe_detector/
│
├── data/detection/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py
│   │   ├── backbone.py
│   │   ├── neck.py
│   │   ├── head.py
│   │   └── detector.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── dataset.py
│   │   └── collate.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── target_assigner.py
│   │   ├── losses.py
│   │   └── train.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── decoder.py
│   │   ├── postprocess.py
│   │   ├── predict.py
│   │   └── video.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── association/
│   │   ├── __init__.py
│   │   └── ppe_association.py
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
│
├── train.py
├── predict.py
├── evaluate.py
├── video_inference.py
├── overfit_test.py
├── requirements.txt
└── README.md
```

## 1. requirements.txt

```txt
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
numpy>=1.24
tqdm>=4.66
matplotlib>=3.7
```

## 2. src/__init__.py

```python
# Package root
```

## 3. src/models/__init__.py

```python
from .blocks import ConvBlock, ResidualBlock
from .backbone import Backbone
from .neck import Neck
from .head import DetectionHead
from .detector import CustomPPEDetector
```

## 4. src/dataset/__init__.py

```python
from .parser import parse_yolo_label
from .dataset import PPEDetectionDataset
from .collate import collate_fn
```

## 5. src/training/__init__.py

```python
from .target_assigner import build_targets
from .losses import DetectionLoss
from .train import train
```

## 6. src/inference/__init__.py

```python
from .decoder import decode_predictions
from .postprocess import nms, postprocess, iou_batch
from .predict import predict_image, preprocess
from .video import run_video
```

## 7. src/evaluation/__init__.py

```python
from .evaluator import evaluate, print_metrics
```

## 8. src/association/__init__.py

```python
from .ppe_association import associate_ppe
```

## 9. src/utils/__init__.py

```python
from .visualization import draw_boxes
```

## 10. src/models/blocks.py

```python
"""
blocks.py
Building blocks cơ bản cho detector.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU"""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block đơn giản."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(c_in, c_out, k=3, s=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)
```

## 11. src/models/backbone.py

```python
"""
backbone.py
Custom CNN backbone, output P3/P4/P5.

Input:  [B, 3, 640, 640]
Output: P3 [B, c3, 80, 80]
        P4 [B, c4, 40, 40]
        P5 [B, c5, 20, 20]
"""

import torch.nn as nn
from .blocks import ConvBlock, ResidualBlock


class Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        c5 = int(512 * width_mult)

        # Stem: 640 -> 320
        self.stem = ConvBlock(3, c1, k=3, s=2)

        # Stage 1: 320 -> 160
        self.stage1 = nn.Sequential(
            ConvBlock(c1, c2, k=3, s=2),
            ResidualBlock(c2, c2),
        )

        # Stage 2: 160 -> 80 (P3)
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2),
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )

        # Stage 3: 80 -> 40 (P4)
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2),
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )

        # Stage 4: 40 -> 20 (P5)
        self.stage4 = nn.Sequential(
            ConvBlock(c4, c5, k=3, s=2),
            ResidualBlock(c5, c5),
            ResidualBlock(c5, c5),
        )

        self.out_channels = [c3, c4, c5]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
```

## 12. src/models/neck.py

```python
"""
neck.py
FPN-like top-down fusion.

P5 -> Conv -> Upsample x2 -> Concat P4 -> Conv -> F4
F4 -> Conv -> Upsample x2 -> Concat P3 -> Conv -> F3
F4 -> Conv stride 2 -> Concat P5 -> Conv -> F5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock


class Neck(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat5 = nn.Conv2d(c5, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.lat3 = nn.Conv2d(c3, out_channels, 1)

        self.smooth4 = ConvBlock(out_channels * 2, out_channels, k=3)
        self.smooth3 = ConvBlock(out_channels * 2, out_channels, k=3)

        self.down5 = ConvBlock(out_channels, out_channels, k=3, s=2)
        self.out_channels = out_channels

    def forward(self, p3, p4, p5):
        l5 = self.lat5(p5)
        l4 = self.lat4(p4)
        l3 = self.lat3(p3)

        up5 = F.interpolate(l5, scale_factor=2, mode='nearest')
        cat4 = torch.cat([up5, l4], dim=1)
        f4 = self.smooth4(cat4)

        up4 = F.interpolate(f4, scale_factor=2, mode='nearest')
        cat3 = torch.cat([up4, l3], dim=1)
        f3 = self.smooth3(cat3)

        f5 = self.down5(f4)
        return f3, f4, f5
```

## 13. src/models/head.py

```python
"""
head.py
Anchor-free detection head.

Output: [B, 5 + num_classes, H, W]
Trong đó: (tx, ty, tw, th, objectness, cls_0..cls_{C-1})
"""

import torch
import torch.nn as nn
from .blocks import ConvBlock


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=3, hidden=128):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = 5 + num_classes

        self.stem = nn.Sequential(
            ConvBlock(in_channels, hidden, k=3),
            ConvBlock(hidden, hidden, k=3),
        )
        self.pred_box = nn.Conv2d(hidden, 5, 1)
        self.pred_cls = nn.Conv2d(hidden, num_classes, 1)

        # Init bias objectness âm để tránh positive ban đầu
        nn.init.constant_(self.pred_box.bias[4], -4.0)

    def forward(self, x):
        feat = self.stem(x)
        box = self.pred_box(feat)
        cls = self.pred_cls(feat)
        return torch.cat([box, cls], dim=1)
```

## 14. src/models/detector.py

```python
"""
detector.py
Custom PPE Detector hoàn chỉnh.

Pipeline:
    Image [B,3,640,640]
        -> Backbone
        -> P3, P4, P5
        -> Neck
        -> F3, F4, F5
        -> 3 Detection Heads
        -> raw outputs
"""

import torch.nn as nn
from .backbone import Backbone
from .neck import Neck
from .head import DetectionHead


class CustomPPEDetector(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0, neck_channels=128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Backbone(width_mult=width_mult)
        self.neck = Neck(self.backbone.out_channels, out_channels=neck_channels)

        self.head_p3 = DetectionHead(neck_channels, num_classes)
        self.head_p4 = DetectionHead(neck_channels, num_classes)
        self.head_p5 = DetectionHead(neck_channels, num_classes)

        self.strides = [8, 16, 32]

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return [self.head_p3(f3), self.head_p4(f4), self.head_p5(f5)]
```

## 15. src/dataset/parser.py

```python
"""
parser.py
Đọc annotation YOLO format: class xc yc w h (normalized 0..1)
"""

import os
import torch


def parse_yolo_label(label_path):
    """
    return: tensor [N, 5] = [class, xc, yc, w, h] normalized
    """
    if not os.path.exists(label_path):
        return torch.zeros(0, 5)
    rows = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = float(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            rows.append([cls, xc, yc, w, h])
    if len(rows) == 0:
        return torch.zeros(0, 5)
    return torch.tensor(rows, dtype=torch.float32)
```

## 16. src/dataset/dataset.py

```python
"""
dataset.py
PPEDetectionDataset cho custom detector.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from .parser import parse_yolo_label


class PPEDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.files = sorted([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir,
                                  os.path.splitext(fname)[0] + '.txt')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        targets = parse_yolo_label(label_path)

        if self.augment and torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            if targets.shape[0] > 0:
                targets[:, 1] = 1.0 - targets[:, 1]

        return img, targets
```

## 17. src/dataset/collate.py

```python
"""
collate.py
Custom collate: mỗi ảnh có số object khác nhau nên không stack target.
"""

import torch


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
```

## 18. src/training/target_assigner.py

```python
"""
target_assigner.py
Gán GT box vào grid cell, encode thành (tx, ty, tw, th, obj, cls).

Chọn scale theo kích thước object: object nhỏ -> P3, lớn -> P5.
"""

import torch


def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    """
    targets: list tensor [N_i, 5] = [class, xc, yc, w, h] normalized
    return: list target tensors [B, 5+C, H, W]
    """
    device = targets[0].device if len(targets) > 0 else 'cpu'
    B = len(targets)

    all_targets = []
    for stride, (H, W) in zip(strides, feat_sizes):
        t = torch.zeros(B, 5 + num_classes, H, W, device=device)
        all_targets.append(t)

    for b in range(B):
        if targets[b] is None or targets[b].shape[0] == 0:
            continue
        gt = targets[b]
        for row in gt:
            cls = int(row[0].item())
            xc, yc, w, h = (row[1:] * img_size).tolist()

            # Chọn scale phù hợp
            obj_size = (w * h) ** 0.5
            best_s = 0
            best_diff = 1e9
            for si, st in enumerate(strides):
                diff = abs(st - obj_size)
                if diff < best_diff:
                    best_diff = diff
                    best_s = si

            stride = strides[best_s]
            H, W = feat_sizes[best_s]
            gx = max(0, min(W - 1, int(xc / stride)))
            gy = max(0, min(H - 1, int(yc / stride)))

            tx = xc / stride - gx
            ty = yc / stride - gy
            tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
            th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

            t = all_targets[best_s][b]
            t[0, gy, gx] = tx
            t[1, gy, gx] = ty
            t[2, gy, gx] = tw
            t[3, gy, gx] = th
            t[4, gy, gx] = 1.0
            t[5 + cls, gy, gx] = 1.0

    return all_targets
```

## 19. src/training/losses.py

```python
"""
losses.py
Detection loss = Box (Smooth L1) + Objectness (BCE) + Class (CE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3,
                 lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 pos_weight=1.0, neg_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, preds, targets):
        device = preds[0].device
        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        n_pos = 0

        for pred, tgt in zip(preds, targets):
            p_box = pred[:, :4]
            p_obj = pred[:, 4]
            p_cls = pred[:, 5:]

            t_box = tgt[:, :4]
            t_obj = tgt[:, 4]
            t_cls = tgt[:, 5:]

            pos_mask = t_obj > 0.5

            # Box loss (chỉ positive)
            if pos_mask.sum() > 0:
                pb = p_box.permute(0, 2, 3, 1)[pos_mask]
                tb = t_box.permute(0, 2, 3, 1)[pos_mask]
                total_box = total_box + F.smooth_l1_loss(pb, tb, reduction='sum')
                n_pos += pos_mask.sum().item()

            # Objectness loss (tất cả cells, weight imbalance)
            obj_loss_map = self.bce(p_obj, t_obj)
            w = torch.where(pos_mask,
                            torch.full_like(obj_loss_map, self.pos_weight),
                            torch.full_like(obj_loss_map, self.neg_weight))
            total_obj = total_obj + (obj_loss_map * w).sum()

            # Class loss (chỉ positive)
            if pos_mask.sum() > 0:
                pc = p_cls.permute(0, 2, 3, 1)[pos_mask]
                tc = t_cls.permute(0, 2, 3, 1)[pos_mask]
                tc_idx = tc.argmax(dim=1)
                total_cls = total_cls + self.ce(pc, tc_idx)

        n_pos = max(n_pos, 1)
        box_loss = total_box / n_pos
        obj_loss = total_obj / (n_pos * 100.0)
        cls_loss = total_cls / n_pos
        total = (self.lambda_box * box_loss +
                 self.lambda_obj * obj_loss +
                 self.lambda_cls * cls_loss)
        return total, box_loss, obj_loss, cls_loss
```

## 20. src/training/train.py

```python
"""
train.py
Training loop cho CustomPPEDetector.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.detector import CustomPPEDetector
from ..dataset.dataset import PPEDetectionDataset
from ..dataset.collate import collate_fn
from .target_assigner import build_targets
from .losses import DetectionLoss


def train_one_epoch(model, loader, optimizer, criterion, device,
                    num_classes, img_size, strides, feat_sizes):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc='Train')
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = [t.to(device) for t in targets]

        raw = model(imgs)
        tgt_list = build_targets(targets, num_classes, img_size,
                                 strides, feat_sizes)
        loss, bl, ol, cl = criterion(raw, tgt_list)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.3f}',
                         box=f'{bl.item():.3f}',
                         obj=f'{ol.item():.3f}',
                         cls=f'{cl.item():.3f}')
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes,
             img_size, strides, feat_sizes):
    model.eval()
    total = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = [t.to(device) for t in targets]
        raw = model(imgs)
        tgt_list = build_targets(targets, num_classes, img_size,
                                 strides, feat_sizes)
        loss, _, _, _ = criterion(raw, tgt_list)
        total += loss.item()
    return total / max(len(loader), 1)


def train(cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f'[Train] device = {device}')

    model = CustomPPEDetector(
        num_classes=cfg['num_classes'],
        width_mult=cfg.get('width_mult', 1.0),
    ).to(device)

    train_ds = PPEDetectionDataset(cfg['train_img'], cfg['train_lbl'],
                                   img_size=cfg['img_size'], augment=True)
    val_ds = PPEDetectionDataset(cfg['val_img'], cfg['val_lbl'],
                                 img_size=cfg['img_size'], augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['num_workers'], collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['num_workers'], collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'])
    criterion = DetectionLoss(num_classes=cfg['num_classes'])

    strides = model.strides
    feat_sizes = [(cfg['img_size'] // s, cfg['img_size'] // s) for s in strides]

    os.makedirs(cfg['save_dir'], exist_ok=True)
    best_val = float('inf')

    for epoch in range(cfg['epochs']):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, cfg['num_classes'], cfg['img_size'],
                                  strides, feat_sizes)
        val_loss = validate(model, val_loader, criterion, device,
                            cfg['num_classes'], cfg['img_size'],
                            strides, feat_sizes)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:03d}/{cfg["epochs"]} | '
              f'train={tr_loss:.4f} val={val_loss:.4f} lr={lr:.2e}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(cfg['save_dir'], 'best_model.pth'))
            print(f'  -> saved best (val={best_val:.4f})')

        torch.save(model.state_dict(),
                   os.path.join(cfg['save_dir'], 'last_model.pth'))

    print(f'[Done] best val loss = {best_val:.4f}')
    return model
```

## 21. src/inference/decoder.py

```python
"""
decoder.py
Decode raw prediction -> boxes pixel.

Công thức:
    cx = (sigmoid(tx) + grid_x) * stride
    cy = (sigmoid(ty) + grid_y) * stride
    w  = exp(tw) * stride
    h  = exp(th) * stride
    x1 = cx - w/2, ...
"""

import torch


def decode_predictions(raw_outputs, strides, num_classes,
                       img_size=640, conf_thresh=0.3):
    device = raw_outputs[0].device
    B = raw_outputs[0].shape[0]
    results = [[] for _ in range(B)]

    for out, stride in zip(raw_outputs, strides):
        B_, C_, H, W = out.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = xs.float().view(1, 1, H, W)
        grid_y = ys.float().view(1, 1, H, W)

        tx = torch.sigmoid(out[:, 0:1])
        ty = torch.sigmoid(out[:, 1:2])
        tw = out[:, 2:3]
        th = out[:, 3:4]
        obj = torch.sigmoid(out[:, 4:5])
        cls_prob = torch.softmax(out[:, 5:], dim=1)

        cx = (tx + grid_x) * stride
        cy = (ty + grid_y) * stride
        w = torch.exp(tw.clamp(-4, 4)) * stride
        h = torch.exp(th.clamp(-4, 4)) * stride

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        max_cls_prob, cls_idx = cls_prob.max(dim=1, keepdim=True)
        score = obj * max_cls_prob

        for b in range(B):
            s = score[b, 0]
            mask = s > conf_thresh
            if mask.sum() == 0:
                continue
            bx1 = x1[b, 0][mask]
            by1 = y1[b, 0][mask]
            bx2 = x2[b, 0][mask]
            by2 = y2[b, 0][mask]
            sc = s[mask]
            ci = cls_idx[b, 0][mask]
            det = torch.stack([bx1, by1, bx2, by2, sc, ci.float()], dim=1)
            results[b].append(det)

    final = []
    for b in range(B):
        if len(results[b]) == 0:
            final.append(torch.zeros(0, 6, device=device))
        else:
            final.append(torch.cat(results[b], dim=0))
    return final
```

## 22. src/inference/postprocess.py

```python
"""
postprocess.py
NMS tự code + pipeline decode -> NMS.
"""

import torch
from .decoder import decode_predictions


def iou_batch(boxes, box):
    """
    boxes: [N,4], box: [4]
    return: [N]
    """
    x1 = torch.max(boxes[:, 0], box[0])
    y1 = torch.max(boxes[:, 1], box[1])
    x2 = torch.min(boxes[:, 2], box[2])
    y2 = torch.min(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(dets, iou_thresh=0.5):
    """
    dets: [N,6] = [x1,y1,x2,y2, score, class]
    return: [M,6]
    """
    if dets.shape[0] == 0:
        return dets
    keep = []
    for c in dets[:, 5].unique():
        cls_mask = dets[:, 5] == c
        cls_dets = dets[cls_mask]
        order = cls_dets[:, 4].argsort(descending=True)
        cls_dets = cls_dets[order]
        while cls_dets.shape[0] > 0:
            best = cls_dets[0]
            keep.append(best)
            if cls_dets.shape[0] == 1:
                break
            ious = iou_batch(cls_dets[1:, :4], best[:4])
            cls_dets = cls_dets[1:][ious < iou_thresh]
    if len(keep) == 0:
        return torch.zeros(0, 6, device=dets.device)
    return torch.stack(keep, dim=0)


def postprocess(raw_outputs, strides, num_classes, img_size=640,
                conf_thresh=0.3, iou_thresh=0.5):
    decoded = decode_predictions(raw_outputs, strides, num_classes,
                                 img_size, conf_thresh)
    return [nms(d, iou_thresh) for d in decoded]
```

## 23. src/inference/predict.py

```python
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
```

## 24. src/inference/video.py

```python
"""
video.py
Video/webcam inference + FPS + logging vi phạm.
"""

import os
import csv
import time
import cv2
import torch

from .postprocess import postprocess
from ..utils.visualization import draw_boxes
from ..association.ppe_association import associate_ppe


@torch.no_grad()
def run_video(model, source, output_path=None, device='cpu',
              img_size=640, conf_thresh=0.3, iou_thresh=0.5,
              num_classes=3, save_csv=None, save_snapshots=None,
              show=True, max_frames=None):
    """
    source: đường dẫn video (str) hoặc 0 (int) cho webcam.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được source: {source}')

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[Video] {w}x{h} @ {fps_src:.1f} FPS')

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    csv_file = None
    csv_writer = None
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or '.', exist_ok=True)
        csv_file = open(save_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'person_id', 'status'])

    if save_snapshots:
        os.makedirs(save_snapshots, exist_ok=True)

    frame_idx = 0
    fps_list = []
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t0 = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = cv2.resize(img_rgb, (img_size, img_size))
        tensor = tensor.astype('float32') / 255.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

        raw = model(tensor)
        dets = postprocess(raw, model.strides, num_classes,
                           img_size, conf_thresh, iou_thresh)[0]

        # Rescale về ảnh gốc
        if dets.shape[0] > 0:
            dets[:, [0, 2]] *= w / img_size
            dets[:, [1, 3]] *= h / img_size

        # Association
        statuses = associate_ppe(dets)

        # Log vi phạm
        for s in statuses:
            if s['status'] != 'SAFE':
                if csv_writer:
                    csv_writer.writerow([frame_idx, s['id'], s['status']])
                if save_snapshots:
                    snap = os.path.join(
                        save_snapshots,
                        f'f{frame_idx:06d}_id{s["id"]}_{s["status"]}.jpg')
                    cv2.imwrite(snap, frame)

        # Vẽ
        vis = draw_boxes(img_rgb, dets)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        for s in statuses:
            x1, y1 = int(s['box'][0]), int(s['box'][1])
            color = (0, 255, 0) if s['status'] == 'SAFE' else (0, 0, 255)
            cv2.putText(vis, s['status'], (x1, max(20, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        dt = time.time() - t0
        fps_list.append(1.0 / max(dt, 1e-6))
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(vis, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer:
            writer.write(vis)
        if show:
            cv2.imshow('PPE Detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()

    avg = sum(fps_list) / max(len(fps_list), 1)
    print(f'[Video] Done. Frames={frame_idx} AvgFPS={avg:.2f}')
    return avg
```

## 25. src/association/ppe_association.py

```python
"""
ppe_association.py
Ghép helmet/vest với person dựa trên ROI.
"""

CLASS_PERSON = 0
CLASS_HELMET = 1
CLASS_VEST = 2


def make_head_roi(box):
    """Head ROI = 0-30% chiều cao box person."""
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + 0.30 * (y2 - y1))


def make_torso_roi(box):
    """Torso ROI = 20-70% chiều cao box person."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + 0.20 * h, x2, y1 + 0.70 * h)


def center_inside(inner_box, roi):
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def associate_ppe(dets):
    """
    dets: [N,6] = [x1,y1,x2,y2,score,class]
    return: list dict {id, box, status, has_helmet, has_vest}
    """
    if dets.shape[0] == 0:
        return []
    dets_np = dets.cpu().numpy() if hasattr(dets, 'cpu') else dets

    persons, helmets, vests = [], [], []
    for d in dets_np:
        x1, y1, x2, y2, score, cls = d
        box = [float(x1), float(y1), float(x2), float(y2)]
        cls = int(cls)
        if cls == CLASS_PERSON:
            persons.append(box)
        elif cls == CLASS_HELMET:
            helmets.append(box)
        elif cls == CLASS_VEST:
            vests.append(box)

    results = []
    for pid, p in enumerate(persons):
        head_roi = make_head_roi(p)
        torso_roi = make_torso_roi(p)
        has_helmet = any(center_inside(h, head_roi) for h in helmets)
        has_vest = any(center_inside(v, torso_roi) for v in vests)

        if not has_helmet and not has_vest:
            status = 'NO_HELMET_NO_VEST'
        elif not has_helmet:
            status = 'NO_HELMET'
        elif not has_vest:
            status = 'NO_VEST'
        else:
            status = 'SAFE'

        results.append({
            'id': pid, 'box': p, 'status': status,
            'has_helmet': has_helmet, 'has_vest': has_vest,
        })
    return results
```

## 26. src/evaluation/evaluator.py

```python
"""
evaluator.py
Precision/Recall/F1/AP/mAP tự code.
"""

import numpy as np
import torch
from ..inference.postprocess import iou_batch, postprocess


def _gt_to_xyxy(gts):
    """gts: [M,5] = [cls, xc, yc, w, h] pixel -> [M,5] = [cls, x1, y1, x2, y2]"""
    if gts.shape[0] == 0:
        return gts
    x1 = gts[:, 1] - gts[:, 3] / 2
    y1 = gts[:, 2] - gts[:, 4] / 2
    x2 = gts[:, 1] + gts[:, 3] / 2
    y2 = gts[:, 2] + gts[:, 4] / 2
    return torch.stack([gts[:, 0], x1, y1, x2, y2], dim=1)


def compute_precision_recall(tp, fp, fn):
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f1 = 2 * p * r / (p + r + 1e-6)
    return p, r, f1


def compute_ap_11point(precisions, recalls):
    """11-point interpolation AP."""
    if len(precisions) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_max = 0.0
        for p, r in zip(precisions, recalls):
            if r >= t and p > p_max:
                p_max = p
        ap += p_max / 11.0
    return ap


def compute_ap_from_curve(scores, tps, num_gt):
    """
    scores: list float (score của từng prediction, đã sort)
    tps:    list int (1 nếu TP, 0 nếu FP)
    num_gt: tổng số GT của class
    """
    if num_gt == 0 or len(scores) == 0:
        return 0.0
    scores = np.array(scores)
    tps = np.array(tps)
    order = np.argsort(-scores)
    tp_cum = np.cumsum(tps[order])
    fp_cum = np.cumsum(1 - tps[order])
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / (num_gt + 1e-6)
    return compute_ap_11point(precisions.tolist(), recalls.tolist())


@torch.no_grad()
def evaluate(model, loader, device, num_classes, img_size,
             strides, conf_thresh=0.3, iou_thresh=0.5,
             class_names=None):
    """
    Trả về dict metrics tổng + per-class.
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    model.eval()
    per_class_preds = {c: [] for c in range(num_classes)}
    per_class_num_gt = {c: 0 for c in range(num_classes)}

    for imgs, targets in loader:
        imgs = imgs.to(device)
        raw = model(imgs)
        dets = postprocess(raw, strides, num_classes, img_size,
                           conf_thresh, iou_thresh)

        for b in range(imgs.shape[0]):
            gt = targets[b].clone().to(device)
            if gt.shape[0] > 0:
                gt[:, 1:] = gt[:, 1:] * img_size
                gt_xyxy = _gt_to_xyxy(gt)
            else:
                gt_xyxy = torch.zeros(0, 5, device=device)

            preds = dets[b]

            for c in range(num_classes):
                p_c = preds[preds[:, 5] == c] if preds.shape[0] > 0 else preds
                g_c = gt_xyxy[gt_xyxy[:, 0] == c] if gt_xyxy.shape[0] > 0 else gt_xyxy

                per_class_num_gt[c] += g_c.shape[0]

                # Sort predictions theo score giảm dần
                if p_c.shape[0] > 0:
                    order = p_c[:, 4].argsort(descending=True)
                    p_c = p_c[order]

                matched_gt = set()
                for p in p_c:
                    best_iou = 0.0
                    best_gi = -1
                    for gi in range(g_c.shape[0]):
                        if gi in matched_gt:
                            continue
                        iou = iou_batch(g_c[gi:gi+1, 1:], p[:4])[0].item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    is_tp = 1 if (best_iou >= iou_thresh and best_gi >= 0) else 0
                    if is_tp:
                        matched_gt.add(best_gi)
                    per_class_preds[c].append((p[4].item(), is_tp))

    metrics = {}
    ap_list = []
    for c in range(num_classes):
        num_gt = per_class_num_gt[c]
        preds_c = per_class_preds[c]
        if len(preds_c) > 0:
            scores = [x[0] for x in preds_c]
            tps = [x[1] for x in preds_c]
        else:
            scores, tps = [], []

        ap = compute_ap_from_curve(scores, tps, num_gt)
        tp_total = sum(tps)
        fp_total = len(tps) - tp_total
        fn_total = num_gt - tp_total
        p, r, f1 = compute_precision_recall(tp_total, fp_total, fn_total)

        metrics[class_names[c]] = {
            'precision': p, 'recall': r, 'f1': f1, 'ap': ap,
            'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
            'num_gt': num_gt,
        }
        ap_list.append(ap)

    metrics['mAP'] = sum(ap_list) / max(len(ap_list), 1)
    return metrics


def print_metrics(metrics, class_names=None):
    print('\n' + '=' * 70)
    print(f'{"Class":<12}{"P":>8}{"R":>8}{"F1":>8}{"AP":>8}'
          f'{"TP":>6}{"FP":>6}{"FN":>6}{"GT":>6}')
    print('-' * 70)
    for k, v in metrics.items():
        if k == 'mAP':
            continue
        print(f'{k:<12}{v["precision"]:>8.3f}{v["recall"]:>8.3f}'
              f'{v["f1"]:>8.3f}{v["ap"]:>8.3f}'
              f'{v["tp"]:>6d}{v["fp"]:>6d}{v["fn"]:>6d}{v["num_gt"]:>6d}')
    print('-' * 70)
    print(f'{"mAP":<12}{"":>8}{"":>8}{"":>8}{metrics["mAP"]:>8.3f}')
    print('=' * 70 + '\n')
```

## 27. src/utils/visualization.py

```python
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
```

## 28. train.py (ROOT)

```python
"""
Entry point train.
"""

import argparse
from src.training.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', type=str, required=True)
    parser.add_argument('--train_lbl', type=str, required=True)
    parser.add_argument('--val_img', type=str, required=True)
    parser.add_argument('--val_lbl', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--width_mult', type=float, default=1.0)
    args = parser.parse_args()
    train(vars(args))


if __name__ == '__main__':
    main()
```

## 29. predict.py (ROOT)

```python
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
```

## 30. evaluate.py (ROOT)

```python
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
```

## 31. video_inference.py (ROOT)

```python
"""
Entry point video inference.
"""

import argparse
import torch

from src.models.detector import CustomPPEDetector
from src.inference.video import run_video


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
```

## 32. overfit_test.py (ROOT)

```python
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

from src.models.detector import CustomPPEDetector
from src.dataset.parser import parse_yolo_label
from src.dataset.collate import collate_fn
from src.training.target_assigner import build_targets
from src.training.losses import DetectionLoss
from src.inference.postprocess import postprocess
from src.utils.visualization import draw_boxes


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
```

## 33. README.md

```markdown
# Custom PPE Detector

Object detector tự xây bằng PyTorch để phát hiện PPE (Person / Helmet / Vest) trên công trường.
**Không dùng YOLO, Ultralytics, Detectron2, MMDetection hay bất kỳ thư viện detection nào.**

## Kiến trúc

Input [B, 3, 640, 640]
  ↓
Backbone (custom CNN)
  ↓
P3 [80x80], P4 [40x40], P5 [20x20]
  ↓
Neck (FPN-like top-down fusion)
  ↓
F3, F4, F5
  ↓
3 Detection Heads (anchor-free, dense prediction)
  ↓
Raw predictions [B, 5+C, H, W]
  ↓
Decode (tx,ty,tw,th → x1,y1,x2,y2)
  ↓
Confidence threshold
  ↓
NMS tự code
  ↓
Final detections [N, 6] = [x1,y1,x2,y2,score,class]

## Cài đặt

    pip install -r requirements.txt

## Cấu trúc dataset (YOLO format)

    data/detection/
    ├── images/
    │   ├── train/*.jpg
    │   ├── val/*.jpg
    │   └── test/*.jpg
    └── labels/
        ├── train/*.txt
        ├── val/*.txt
        └── test/*.txt

Mỗi file `.txt` chứa các dòng:

    class x_center y_center width height

với tọa độ normalized [0, 1]. `class`: 0=person, 1=helmet, 2=vest.

## Training

**Bước 1: Test overfit 1 ảnh (BẮT BUỘC)**

    python overfit_test.py --img path/to/img.jpg --lbl path/to/img.txt --epochs 300

Nếu model không overfit được 1 ảnh → debug trước, KHÔNG train dataset lớn.

**Bước 2: Train full dataset**

    python train.py \
        --train_img data/detection/images/train \
        --train_lbl data/detection/labels/train \
        --val_img data/detection/images/val \
        --val_lbl data/detection/labels/val \
        --epochs 50 --batch_size 8 --lr 1e-3 --img_size 640

Checkpoint lưu tại `outputs/checkpoints/best_model.pth`.

## Inference

**1 ảnh:**

    python predict.py \
        --weights outputs/checkpoints/best_model.pth \
        --image test.jpg \
        --out outputs/predictions/result.jpg

**Video:**

    python video_inference.py \
        --weights outputs/checkpoints/best_model.pth \
        --source path/to/video.mp4 \
        --output outputs/videos/result.mp4 \
        --save_csv outputs/logs/violations.csv

**Webcam:**

    python video_inference.py --weights outputs/checkpoints/best_model.pth --source 0

## Evaluation

    python evaluate.py \
        --weights outputs/checkpoints/best_model.pth \
        --test_img data/detection/images/test \
        --test_lbl data/detection/labels/test

Output: precision / recall / F1 / AP / mAP từng class.

## PPE Association

Sau khi detector trả về person/helmet/vest, module `ppe_association.py` ghép PPE vào từng người:
- Head ROI: 0-30% chiều cao box person → check helmet
- Torso ROI: 20-70% chiều cao box person → check vest
- Output: `SAFE / NO_HELMET / NO_VEST / NO_HELMET_NO_VEST`

## Các bước debug bắt buộc

1. **Forward random input**

       import torch
       from src.models.detector import CustomPPEDetector
       m = CustomPPEDetector()
       x = torch.randn(2, 3, 640, 640)
       outs = m(x)
       for o in outs: print(o.shape)  # [2,8,80,80], [2,8,40,40], [2,8,20,20]

2. **Overfit 1 ảnh** — `python overfit_test.py --img ... --lbl ...`

3. **Train 10-20 ảnh** — dùng `train.py` với subset nhỏ

4. **Train full dataset**

## Failure analysis (cần làm trong báo cáo)

Lưu các case lỗi:
- False Positive / False Negative
- Missed Helmet (vi phạm an toàn → quan trọng)
- Wrong Class / Wrong Box / Duplicate Box
- Small Object / Occlusion

## Giới hạn

- Backbone nhỏ, không mạnh bằng YOLO.
- Head anchor-free đơn giản, assignment dựa vào kích thước object.
- Chưa có tracking → identity giữa các frame không ổn định.
- Chưa có temporal smoothing → cảnh báo có thể nhấp nháy.

## Hướng phát triển

- Thêm tracking (ByteTrack đơn giản) để giữ ID person.
- Temporal smoothing (N-of-M frames).
- Domain adaptation khi chuyển camera.
- Export ONNX/TensorRT để tăng FPS.
```

## Checklist chạy

Sau khi copy hết, chạy theo thứ tự:

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Test forward
python -c "
import torch
from src.models.detector import CustomPPEDetector
m = CustomPPEDetector()
x = torch.randn(2, 3, 640, 640)
outs = m(x)
for o in outs: print(o.shape)
"
# Kỳ vọng: torch.Size([2, 8, 80, 80])
#          torch.Size([2, 8, 40, 40])
#          torch.Size([2, 8, 20, 20])

# 3. Overfit 1 ảnh
python overfit_test.py --img data/detection/images/train/img001.jpg \
    --lbl data/detection/labels/train/img001.txt --epochs 300

# 4. Train full
python train.py --train_img data/detection/images/train \
    --train_lbl data/detection/labels/train \
    --val_img data/detection/images/val \
    --val_lbl data/detection/labels/val \
    --epochs 50 --batch_size 8 --lr 1e-3
```
