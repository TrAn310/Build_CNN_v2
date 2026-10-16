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

# Custom PPE Detector

## 2. Cấu Trúc Thư Mục

```text
custom_ppe_detector/
│
├── data/
│   └── detection/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
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
│       ├── visualization.py
│       └── metrics.py
│
├── train.py
├── predict.py
├── evaluate.py
├── video_inference.py
├── overfit_test.py
├── requirements.txt
└── README.md
```

---

## 3. Kiến Trúc Model

### 3.1 Tổng Quan

```text
Input 640×640×3

        ↓

CUSTOM BACKBONE
Block 1
Block 2
Block 3
Block 4

        ↓

Feature maps:
P3 = 128 × 80 × 80
P4 = 256 × 40 × 40
P5 = 512 × 20 × 20

        ↓

CUSTOM NECK
Feature fusion
Upsample
Concatenate
Conv

        ↓

CUSTOM DETECTION HEAD (3 heads)

        ↓

Raw predictions
P3: [B, 5+C, 80, 80]
P4: [B, 5+C, 40, 40]
P5: [B, 5+C, 20, 20]

        ↓

Decode

        ↓

Confidence threshold

        ↓

NMS

        ↓

Final detections
```

### 3.2 Backbone Chi Tiết

```text
Input
3 × 640 × 640

↓ Conv stride 2

32 × 320 × 320

↓ Conv stride 2

64 × 160 × 160

↓ Conv stride 2

128 × 80 × 80   → P3

↓ Conv stride 2

256 × 40 × 40   → P4

↓ Conv stride 2

512 × 20 × 20   → P5
```

**Nguyên lý multi-scale:**

- Càng đi sâu: resolution ↓, semantic information ↑, receptive field ↑
- P3 phù hợp object nhỏ (helmet)
- P4 phù hợp object trung bình
- P5 phù hợp object lớn (person)

### 3.3 Neck Chi Tiết (FPN-like Top-down)

```text
P5 [B,512,20,20]
 ↓ Conv 1×1
[B,256,20,20]
 ↓ Upsample ×2
[B,256,40,40]
Concat P4 [B,256,40,40]
 ↓
[B,512,40,40]
 ↓ Conv 3×3
F4 = [B,256,40,40]
 ↓ Upsample ×2
[B,256,80,80]
Concat P3 [B,128,80,80] → sau Conv 1×1 là [B,256,80,80]
 ↓
[B,512,80,80]
 ↓ Conv 3×3
F3 = [B,256,80,80]

F4 → Conv stride 2 → F5 = [B,256,20,20]
```

### 3.4 Detection Head

Output mỗi head: `[B, 5 + num_classes, H, W]`

Với `num_classes = 3` → `[B, 8, H, W]`

Trong đó 5 giá trị đầu:

- `tx`, `ty`: offset tâm box so với grid cell
- `tw`, `th`: log kích thước box
- `objectness`: có object hay không

`num_classes` giá trị sau: class scores.

### 3.5 Box Encoding / Decoding

**Encode (khi tạo target):**

```text
tx = xc / stride - grid_x
ty = yc / stride - grid_y
tw = log(w / stride)
th = log(h / stride)
```

**Decode (khi inference):**

```text
cx = (sigmoid(tx) + grid_x) * stride
cy = (sigmoid(ty) + grid_y) * stride
w  = exp(tw) * stride
h  = exp(th) * stride

x1 = cx - w/2
y1 = cy - h/2
x2 = cx + w/2
y2 = cy + h/2
```

**Giải thích:**

- `sigmoid(tx)`: đưa offset về [0,1], đảm bảo tâm box nằm trong cell
- `grid_x`, `grid_y`: tọa độ cell trong feature map
- `stride`: hệ số scale từ feature map về ảnh gốc
- `exp(tw)`: đảm bảo width > 0

---

## 4. Pipeline Chi Tiết

### 4.1 Training Pipeline

```text
Image [B,3,640,640]
      ↓
Model forward
      ↓
Raw outputs [P3, P4, P5]
      ↓
build_targets() → target tensors
      ↓
DetectionLoss()
  ├── Box loss (CIoU)
  ├── Objectness loss (BCE weighted)
  └── Class loss (Focal)
      ↓
loss.backward()
      ↓
optimizer.step()
```

### 4.2 Inference Pipeline

```text
Image
      ↓
Resize / Letterbox
      ↓
Normalize /255
      ↓
Tensor [1,3,640,640]
      ↓
Model forward
      ↓
Raw outputs
      ↓
decode_predictions()
      ↓
postprocess()
  ├── NMS
  └── Max detections
      ↓
Final detections [N,6] = [x1,y1,x2,y2,score,class]
```

### 4.3 Evaluation Pipeline

```text
Model
      ↓
decode
      ↓
confidence threshold
      ↓
NMS
      ↓
compare prediction vs ground truth
      ↓
IoU matching (match_iou)
      ↓
TP / FP / FN
      ↓
precision / recall
      ↓
AP / mAP
```

---

 Các Module Chính

 `blocks.py`

Building blocks cơ bản:

```python
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → SiLU"""
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

### 5.2 `backbone.py`

```python
class Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(256 * width_mult)
        c5 = int(512 * width_mult)

        self.stem = ConvBlock(3, c1, k=3, s=2)
        self.stage1 = nn.Sequential(
            ConvBlock(c1, c2, k=3, s=2),
            ResidualBlock(c2, c2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2),
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2),
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )
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

### 5.3 `neck.py`

```python
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

### 5.4 `head.py`

```python
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

### 5.5 `detector.py`

```python
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

### 5.6 `target_assigner.py`

```python
def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    """
    Gán object vào scale phù hợp theo kích thước (YOLOv5 style).
    """
    device = targets[0].device if len(targets) > 0 else 'cpu'
    B = len(targets)

    # Ngưỡng kích thước cho từng scale (pixel)
    scale_ranges = [(0, 96), (48, 160), (96, 99999)]

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
            obj_size = max(w, h)

            for s_idx, stride in enumerate(strides):
                lo, hi = scale_ranges[s_idx]
                if not (lo <= obj_size < hi):
                    continue

                H, W = feat_sizes[s_idx]
                gx = max(0, min(W - 1, int(xc / stride)))
                gy = max(0, min(H - 1, int(yc / stride)))

                tx = xc / stride - gx
                ty = yc / stride - gy
                tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
                th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

                t = all_targets[s_idx][b]
                if t[4, gy, gx] < 0.5:
                    t[0, gy, gx] = tx
                    t[1, gy, gx] = ty
                    t[2, gy, gx] = tw
                    t[3, gy, gx] = th
                    t[4, gy, gx] = 0.9
                    t[5 + cls, gy, gx] = 1.0

    return all_targets
```

### 5.7 `losses.py`

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             weight=self.class_weights)
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.sum()


def ciou_loss_from_offsets(pred_off, target_off, eps=1e-7):
    """CIoU loss trên offset (tx, ty, tw, th) — cùng cell."""
    px1 = pred_off[:, 0] - torch.exp(pred_off[:, 2]) / 2
    py1 = pred_off[:, 1] - torch.exp(pred_off[:, 3]) / 2
    px2 = pred_off[:, 0] + torch.exp(pred_off[:, 2]) / 2
    py2 = pred_off[:, 1] + torch.exp(pred_off[:, 3]) / 2

    tx1 = target_off[:, 0] - torch.exp(target_off[:, 2]) / 2
    ty1 = target_off[:, 1] - torch.exp(target_off[:, 3]) / 2
    tx2 = target_off[:, 0] + torch.exp(target_off[:, 2]) / 2
    ty2 = target_off[:, 1] + torch.exp(target_off[:, 3]) / 2

    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    c_area = (cx2 - cx1) * (cy2 - cy1) + eps

    rho2 = (pred_off[:, 0] - target_off[:, 0]) ** 2 + \
           (pred_off[:, 1] - target_off[:, 1]) ** 2

    w_p = torch.exp(pred_off[:, 2]); h_p = torch.exp(pred_off[:, 3])
    w_t = torch.exp(target_off[:, 2]); h_t = torch.exp(target_off[:, 3])
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w_t / (h_t + eps)) - torch.atan(w_p / (h_p + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c_area - alpha * v
    return (1 - ciou).mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=10,
                 lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 pos_weight=10.0, neg_weight=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = FocalLoss(alpha=0.75, gamma=2.0)

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

            # ---- Box loss (CIoU) ----
            if pos_mask.sum() > 0:
                pb = p_box.permute(0, 2, 3, 1)[pos_mask]
                tb = t_box.permute(0, 2, 3, 1)[pos_mask]
                total_box = total_box + ciou_loss_from_offsets(pb, tb) * pos_mask.sum()
                n_pos += pos_mask.sum().item()

            # ---- Objectness loss ----
            obj_loss_map = self.bce(p_obj, t_obj)
            w = torch.where(pos_mask,
                            torch.full_like(obj_loss_map, self.pos_weight),
                            torch.full_like(obj_loss_map, self.neg_weight))
            total_obj = total_obj + (obj_loss_map * w).mean()

            # ---- Class loss (Focal) ----
            if pos_mask.sum() > 0:
                pc = p_cls.permute(0, 2, 3, 1)[pos_mask]
                tc = t_cls.permute(0, 2, 3, 1)[pos_mask]
                tc_idx = tc.argmax(dim=1)
                total_cls = total_cls + self.ce(pc, tc_idx)

        n_pos = max(n_pos, 1)

        box_loss = total_box / n_pos
        obj_loss = total_obj
        cls_loss = total_cls / n_pos

        total = (self.lambda_box * box_loss +
                 self.lambda_obj * obj_loss +
                 self.lambda_cls * cls_loss)
        return total, box_loss, obj_loss, cls_loss
```

### 5.8 `decoder.py`

```python
def decode_predictions(raw_outputs, strides, num_classes,
                       img_size=640, conf_thresh=0.3,
                       obj_thresh=0.5):
    """
    raw_outputs: list [B, 5+C, H, W] cho mỗi scale
    strides: [8, 16, 32]
    return: list [B] mỗi phần tử là [N, 6] = [x1, y1, x2, y2, score, class]
    """
    device = raw_outputs[0].device
    B = raw_outputs[0].shape[0]

    all_dets = [[] for _ in range(B)]

    for s_idx, (pred, stride) in enumerate(zip(raw_outputs, strides)):
        _, _, H, W = pred.shape

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        p_box = pred[:, :4]
        p_obj = pred[:, 4]
        p_cls = pred[:, 5:]

        xc = (grid_x.unsqueeze(0) + p_box[:, 0]) * stride
        yc = (grid_y.unsqueeze(0) + p_box[:, 1]) * stride
        w = torch.exp(p_box[:, 2].clamp(-10, 10)) * stride
        h = torch.exp(p_box[:, 3].clamp(-10, 10)) * stride

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        obj_score = torch.sigmoid(p_obj)
        cls_score = torch.softmax(p_cls, dim=1)
        cls_max, cls_idx = cls_score.max(dim=1)
        score = obj_score * cls_max

        for b in range(B):
            mask = (score[b] > conf_thresh) & (obj_score[b] > obj_thresh)
            if mask.sum() == 0:
                continue

            bx1 = x1[b][mask].clamp(0, img_size)
            by1 = y1[b][mask].clamp(0, img_size)
            bx2 = x2[b][mask].clamp(0, img_size)
            by2 = y2[b][mask].clamp(0, img_size)
            bscore = score[b][mask]
            bcls = cls_idx[b][mask].float()

            valid = (bx2 > bx1) & (by2 > by1)
            if valid.sum() == 0:
                continue

            dets = torch.stack([
                bx1[valid], by1[valid], bx2[valid], by2[valid],
                bscore[valid], bcls[valid]
            ], dim=1)

            all_dets[b].append(dets)

    results = []
    for b in range(B):
        if len(all_dets[b]) == 0:
            results.append(torch.zeros(0, 6, device=device))
        else:
            results.append(torch.cat(all_dets[b], dim=0))

    return results
```

### 5.9 `postprocess.py`

```python
def iou_batch(boxes, box):
    x1 = torch.max(boxes[:, 0], box[0])
    y1 = torch.max(boxes[:, 1], box[1])
    x2 = torch.min(boxes[:, 2], box[2])
    y2 = torch.min(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
             (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area_b = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(dets, iou_thresh=0.5, img_size=640, min_box_size=2):
    if dets.shape[0] == 0:
        return dets

    dets = dets.clone()
    dets[:, 0] = dets[:, 0].clamp(0, img_size)
    dets[:, 1] = dets[:, 1].clamp(0, img_size)
    dets[:, 2] = dets[:, 2].clamp(0, img_size)
    dets[:, 3] = dets[:, 3].clamp(0, img_size)

    w = dets[:, 2] - dets[:, 0]
    h = dets[:, 3] - dets[:, 1]
    valid = (w > min_box_size) & (h > min_box_size)
    dets = dets[valid]

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
                conf_thresh=0.3, iou_thresh=0.5,
                max_detections=300, obj_thresh=0.5):
    decoded = decode_predictions(raw_outputs, strides, num_classes,
                                 img_size, conf_thresh, obj_thresh)
    results = []
    for d in decoded:
        d_nms = nms(d, iou_thresh, img_size)
        if d_nms.shape[0] > max_detections:
            order = d_nms[:, 4].argsort(descending=True)
            d_nms = d_nms[order[:max_detections]]
        results.append(d_nms)
    return results
```

### 5.10 `evaluator.py`

```python
def evaluate(model, loader, device, num_classes, img_size,
             strides, conf_thresh=0.3, nms_iou=0.5, match_iou=0.5,
             class_names=None):
    """
    Trả về dict metrics tổng + per-class.
    - nms_iou: IoU threshold cho NMS
    - match_iou: IoU threshold để match TP/FP (mAP@match_iou)
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
                           conf_thresh, nms_iou)

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
                    is_tp = 1 if (best_iou >= match_iou and best_gi >= 0) else 0
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
```