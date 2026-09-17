"""
run_overfit.py
Test overfit 1 ảnh - tự động đọc số class từ data.yaml.
"""

import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.dataset_info import get_dataset_info


# ============================================================
# 1. MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
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
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        c1 = int(32 * width_mult); c2 = int(64 * width_mult)
        c3 = int(128 * width_mult); c4 = int(256 * width_mult); c5 = int(512 * width_mult)
        self.stem = ConvBlock(3, c1, k=3, s=2)
        self.stage1 = nn.Sequential(ConvBlock(c1, c2, k=3, s=2), ResidualBlock(c2, c2))
        self.stage2 = nn.Sequential(ConvBlock(c2, c3, k=3, s=2), ResidualBlock(c3, c3), ResidualBlock(c3, c3))
        self.stage3 = nn.Sequential(ConvBlock(c3, c4, k=3, s=2), ResidualBlock(c4, c4), ResidualBlock(c4, c4))
        self.stage4 = nn.Sequential(ConvBlock(c4, c5, k=3, s=2), ResidualBlock(c5, c5), ResidualBlock(c5, c5))
        self.out_channels = [c3, c4, c5]
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x)
        p3 = self.stage2(x); p4 = self.stage3(p3); p5 = self.stage4(p4)
        return p3, p4, p5


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
        l5 = self.lat5(p5); l4 = self.lat4(p4); l3 = self.lat3(p3)
        up5 = F.interpolate(l5, scale_factor=2, mode='nearest')
        f4 = self.smooth4(torch.cat([up5, l4], dim=1))
        up4 = F.interpolate(f4, scale_factor=2, mode='nearest')
        f3 = self.smooth3(torch.cat([up4, l3], dim=1))
        f5 = self.down5(f4)
        return f3, f4, f5


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=3, hidden=128):
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(ConvBlock(in_channels, hidden, k=3), ConvBlock(hidden, hidden, k=3))
        self.pred_box = nn.Conv2d(hidden, 5, 1)
        self.pred_cls = nn.Conv2d(hidden, num_classes, 1)
        nn.init.constant_(self.pred_box.bias[4], -4.0)
    def forward(self, x):
        feat = self.stem(x)
        return torch.cat([self.pred_box(feat), self.pred_cls(feat)], dim=1)


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


# ============================================================
# 2. TARGET ASSIGNER
# ============================================================

def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    """
    Gán GT box vào VÙNG 3×3 cells quanh tâm object cho MỖI scale.
    Giải quyết:
      - Miss object do ghi đè cùng cell (2 objects gần nhau)
      - Object lớn cần nhiều cells để model học
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

            for s_idx, stride in enumerate(strides):
                H, W = feat_sizes[s_idx]
                gx = max(0, min(W - 1, int(xc / stride)))
                gy = max(0, min(H - 1, int(yc / stride)))

                # === GÁN VÙNG 3x3 QUANH CELL (gy, gx) ===
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = gy + dy
                        xx = gx + dx
                        if yy < 0 or yy >= H or xx < 0 or xx >= W:
                            continue

                        # Encode box TƯƠNG ĐỐI so với cell (yy, xx), KHÔNG phải (gy, gx)
                        tx = xc / stride - xx
                        ty = yc / stride - yy
                        tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
                        th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

                        t = all_targets[s_idx][b]
                        # Chỉ ghi đè nếu cell chưa có object khác
                        if t[4, yy, xx] < 0.5:
                            t[0, yy, xx] = tx
                            t[1, yy, xx] = ty
                            t[2, yy, xx] = tw
                            t[3, yy, xx] = th
                            t[4, yy, xx] = 1.0
                            t[5 + cls, yy, xx] = 1.0

    return all_targets

# ============================================================
# 3. LOSS
# ============================================================

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 pos_weight=1.0, neg_weight=1.0):
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
            p_box = pred[:, :4]; p_obj = pred[:, 4]; p_cls = pred[:, 5:]
            t_box = tgt[:, :4]; t_obj = tgt[:, 4]; t_cls = tgt[:, 5:]
            pos_mask = t_obj > 0.5
            if pos_mask.sum() > 0:
                pb = p_box.permute(0, 2, 3, 1)[pos_mask]
                tb = t_box.permute(0, 2, 3, 1)[pos_mask]
                total_box = total_box + F.smooth_l1_loss(pb, tb, reduction='sum')
                n_pos += pos_mask.sum().item()
            obj_loss_map = self.bce(p_obj, t_obj)
            w = torch.where(pos_mask,
                            torch.full_like(obj_loss_map, self.pos_weight),
                            torch.full_like(obj_loss_map, self.neg_weight))
            total_obj = total_obj + (obj_loss_map * w).sum()
            if pos_mask.sum() > 0:
                pc = p_cls.permute(0, 2, 3, 1)[pos_mask]
                tc = t_cls.permute(0, 2, 3, 1)[pos_mask]
                tc_idx = tc.argmax(dim=1)
                total_cls = total_cls + self.ce(pc, tc_idx)
        n_pos = max(n_pos, 1)
        box_loss = total_box / n_pos
        obj_loss = total_obj / (n_pos * 100.0)
        cls_loss = total_cls / n_pos
        total = self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_cls * cls_loss
        return total, box_loss, obj_loss, cls_loss


# ============================================================
# 4. DECODER + NMS
# ============================================================

def iou_batch(boxes, box):
    x1 = torch.max(boxes[:, 0], box[0]); y1 = torch.max(boxes[:, 1], box[1])
    x2 = torch.min(boxes[:, 2], box[2]); y2 = torch.min(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms(dets, iou_thresh=0.5, score_thresh=0.5, dist_thresh=80):
    """
    NMS cải tiến: gộp box nếu IoU cao HOẶC tâm gần nhau.
    Xử lý trường hợp multi-cell assignment tạo nhiều box cho 1 object.
    """
    if dets.shape[0] == 0:
        return dets

    # Lọc theo score trước
    dets = dets[dets[:, 4] >= score_thresh]
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

            best_cx = (best[0] + best[2]) / 2
            best_cy = (best[1] + best[3]) / 2
            other_cx = (cls_dets[1:, 0] + cls_dets[1:, 2]) / 2
            other_cy = (cls_dets[1:, 1] + cls_dets[1:, 3]) / 2
            dists = torch.sqrt((best_cx - other_cx)**2 + (best_cy - other_cy)**2)

            keep_mask = (ious < iou_thresh) & (dists > dist_thresh)
            cls_dets = cls_dets[1:][keep_mask]

    if len(keep) == 0:
        return torch.zeros(0, 6, device=dets.device)
    return torch.stack(keep, dim=0)


def decode_predictions(raw_outputs, strides, num_classes, img_size=640, conf_thresh=0.3):
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
        tx = torch.sigmoid(out[:, 0:1]); ty = torch.sigmoid(out[:, 1:2])
        tw = out[:, 2:3]; th = out[:, 3:4]
        obj = torch.sigmoid(out[:, 4:5])
        cls_prob = torch.softmax(out[:, 5:], dim=1)
        cx = (tx + grid_x) * stride; cy = (ty + grid_y) * stride
        w = torch.exp(tw.clamp(-4, 4)) * stride
        h = torch.exp(th.clamp(-4, 4)) * stride
        x1 = (cx - w / 2).clamp(0, img_size); y1 = (cy - h / 2).clamp(0, img_size)
        x2 = (cx + w / 2).clamp(0, img_size); y2 = (cy + h / 2).clamp(0, img_size)
        max_cls_prob, cls_idx = cls_prob.max(dim=1, keepdim=True)
        score = obj * max_cls_prob
        for b in range(B):
            s = score[b, 0]
            mask = s > conf_thresh
            if mask.sum() == 0:
                continue
            det = torch.stack([
                x1[b, 0][mask], y1[b, 0][mask],
                x2[b, 0][mask], y2[b, 0][mask],
                s[mask], cls_idx[b, 0][mask].float(),
            ], dim=1)
            results[b].append(det)
    final = []
    for b in range(B):
        if len(results[b]) == 0:
            final.append(torch.zeros(0, 6, device=device))
        else:
            final.append(torch.cat(results[b], dim=0))
    return final


def postprocess(raw_outputs, strides, num_classes, img_size=640,
                conf_thresh=0.3, iou_thresh=0.5, score_thresh=0.5):
    decoded = decode_predictions(raw_outputs, strides, num_classes, img_size, conf_thresh)
    return [nms(d, iou_thresh, score_thresh) for d in decoded]


# ============================================================
# 5. VISUALIZATION
# ============================================================

def get_class_color(cls_id):
    import colorsys
    hue = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_boxes(img_rgb, dets, class_names=None):
    if isinstance(dets, torch.Tensor):
        dets = dets.detach().cpu().numpy()
    img = img_rgb.copy()
    if dets is None or len(dets) == 0:
        return img
    for d in dets:
        x1, y1, x2, y2, score, cls = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        color = get_class_color(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = (class_names[cls] if class_names and cls < len(class_names)
                else f'cls{cls}')
        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# ============================================================
# 6. PARSER LABEL
# ============================================================

def parse_yolo_label(label_path):
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


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='overfit_test/test.jpg')
    parser.add_argument('--lbl', type=str, default='overfit_test/test.txt')
    parser.add_argument('--data_yaml', type=str, default='Data/data.yaml')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default='outputs/overfit')
    parser.add_argument('--log_every', type=int, default=20)
    args = parser.parse_args()

    # Auto-detect num_classes
    if args.num_classes is not None:
        num_classes = args.num_classes
        class_names = [f'class_{i}' for i in range(num_classes)]
        print(f'[Config] num_classes = {num_classes} (from CLI)')
    else:
        num_classes, class_names = get_dataset_info(args.data_yaml)
        print(f'[Config] Auto-detected num_classes = {num_classes}')
        print(f'[Config] class_names = {class_names}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    print('=' * 60)
    print(f'[Overfit] Image: {args.img}')
    print(f'[Overfit] Label: {args.lbl}')
    print(f'[Overfit] Device: {device}')
    print(f'[Overfit] num_classes: {num_classes}')
    print(f'[Overfit] Epochs: {args.epochs}, LR: {args.lr}')
    print('=' * 60)

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        print(f'KHÔNG ĐỌC ĐƯỢC ẢNH: {args.img}')
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))
    img_tensor = img_resized.astype('float32') / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).contiguous()

    targets = parse_yolo_label(args.lbl)
    print(f'[Overfit] GT: {targets.shape[0]} objects')

    if targets.shape[0] == 0:
        print('CẢNH BÁO: file label rỗng hoặc sai path!')
        return

    max_cls = int(targets[:, 0].max().item())
    if max_cls >= num_classes:
        print(f'CẢNH BÁO: Label có class id = {max_cls} nhưng num_classes = {num_classes}')
        return

    print(f'[Overfit] Label data:\n{targets}')

    model = CustomPPEDetector(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DetectionLoss(num_classes=num_classes)
    strides = model.strides
    feat_sizes = [(args.img_size // s, args.img_size // s) for s in strides]

    x = img_tensor.unsqueeze(0).to(device)
    t = targets.unsqueeze(0).to(device)

    model.train()
    for epoch in range(1, args.epochs + 1):
        raw = model(x)
        tgt_list = build_targets([t[0]], num_classes, args.img_size, strides, feat_sizes)
        loss, bl, ol, cl = criterion(raw, tgt_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0:
            print(f'Epoch {epoch:4d} | loss={loss.item():.4f} '
                  f'box={bl.item():.4f} obj={ol.item():.4f} cls={cl.item():.4f}')

    model.eval()
    with torch.no_grad():
        raw = model(x)
        dets = postprocess(raw, strides, num_classes,
                   args.img_size, conf_thresh=0.5, iou_thresh=0.5,
                   score_thresh=0.7)[0]
    print(f'\n[Overfit] Detections: {dets.shape[0]}')
    if dets.shape[0] > 0:
        for d in dets.cpu().numpy():
            cid = int(d[5])
            cname = class_names[cid] if cid < len(class_names) else f'cls{cid}'
            print(f'  [{cname}] score={d[4]:.3f}  box=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f}]')

    vis = draw_boxes(img_resized, dets, class_names)
    out_path = os.path.join(args.out, 'overfit_result.jpg')
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f'[Overfit] Saved: {out_path}')

    torch.save(model.state_dict(), os.path.join(args.out, 'overfit_model.pth'))
    print('[Overfit] Done.')


if __name__ == '__main__':
    main()