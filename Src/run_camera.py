"""
src/run_camera.py
Chạy detection qua webcam — self-contained.
"""

import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorsys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CẤU HÌNH
# ============================================================
WEIGHTS = 'src/outputs/checkpoints/best_model.pth'
DATA_YAML = 'src/Data/data.yaml'
CAMERA_ID = 0             # 0 = webcam mặc định, 1 = camera thứ 2
IMG_SIZE = 640
CONF_THRESH = 0.3
IOU_THRESH = 0.5
DEVICE = 'cpu'            # 'cuda' nếu có GPU


# ============================================================
# ĐỌC CLASS
# ============================================================
def load_class_names(yaml_path):
    if not os.path.exists(yaml_path):
        return None, None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    nc = None
    for line in content.splitlines():
        if line.strip().startswith('nc:'):
            try: nc = int(line.split(':', 1)[1].strip())
            except: pass
            break
    names = None
    if 'names:' in content:
        idx = content.find('names:')
        rest = content[idx + len('names:'):].strip()
        if rest.startswith('['):
            end = rest.find(']')
            if end != -1:
                names = [s.strip().strip("'\"").strip() for s in rest[1:end].split(',')]
    return nc, names


# ============================================================
# MODEL
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


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
        c1=int(32*width_mult); c2=int(64*width_mult)
        c3=int(128*width_mult); c4=int(256*width_mult); c5=int(512*width_mult)
        self.stem = ConvBlock(3, c1, k=3, s=2)
        self.stage1 = nn.Sequential(ConvBlock(c1,c2,k=3,s=2), ResidualBlock(c2,c2))
        self.stage2 = nn.Sequential(ConvBlock(c2,c3,k=3,s=2), ResidualBlock(c3,c3), ResidualBlock(c3,c3))
        self.stage3 = nn.Sequential(ConvBlock(c3,c4,k=3,s=2), ResidualBlock(c4,c4), ResidualBlock(c4,c4))
        self.stage4 = nn.Sequential(ConvBlock(c4,c5,k=3,s=2), ResidualBlock(c5,c5), ResidualBlock(c5,c5))
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
        self.smooth4 = ConvBlock(out_channels*2, out_channels, k=3)
        self.smooth3 = ConvBlock(out_channels*2, out_channels, k=3)
        self.down5 = ConvBlock(out_channels, out_channels, k=3, s=2)
    def forward(self, p3, p4, p5):
        l5=self.lat5(p5); l4=self.lat4(p4); l3=self.lat3(p3)
        up5 = F.interpolate(l5, scale_factor=2, mode='nearest')
        f4 = self.smooth4(torch.cat([up5, l4], dim=1))
        up4 = F.interpolate(f4, scale_factor=2, mode='nearest')
        f3 = self.smooth3(torch.cat([up4, l3], dim=1))
        f5 = self.down5(f4)
        return f3, f4, f5


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=3, hidden=128):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(in_channels, hidden, k=3), ConvBlock(hidden, hidden, k=3))
        self.pred_box = nn.Conv2d(hidden, 5, 1)
        self.pred_cls = nn.Conv2d(hidden, num_classes, 1)
    def forward(self, x):
        feat = self.stem(x)
        return torch.cat([self.pred_box(feat), self.pred_cls(feat)], dim=1)


class CustomPPEDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(self.backbone.out_channels)
        self.head_p3 = DetectionHead(128, num_classes)
        self.head_p4 = DetectionHead(128, num_classes)
        self.head_p5 = DetectionHead(128, num_classes)
        self.strides = [8, 16, 32]
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return [self.head_p3(f3), self.head_p4(f4), self.head_p5(f5)]


# ============================================================
# DECODE + NMS
# ============================================================
def iou_batch(boxes, box):
    x1 = torch.max(boxes[:,0], box[0]); y1 = torch.max(boxes[:,1], box[1])
    x2 = torch.min(boxes[:,2], box[2]); y2 = torch.min(boxes[:,3], box[3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    area_a = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    area_b = (box[2]-box[0]) * (box[3]-box[1])
    return inter / (area_a + area_b - inter + 1e-6)


def nms(dets, iou_thresh=0.5):
    if dets.shape[0] == 0: return dets
    keep = []
    for c in dets[:,5].unique():
        cd = dets[dets[:,5]==c]
        cd = cd[cd[:,4].argsort(descending=True)]
        while cd.shape[0] > 0:
            keep.append(cd[0])
            if cd.shape[0] == 1: break
            ious = iou_batch(cd[1:,:4], cd[0,:4])
            cd = cd[1:][ious < iou_thresh]
    return torch.stack(keep, dim=0) if keep else torch.zeros(0,6)


@torch.no_grad()
def detect(model, img_rgb, num_classes, img_size, device, conf=0.3, iou=0.5):
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = img_resized.astype('float32') / 255.0
    tensor = torch.from_numpy(tensor).permute(2,0,1).unsqueeze(0).to(device)
    raw = model(tensor)

    all_dets = []
    for out, stride in zip(raw, model.strides):
        B, C, H, W = out.shape
        ys, xs = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        gx = xs.float().view(1,1,H,W); gy = ys.float().view(1,1,H,W)
        tx = torch.sigmoid(out[:,0:1]); ty = torch.sigmoid(out[:,1:2])
        tw = out[:,2:3]; th = out[:,3:4]
        obj = torch.sigmoid(out[:,4:5])
        cls_prob = torch.softmax(out[:,5:], dim=1)
        cx = (tx + gx) * stride; cy = (ty + gy) * stride
        w = torch.exp(tw.clamp(-4,4)) * stride
        h = torch.exp(th.clamp(-4,4)) * stride
        x1 = (cx - w/2).clamp(0, img_size); y1 = (cy - h/2).clamp(0, img_size)
        x2 = (cx + w/2).clamp(0, img_size); y2 = (cy + h/2).clamp(0, img_size)
        max_p, cls_i = cls_prob.max(dim=1, keepdim=True)
        score = obj * max_p
        s = score[0,0]; mask = s > conf
        if mask.sum() > 0:
            det = torch.stack([x1[0,0][mask], y1[0,0][mask], x2[0,0][mask],
                               y2[0,0][mask], s[mask],
                               cls_i[0,0][mask].float()], dim=1)
            all_dets.append(det)

    if not all_dets:
        return torch.zeros(0,6)
    return nms(torch.cat(all_dets, dim=0), iou)


# ============================================================
# MÀU + VẼ
# ============================================================
def get_color(i):
    h = (i * 0.618) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
    return (int(r*255), int(g*255), int(b*255))


def draw(img, dets, names):
    if isinstance(dets, torch.Tensor):
        dets = dets.detach().cpu().numpy()
    out = img.copy()
    for d in dets:
        x1,y1,x2,y2,score,cls = d
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cls = int(cls)
        c = get_color(cls)
        cv2.rectangle(out, (x1,y1), (x2,y2), c, 2)
        name = names[cls] if names and cls < len(names) else f'cls{cls}'
        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
        cv2.putText(out, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    print('=' * 60)
    print('PPE CAMERA DEMO')
    print('=' * 60)

    nc, names = load_class_names(DATA_YAML)
    if nc is None:
        print(f'Không đọc được data.yaml: {DATA_YAML}')
        return
    print(f'num_classes = {nc}')
    print(f'class_names = {names}')

    device = torch.device(DEVICE)
    model = CustomPPEDetector(num_classes=nc).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    print(f'Loaded weights: {WEIGHTS}')

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f'KHÔNG mở được camera {CAMERA_ID}!')
        return

    w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera: {w_cam}x{h_cam}')
    print('Nhấn q để thoát.')

    fps_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Không đọc được frame.')
            break

        t0 = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detect(model, img_rgb, nc, IMG_SIZE, device,
                      conf=CONF_THRESH, iou=IOU_THRESH)

        if dets.shape[0] > 0:
            sx = w_cam / IMG_SIZE; sy = h_cam / IMG_SIZE
            dets[:, 0] *= sx; dets[:, 2] *= sx
            dets[:, 1] *= sy; dets[:, 3] *= sy

        vis = draw(img_rgb, dets, names)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        dt = time.time() - t0
        fps_list.append(1.0 / max(dt, 1e-6))
        fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f'Dets: {dets.shape[0]}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('PPE Camera', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Đã thoát.')


if __name__ == '__main__':
    main()