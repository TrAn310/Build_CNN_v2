"""
dataset.py
PPEDetectionDataset cho custom detector.
Có mosaic augmentation (YOLOv5 style).
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from Src.dataset.parser import parse_yolo_label


class PPEDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=False, cache=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.cache = cache

        self.files = sorted([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Cache
        self.cache_imgs = None
        self.cache_labels = None
        if self.cache:
            n = len(self.files)
            print(f'[Dataset] Đang cache {n} ảnh (uint8)...')
            ram_gb = n * img_size * img_size * 3 / (1024**3)
            print(f'[Dataset] Ước tính cần ~{ram_gb:.1f} GB RAM cho cache')

            self.cache_imgs = []
            self.cache_labels = []
            for i, fname in enumerate(self.files):
                img_path = os.path.join(self.img_dir, fname)
                label_path = os.path.join(self.label_dir,
                                          os.path.splitext(fname)[0] + '.txt')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
                self.cache_imgs.append(img)

                targets = parse_yolo_label(label_path)
                if targets is None or not isinstance(targets, torch.Tensor):
                    targets = torch.zeros((0, 5), dtype=torch.float32)
                self.cache_labels.append(targets)

                if (i + 1) % 1000 == 0:
                    print(f'[Dataset] Đã cache {i+1}/{n}')

            print(f'[Dataset] Cache xong! {n} mẫu đã vào RAM.')

    def __len__(self):
        return len(self.files)

    # ============================================================
    # Helper: load 1 ảnh + label
    # ============================================================
    def _load_image_and_label(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        label_path = os.path.join(self.label_dir,
                                  os.path.splitext(fname)[0] + '.txt')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        targets = parse_yolo_label(label_path)
        if targets is None or not isinstance(targets, torch.Tensor):
            targets = torch.zeros((0, 5), dtype=torch.float32)
        return img, targets

    # ============================================================
    # Mosaic augmentation
    # ============================================================
    def _mosaic(self, idx):
        """Ghép 4 ảnh thành 1 (YOLOv5 style)."""
        indices = [idx] + [np.random.randint(0, len(self.files)) for _ in range(3)]
        s = self.img_size
        yc, xc = [int(np.random.uniform(s * 0.4, s * 0.6)) for _ in range(2)]

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        mosaic_targets = []

        for i, index in enumerate(indices):
            img, targets = self._load_image_and_label(index)
            h, w = img.shape[:2]

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s), min(yc + h, s)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            if targets.shape[0] > 0:
                t = targets.clone()
                # xywh (norm) -> xyxy (pixel)
                xc_t = t[:, 1] * w
                yc_t = t[:, 2] * h
                w_t = t[:, 3] * w
                h_t = t[:, 4] * h
                x1_t = xc_t - w_t / 2
                y1_t = yc_t - h_t / 2
                x2_t = xc_t + w_t / 2
                y2_t = yc_t + h_t / 2
                # Dịch chuyển
                x1_t = x1_t + x1a - x1b
                y1_t = y1_t + y1a - y1b
                x2_t = x2_t + x1a - x1b
                y2_t = y2_t + y1a - y1b
                # Clip
                x1_t = x1_t.clamp(0, s)
                y1_t = y1_t.clamp(0, s)
                x2_t = x2_t.clamp(0, s)
                y2_t = y2_t.clamp(0, s)
                # Lọc box quá nhỏ
                keep = (x2_t - x1_t > 4) & (y2_t - y1_t > 4)
                if keep.sum() > 0:
                    x1_t = x1_t[keep]; y1_t = y1_t[keep]
                    x2_t = x2_t[keep]; y2_t = y2_t[keep]
                    cls_t = t[keep, 0]
                    # Về normalized xywh
                    xc_n = (x1_t + x2_t) / 2 / s
                    yc_n = (y1_t + y2_t) / 2 / s
                    w_n = (x2_t - x1_t) / s
                    h_n = (y2_t - y1_t) / s
                    mosaic_targets.append(torch.stack([
                        cls_t, xc_n, yc_n, w_n, h_n
                    ], dim=1))

        if len(mosaic_targets) > 0:
            targets = torch.cat(mosaic_targets, dim=0)
        else:
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return mosaic_img, targets

    # ============================================================
    # __getitem__
    # ============================================================
    def __getitem__(self, idx):
        use_mosaic = self.augment and np.random.rand() < 0.5

        if use_mosaic:
            img, targets = self._mosaic(idx)
            img = img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        else:
            # Load thường
            if self.cache and self.cache_imgs is not None:
                img = self.cache_imgs[idx]
                targets = self.cache_labels[idx].clone()
                img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
            else:
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
                if targets is None or not isinstance(targets, torch.Tensor):
                    targets = torch.zeros((0, 5), dtype=torch.float32)

        # ---- Augment (chỉ khi không mosaic, để tránh conflict) ----
        if self.augment and not use_mosaic:
            # 1. Horizontal flip
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                if targets.shape[0] > 0:
                    targets = targets.clone()
                    targets[:, 1] = 1.0 - targets[:, 1]

            # 2. Color jitter
            if torch.rand(1).item() < 0.5:
                img = TF.adjust_brightness(img, 1.0 + (torch.rand(1).item() - 0.5) * 0.4)
                img = TF.adjust_contrast(img, 1.0 + (torch.rand(1).item() - 0.5) * 0.4)
                img = TF.adjust_saturation(img, 1.0 + (torch.rand(1).item() - 0.5) * 0.4)

            # 3. Random translate nhẹ
            if torch.rand(1).item() < 0.3 and targets.shape[0] > 0:
                max_shift = 0.05
                dx = (torch.rand(1).item() - 0.5) * 2 * max_shift
                dy = (torch.rand(1).item() - 0.5) * 2 * max_shift
                img = TF.affine(img, angle=0,
                                translate=[int(dx * self.img_size),
                                           int(dy * self.img_size)],
                                scale=1.0, shear=0.0)
                targets = targets.clone()
                targets[:, 1] = torch.clamp(targets[:, 1] + dx, 0, 1)
                targets[:, 2] = torch.clamp(targets[:, 2] + dy, 0, 1)

            # 4. Random scale nhẹ
            if torch.rand(1).item() < 0.3 and targets.shape[0] > 0:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
                img = TF.affine(img, angle=0, translate=[0, 0],
                                scale=scale, shear=0.0)
                targets = targets.clone()
                targets[:, 3] = torch.clamp(targets[:, 3] * scale, 0, 1)
                targets[:, 4] = torch.clamp(targets[:, 4] * scale, 0, 1)

        return img, targets