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