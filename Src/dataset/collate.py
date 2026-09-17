"""
collate.py
Custom collate: mỗi ảnh có số object khác nhau nên không stack target.
"""

import torch


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets