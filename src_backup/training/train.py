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
