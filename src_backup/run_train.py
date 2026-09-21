"""
src/run_train.py
Chạy train full dataset — CPU mode.

Cách dùng:
    cd src
    python run_train.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CẤU HÌNH — SỬA Ở ĐÂY
# ============================================================
CONFIG = {
    'train_img':   'Data/train/images',
    'train_lbl':   'Data/train/labels',
    'val_img':     'Data/valid/images',
    'val_lbl':     'Data/valid/labels',
    'data_yaml':   'Data/data.yaml',
    'save_dir':    'outputs/checkpoints',
    'log_dir':     'outputs/logs',
    'img_size':    640,
    'epochs':      30,           # CPU: giảm còn 20-30
    'batch_size':  4,            # CPU: nhỏ (4-8)
    'lr':          1e-3,
    'num_workers': 0,            # Windows: 0
    'device':      'cpu',
    'width_mult':  1.0,
    'save_every':  5,            # save last_model mỗi N epoch
}


# ============================================================
# ĐỌC CLASS TỪ data.yaml
# ============================================================
def load_class_names(yaml_path):
    if not os.path.exists(yaml_path):
        return None, None
    nc = None
    names = None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for line in content.splitlines():
        if line.strip().startswith('nc:'):
            try:
                nc = int(line.split(':', 1)[1].strip())
            except ValueError:
                pass
            break
    if 'names:' in content:
        idx = content.find('names:')
        rest = content[idx + len('names:'):].strip()
        if rest.startswith('['):
            end = rest.find(']')
            if end != -1:
                names = [s.strip().strip("'\"").strip()
                         for s in rest[1:end].split(',')]
    return nc, names


# ============================================================
# MAIN
# ============================================================
def main():
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from models.detector import CustomPPEDetector
    from dataset.dataset import PPEDetectionDataset
    from dataset.collate import collate_fn
    from training.target_assigner import build_targets
    from training.losses import DetectionLoss

    print('=' * 70)
    print('TRAIN FULL DATASET — CPU MODE')
    print('=' * 70)

    device = torch.device('cpu')
    print(f'Device: {device}')
    print(f'CPU count: {os.cpu_count()}')

    # --- Classes ---
    nc, class_names = load_class_names(CONFIG['data_yaml'])
    if nc is None:
        print(f'Không đọc được data.yaml: {CONFIG["data_yaml"]}')
        return
    print(f'\nnum_classes = {nc}')
    print(f'class_names = {class_names}')

    # --- Dataset ---
    train_ds = PPEDetectionDataset(CONFIG['train_img'], CONFIG['train_lbl'],
                                   img_size=CONFIG['img_size'], augment=True)
    val_ds = PPEDetectionDataset(CONFIG['val_img'], CONFIG['val_lbl'],
                                 img_size=CONFIG['img_size'], augment=False)
    print(f'\nTrain samples: {len(train_ds)}')
    print(f'Val samples:   {len(val_ds)}')
    print(f'Batch size:    {CONFIG["batch_size"]}')
    print(f'Epochs:        {CONFIG["epochs"]}')
    print(f'Image size:    {CONFIG["img_size"]}')

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], collate_fn=collate_fn,
    )

    # --- Model ---
    model = CustomPPEDetector(num_classes=nc,
                              width_mult=CONFIG['width_mult']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel params: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'])
    criterion = DetectionLoss(num_classes=nc)

    strides = model.strides
    feat_sizes = [(CONFIG['img_size']//s, CONFIG['img_size']//s) for s in strides]

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    log_path = os.path.join(CONFIG['log_dir'], 'train_cpu_log.txt')

    best_val = float('inf')
    start_time = time.time()

    print('\n' + '=' * 70)
    print('BẮT ĐẦU TRAIN')
    print('=' * 70)

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f'device=cpu\n')
        log_f.write(f'num_classes={nc}\n')
        log_f.write(f'class_names={class_names}\n')
        log_f.write(f'train_samples={len(train_ds)}\n')
        log_f.write(f'val_samples={len(val_ds)}\n\n')

        for epoch in range(1, CONFIG['epochs']+1):
            # ---- TRAIN ----
            model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}/{CONFIG["epochs"]}')
            for imgs, targets in pbar:
                imgs = imgs.to(device)
                targets = [t.to(device) for t in targets]

                optimizer.zero_grad(set_to_none=True)

                raw = model(imgs)
                tgt_list = build_targets(targets, nc, CONFIG['img_size'],
                                         strides, feat_sizes)
                loss, bl, ol, cl = criterion(raw, tgt_list)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.3f}',
                                 box=f'{bl.item():.3f}',
                                 obj=f'{ol.item():.3f}',
                                 cls=f'{cl.item():.3f}')

            tr_loss = total_loss / max(len(train_loader), 1)

            # ---- VALIDATE ----
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    targets = [t.to(device) for t in targets]
                    raw = model(imgs)
                    tgt_list = build_targets(targets, nc, CONFIG['img_size'],
                                             strides, feat_sizes)
                    loss, _, _, _ = criterion(raw, tgt_list)
                    val_total += loss.item()
            val_loss = val_total / max(len(val_loader), 1)

            scheduler.step()
            lr_now = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            line = (f'Epoch {epoch:03d}/{CONFIG["epochs"]} | '
                    f'train={tr_loss:.4f} val={val_loss:.4f} '
                    f'lr={lr_now:.2e} | {elapsed/60:.1f} phút')
            print(line)
            log_f.write(line + '\n')
            log_f.flush()

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(CONFIG['save_dir'], 'best_model.pth'))
                print(f'  -> saved best (val={best_val:.4f})')

            if epoch % CONFIG['save_every'] == 0:
                torch.save(model.state_dict(),
                           os.path.join(CONFIG['save_dir'], 'last_model.pth'))

        torch.save(model.state_dict(),
                   os.path.join(CONFIG['save_dir'], 'last_model.pth'))

    print('\n' + '=' * 70)
    print(f'DONE! Best val loss = {best_val:.4f}')
    print(f'Checkpoint: {CONFIG["save_dir"]}/best_model.pth')
    print(f'Log: {log_path}')
    print(f'Tổng thời gian: {(time.time()-start_time)/60:.1f} phút')
    print('=' * 70)


if __name__ == '__main__':
    main()