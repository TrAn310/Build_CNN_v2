"""
src/run_train.py
Chạy train full dataset với GPU (RTX 5060 Ti).

Cách dùng:
    cd src
    python run_train.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CẤU HÌNH — CHỈNH CHO RTX 5060 Ti (16GB VRAM)
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
    'epochs':      50,
    'batch_size':  16,         # RTX 5060 Ti 16GB → 16 ổn
    'lr':          1e-3,
    'num_workers': 4,          # Windows: 0 nếu bị lỗi, Linux: 4-8
    'device':      'cuda',
    'width_mult':  1.0,
    'amp':         True,       # Mixed precision (nhanh hơn ~30%)
}


# ============================================================
# ĐỌC CLASS
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

    from models.detector import CustomPPEDetector
    from dataset.dataset import PPEDetectionDataset
    from dataset.collate import collate_fn
    from training.target_assigner import build_targets
    from training.losses import DetectionLoss
    from training.train import train_one_epoch, validate

    print('=' * 70)
    print('TRAIN FULL DATASET — GPU MODE')
    print('=' * 70)

    # --- CUDA check ---
    if not torch.cuda.is_available():
        print('LỖI: CUDA không khả dụng. Không chạy được GPU.')
        print('Chạy lệnh này để kiểm tra:')
        print('  python -c "import torch; print(torch.cuda.is_available())"')
        return

    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    torch.backends.cudnn.benchmark = True

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
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], collate_fn=collate_fn,
        pin_memory=True,
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

    # AMP (mixed precision) — tăng tốc trên GPU
    scaler = torch.amp.GradScaler('cuda') if CONFIG['amp'] else None
    if CONFIG['amp']:
        print('AMP (mixed precision): ON')

    strides = model.strides
    feat_sizes = [(CONFIG['img_size']//s, CONFIG['img_size']//s) for s in strides]

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    log_path = os.path.join(CONFIG['log_dir'], 'train_gpu_log.txt')

    best_val = float('inf')
    start_time = time.time()

    print('\n' + '=' * 70)
    print('BẮT ĐẦU TRAIN')
    print('=' * 70)

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write(f'GPU: {torch.cuda.get_device_name(0)}\n')
        log_f.write(f'num_classes={nc}\n')
        log_f.write(f'class_names={class_names}\n')
        log_f.write(f'train_samples={len(train_ds)}\n')
        log_f.write(f'val_samples={len(val_ds)}\n\n')

        for epoch in range(1, CONFIG['epochs']+1):
            # ---- TRAIN ----
            model.train()
            total_loss = 0.0
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}/{CONFIG["epochs"]}')
            for imgs, targets in pbar:
                imgs = imgs.to(device, non_blocking=True)
                targets = [t.to(device) for t in targets]

                optimizer.zero_grad(set_to_none=True)

                if CONFIG['amp']:
                    with torch.amp.autocast('cuda'):
                        raw = model(imgs)
                        tgt_list = build_targets(targets, nc, CONFIG['img_size'],
                                                 strides, feat_sizes)
                        loss, bl, ol, cl = criterion(raw, tgt_list)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                    imgs = imgs.to(device, non_blocking=True)
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

            # Log
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

            # Save last mỗi 5 epoch
            if epoch % 5 == 0:
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