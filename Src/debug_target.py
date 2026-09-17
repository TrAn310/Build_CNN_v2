"""
debug_target.py
Kiểm tra target assignment cho 2 person trong ảnh overfit.
"""

import sys
import os
sys.path.insert(0, r'D:\NCKH\CNN_ver2\Build_CNN_v2\Src')

import torch
from run_overfit import build_targets, parse_yolo_label

# Load label — dùng absolute path
LABEL_PATH = r'D:\NCKH\CNN_ver2\Build_CNN_v2\Src\overfit_test\test.txt'

targets = parse_yolo_label(LABEL_PATH)
print(f'Loaded label: {LABEL_PATH}')
print(f'GT objects: {targets.shape[0]}')
for i, row in enumerate(targets):
    cls = int(row[0].item())
    xc, yc, w, h = row[1:].tolist()
    print(f'  [{i}] cls={cls} xc={xc:.4f} yc={yc:.4f} w={w:.4f} h={h:.4f}')

# Build targets
img_size = 640
strides = [8, 16, 32]
feat_sizes = [(80, 80), (40, 40), (20, 20)]
t = [targets]

out = build_targets(t, 10, img_size, strides, feat_sizes)

print('\n=== Target assignment ===')
for s_idx, name in enumerate(['P3 (80x80, stride=8)',
                              'P4 (40x40, stride=16)',
                              'P5 (20x20, stride=32)']):
    ti = out[s_idx][0]
    pos_mask = ti[4] > 0.5
    n_pos = pos_mask.sum().item()
    print(f'\n{name}: {n_pos} positive cells')

    if n_pos > 0:
        positions = pos_mask.nonzero()
        for gy, gx in positions.tolist():
            cls_idx = ti[5:, gy, gx].argmax().item()
            tx = ti[0, gy, gx].item()
            ty = ti[1, gy, gx].item()
            tw = ti[2, gy, gx].item()
            th = ti[3, gy, gx].item()
            print(f'  cell (gy={gy}, gx={gx}) -> class={cls_idx}  '
                  f'tx={tx:.3f} ty={ty:.3f} tw={tw:.3f} th={th:.3f}')