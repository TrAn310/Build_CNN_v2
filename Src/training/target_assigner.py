"""
target_assigner.py
Gán GT box vào vùng 3x3 cells quanh tâm object cho mỗi scale.
"""

import torch


def build_targets(targets, num_classes, img_size, strides, feat_sizes):
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

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = gy + dy
                        xx = gx + dx
                        if yy < 0 or yy >= H or xx < 0 or xx >= W:
                            continue

                        tx = xc / stride - xx
                        ty = yc / stride - yy
                        tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
                        th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

                        t = all_targets[s_idx][b]
                        if t[4, yy, xx] < 0.5:
                            t[0, yy, xx] = tx
                            t[1, yy, xx] = ty
                            t[2, yy, xx] = tw
                            t[3, yy, xx] = th
                            t[4, yy, xx] = 1.0
                            t[5 + cls, yy, xx] = 1.0

    return all_targets