"""
target_assigner.py
Gán GT box vào grid cell, encode thành (tx, ty, tw, th, obj, cls).

Chọn scale theo kích thước object: object nhỏ -> P3, lớn -> P5.
"""

import torch


def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    """
    targets: list tensor [N_i, 5] = [class, xc, yc, w, h] normalized
    return: list target tensors [B, 5+C, H, W]
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

            # Chọn scale phù hợp
            obj_size = (w * h) ** 0.5
            best_s = 0
            best_diff = 1e9
            for si, st in enumerate(strides):
                diff = abs(st - obj_size)
                if diff < best_diff:
                    best_diff = diff
                    best_s = si

            stride = strides[best_s]
            H, W = feat_sizes[best_s]
            gx = max(0, min(W - 1, int(xc / stride)))
            gy = max(0, min(H - 1, int(yc / stride)))

            tx = xc / stride - gx
            ty = yc / stride - gy
            tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
            th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

            t = all_targets[best_s][b]
            t[0, gy, gx] = tx
            t[1, gy, gx] = ty
            t[2, gy, gx] = tw
            t[3, gy, gx] = th
            t[4, gy, gx] = 1.0
            t[5 + cls, gy, gx] = 1.0

    return all_targets
