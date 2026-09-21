"""
target_assigner.py
Gán GT box vào vùng 3x3 cells quanh tâm object cho mỗi scale.
"""

import torch


def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    """
    Gán object vào scale phù hợp theo kích thước (YOLOv5 style).
    """
    device = targets[0].device if len(targets) > 0 else 'cpu'
    B = len(targets)
    
    # Ngưỡng kích thước cho từng scale (pixel)
    # P3/8 → object 0-64px, P4/16 → 64-128px, P5/32 → 128+px
    scale_ranges = [(0, 96), (48, 160), (96, 99999)]

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
            obj_size = max(w, h)

            for s_idx, stride in enumerate(strides):
                lo, hi = scale_ranges[s_idx]
                if not (lo <= obj_size < hi):
                    continue   # ← CHỈ gán vào scale phù hợp
                
                H, W = feat_sizes[s_idx]
                gx = max(0, min(W - 1, int(xc / stride)))
                gy = max(0, min(H - 1, int(yc / stride)))

                # Giảm từ 3x3 xuống 1 cell (chuẩn YOLOv5)
                tx = xc / stride - gx
                ty = yc / stride - gy
                tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
                th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

                t = all_targets[s_idx][b]
                if t[4, gy, gx] < 0.5:
                    t[0, gy, gx] = tx
                    t[1, gy, gx] = ty
                    t[2, gy, gx] = tw
                    t[3, gy, gx] = th
                    t[4, gy, gx] = 0.9
                    t[5 + cls, gy, gx] = 1.0

    return all_targets