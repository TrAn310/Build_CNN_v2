"""
target_assigner.py
Gán GT box vào vùng 3x3 cells quanh tâm object cho mỗi scale.
"""
import torch


def build_targets(targets, num_classes, img_size, strides, feat_sizes):
    device = targets[0].device if len(targets) > 0 else 'cpu'
    B = len(targets)

    # Scale ranges (pixel)
    scale_ranges = [(0, 96), (48, 160), (96, 99999)]

    all_targets = []
    for stride, (H, W) in zip(strides, feat_sizes):
        t = torch.zeros(B, 5 + num_classes, H, W, device=device)
        all_targets.append(t)

    for b in range(B):
        if targets[b] is None or targets[b].shape[0] == 0:
            continue
        for row in targets[b]:
            cls = int(row[0].item())
            xc, yc, w, h = (row[1:] * img_size).tolist()
            obj_size = max(w, h)

            for s_idx, stride in enumerate(strides):
                lo, hi = scale_ranges[s_idx]
                if not (lo <= obj_size < hi):
                    continue

                H, W = feat_sizes[s_idx]
                gx = max(0, min(W - 1, int(xc / stride)))
                gy = max(0, min(H - 1, int(yc / stride)))

                t = all_targets[s_idx][b]

                # ---- GÁN 3x3 NEIGHBORHOOD ----
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = gy + dy, gx + dx
                        if not (0 <= ny < H and 0 <= nx < W):
                            continue
                        if t[4, ny, nx] > 0.5:
                            continue  # đã có object

                        # Offset trong cell (ny, nx)
                        tx = xc / stride - nx
                        ty = yc / stride - ny
                        tw = torch.log(torch.tensor(max(w / stride, 1e-3), device=device))
                        th = torch.log(torch.tensor(max(h / stride, 1e-3), device=device))

                        t[0, ny, nx] = tx
                        t[1, ny, nx] = ty
                        t[2, ny, nx] = tw
                        t[3, ny, nx] = th
                        t[4, ny, nx] = 1.0    # ← target = 1.0
                        t[5 + cls, ny, nx] = 1.0

    return all_targets
