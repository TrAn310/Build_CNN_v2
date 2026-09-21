"""
decoder.py
Decode raw output của model thành box [x1, y1, x2, y2, score, class].
Đã fix: clamp tọa độ, lọc box invalid, lọc objectness thấp.
"""

import torch


def decode_predictions(raw_outputs, strides, num_classes,
                       img_size=640, conf_thresh=0.3,
                       obj_thresh=0.5):
    """
    raw_outputs: list [B, 5+C, H, W] cho mỗi scale
    strides: [8, 16, 32]
    obj_thresh: ngưỡng objectness tối thiểu để giữ box
    return: list [B] mỗi phần tử là [N, 6] = [x1, y1, x2, y2, score, class]
    """
    device = raw_outputs[0].device
    B = raw_outputs[0].shape[0]

    all_dets = [[] for _ in range(B)]

    for s_idx, (pred, stride) in enumerate(zip(raw_outputs, strides)):
        _, _, H, W = pred.shape

        # ---- Grid cell coordinates ----
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        # ---- Tách thành phần ----
        p_box = pred[:, :4]          # [B, 4, H, W]
        p_obj = pred[:, 4]           # [B, H, W]
        p_cls = pred[:, 5:]          # [B, C, H, W]

        # ---- Decode box ----
        xc = (grid_x.unsqueeze(0) + p_box[:, 0]) * stride
        yc = (grid_y.unsqueeze(0) + p_box[:, 1]) * stride
        w = torch.exp(p_box[:, 2].clamp(-10, 10)) * stride
        h = torch.exp(p_box[:, 3].clamp(-10, 10)) * stride

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        # ---- Score ----
        obj_score = torch.sigmoid(p_obj)              # [B, H, W]
        cls_score = torch.softmax(p_cls, dim=1)       # [B, C, H, W]
        cls_max, cls_idx = cls_score.max(dim=1)       # [B, H, W]
        score = obj_score * cls_max                    # [B, H, W]

        # ---- Lọc theo confidence VÀ objectness ----
        for b in range(B):
            mask = (score[b] > conf_thresh) & (obj_score[b] > obj_thresh)
            if mask.sum() == 0:
                continue

            bx1 = x1[b][mask]
            by1 = y1[b][mask]
            bx2 = x2[b][mask]
            by2 = y2[b][mask]
            bscore = score[b][mask]
            bcls = cls_idx[b][mask].float()

            # Clamp box về [0, img_size]
            bx1 = bx1.clamp(0, img_size)
            by1 = by1.clamp(0, img_size)
            bx2 = bx2.clamp(0, img_size)
            by2 = by2.clamp(0, img_size)

            # Lọc box invalid
            valid = (bx2 > bx1) & (by2 > by1)
            if valid.sum() == 0:
                continue

            dets = torch.stack([
                bx1[valid], by1[valid], bx2[valid], by2[valid],
                bscore[valid], bcls[valid]
            ], dim=1)

            all_dets[b].append(dets)

    results = []
    for b in range(B):
        if len(all_dets[b]) == 0:
            results.append(torch.zeros(0, 6, device=device))
        else:
            results.append(torch.cat(all_dets[b], dim=0))

    return results