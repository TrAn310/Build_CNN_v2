"""
decoder.py
Decode raw prediction -> boxes pixel.

Công thức:
    cx = (sigmoid(tx) + grid_x) * stride
    cy = (sigmoid(ty) + grid_y) * stride
    w  = exp(tw) * stride
    h  = exp(th) * stride
    x1 = cx - w/2, ...
"""

import torch


def decode_predictions(raw_outputs, strides, num_classes,
                       img_size=640, conf_thresh=0.3):
    device = raw_outputs[0].device
    B = raw_outputs[0].shape[0]
    results = [[] for _ in range(B)]

    for out, stride in zip(raw_outputs, strides):
        B_, C_, H, W = out.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        grid_x = xs.float().view(1, 1, H, W)
        grid_y = ys.float().view(1, 1, H, W)

        tx = torch.sigmoid(out[:, 0:1])
        ty = torch.sigmoid(out[:, 1:2])
        tw = out[:, 2:3]
        th = out[:, 3:4]
        obj = torch.sigmoid(out[:, 4:5])
        cls_prob = torch.softmax(out[:, 5:], dim=1)

        cx = (tx + grid_x) * stride
        cy = (ty + grid_y) * stride
        w = torch.exp(tw.clamp(-4, 4)) * stride
        h = torch.exp(th.clamp(-4, 4)) * stride

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        max_cls_prob, cls_idx = cls_prob.max(dim=1, keepdim=True)
        score = obj * max_cls_prob

        for b in range(B):
            s = score[b, 0]
            mask = s > conf_thresh
            if mask.sum() == 0:
                continue
            bx1 = x1[b, 0][mask]
            by1 = y1[b, 0][mask]
            bx2 = x2[b, 0][mask]
            by2 = y2[b, 0][mask]
            sc = s[mask]
            ci = cls_idx[b, 0][mask]
            det = torch.stack([bx1, by1, bx2, by2, sc, ci.float()], dim=1)
            results[b].append(det)

    final = []
    for b in range(B):
        if len(results[b]) == 0:
            final.append(torch.zeros(0, 6, device=device))
        else:
            final.append(torch.cat(results[b], dim=0))
    return final
