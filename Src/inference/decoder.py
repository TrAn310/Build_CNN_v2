
import torch


def decode_predictions(raw_pred, stride, base_w, base_h):
    """
    Decode cho 1 scale (giữ nguyên như cũ).

    Args:
        raw_pred: [B, 5+C, H, W]
        stride:   int
        base_w, base_h: float - kich thuoc anchor co so cua scale nay

    Returns:
        boxes:       [B, H, W, 4]
        objectness:  [B, H, W]
        class_probs: [B, H, W, C]
    """
    B, ch, H, W = raw_pred.shape
    num_classes = ch - 5

    pred = raw_pred.permute(0, 2, 3, 1)  # [B, H, W, 5+C]

    tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    t_obj = pred[..., 4]
    t_cls = pred[..., 5:]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=raw_pred.device),
        torch.arange(W, device=raw_pred.device),
        indexing="ij",
    )
    grid_x = grid_x.float().unsqueeze(0)  # [1, H, W]
    grid_y = grid_y.float().unsqueeze(0)  # [1, H, W]

    cx = (torch.sigmoid(tx) + grid_x) * stride
    cy = (torch.sigmoid(ty) + grid_y) * stride

    w = torch.exp(tw) * base_w
    h = torch.exp(th) * base_h

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H, W, 4]
    objectness = torch.sigmoid(t_obj)              # [B, H, W]
    class_probs = torch.softmax(t_cls, dim=-1)      # [B, H, W, C]

    return boxes, objectness, class_probs


def decode_multi_scale(preds, strides, base_sizes):
    """
    Decode cho nhiều scale (P3, P4, P5, ...) va gop lai thanh 1 tap detection.

    Args:
        preds:      list/tuple cac raw_pred, vi du (pred_p3, pred_p4, pred_p5)
                    moi cai [B, 5+C, H_i, W_i]
        strides:    list cac stride tuong ung, vi du [8, 16, 32]
        base_sizes: list cac (base_w, base_h) tuong ung cho tung scale,
                    vi du [(32,32), (64,64), (128,128)]
                    -> anchor co so khac nhau theo scale (scale nho decode vat nho,
                       scale lon decode vat lon)

    Returns:
        boxes_all:       [B, N_total, 4]
        objectness_all:  [B, N_total]
        class_probs_all: [B, N_total, C]
        N_total = sum(H_i * W_i) tren tat ca scale
    """
    assert len(preds) == len(strides) == len(base_sizes), \
        "preds, strides, base_sizes phai cung so luong scale"

    boxes_list, obj_list, cls_list = [], [], []

    for raw_pred, stride, (base_w, base_h) in zip(preds, strides, base_sizes):
        boxes, objectness, class_probs = decode_predictions(
            raw_pred, stride=stride, base_w=base_w, base_h=base_h
        )

        B, H, W, _ = boxes.shape
        C = class_probs.shape[-1]

        # Flatten [B, H, W, ...] -> [B, H*W, ...] de co the concat giua cac scale
        # (vi H, W khac nhau giua cac scale nen khong the stack truc tiep)
        boxes_list.append(boxes.reshape(B, H * W, 4))
        obj_list.append(objectness.reshape(B, H * W))
        cls_list.append(class_probs.reshape(B, H * W, C))

    boxes_all = torch.cat(boxes_list, dim=1)        # [B, N_total, 4]
    objectness_all = torch.cat(obj_list, dim=1)      # [B, N_total]
    class_probs_all = torch.cat(cls_list, dim=1)     # [B, N_total, C]

    return boxes_all, objectness_all, class_probs_all


if __name__ == "__main__":
    # ---- TEST MULTI-SCALE DECODE ----
    B, C = 1, 3

    # Gia lap 3 raw prediction tu detector.py (giong shape that)
    pred_p3 = torch.zeros(B, 5 + C, 80, 80)  # stride 8
    pred_p4 = torch.zeros(B, 5 + C, 40, 40)  # stride 16
    pred_p5 = torch.zeros(B, 5 + C, 20, 20)  # stride 32

    # Gia lap 1 detection o P5 (giong test truoc, gy=5, gx=7)
    gy, gx = 5, 7
    pred_p5[0, 0, gy, gx] = 0.2   # tx
    pred_p5[0, 1, gy, gx] = -0.5  # ty
    pred_p5[0, 2, gy, gx] = 0.1   # tw
    pred_p5[0, 3, gy, gx] = -0.3  # th
    pred_p5[0, 4, gy, gx] = 5.0   # objectness
    pred_p5[0, 5, gy, gx] = 5.0   # class Person

    preds = (pred_p3, pred_p4, pred_p5)
    strides = [8, 16, 32]
    # Anchor co so khac nhau theo tung scale (scale nho -> vat nho, scale lon -> vat lon)
    base_sizes = [(16, 16), (32, 32), (128, 128)]

    boxes_all, objectness_all, class_probs_all = decode_multi_scale(
        preds, strides, base_sizes
    )

    print("Boxes all shape      :", boxes_all.shape)        # [1, 8400, 4]
    print("Objectness all shape :", objectness_all.shape)    # [1, 8400]
    print("Class probs all shape:", class_probs_all.shape)   # [1, 8400, 3]

    # Tinh vi tri flatten cua detection gia trong P5
    # (de kiem tra ket qua co dung khong)
    offset_p3 = 80 * 80
    offset_p4 = 40 * 40
    idx_in_p5 = gy * 20 + gx
    flat_idx = offset_p3 + offset_p4 + idx_in_p5

    print("\n--- Ket qua tai vi tri flatten cua (P5, gy=5, gx=7) ---")
    print("Box (x1,y1,x2,y2):", boxes_all[0, flat_idx].tolist())
    print("Objectness       :", objectness_all[0, flat_idx].item())
    print("Class probs      :", class_probs_all[0, flat_idx].tolist())