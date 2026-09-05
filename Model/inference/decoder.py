
import torch

def decode_predictions(raw_pred, stride, base_w, base_h):
    """
    Args:
        raw_pred: tensor [B, 5+C, H, W] -- output tho tu Detection Head
        stride:   bước nhảy, độ co
        base_w:   float, kich thuoc co so cua box theo chieu rong
        base_h:   float, kich thuoc co so cua box theo chieu cao

    Returns:
        boxes:       [B, H, W, 4]  (x1, y1, x2, y2) tren anh goc
        objectness:  [B, H, W]     da qua sigmoid, trong khoang (0,1)
        class_probs: [B, H, W, C]  da qua softmax
    """
    B, ch, H, W = raw_pred.shape
    num_classes = ch - 5

    # Dua channel ve cuoi de de thao tac: [B, H, W, 5+C], đúng form của pytorch 
    pred = raw_pred.permute(0, 2, 3, 1)  # [B, H, W, 5+C]

    tx = pred[..., 0]
    ty = pred[..., 1]
    tw = pred[..., 2]
    th = pred[..., 3]
    t_obj = pred[..., 4]
    t_cls = pred[..., 5:]  # [B, H, W, C]

    # ---- Tao luoi toa do grid (gx, gy) cho tung o ----
    # gy: chi so hang (0..H-1), gx: chi so cot (0..W-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=raw_pred.device),
        torch.arange(W, device=raw_pred.device),
        indexing="ij",
    )
    grid_x = grid_x.float()  # [H, W]
    grid_y = grid_y.float()  # [H, W]

    # them chieu batch de broadcast: [1, H, W]
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    # ---- Decode tam box (cx, cy) ----
    cx = (torch.sigmoid(tx) + grid_x) * stride  # [B, H, W]
    cy = (torch.sigmoid(ty) + grid_y) * stride  # [B, H, W]

    # ---- Decode kich thuoc box (w, h) ----
    w = torch.exp(tw) * base_w  # [B, H, W]
    h = torch.exp(th) * base_h  # [B, H, W]

    # ---- Tu (cx,cy,w,h) sang (x1,y1,x2,y2) ----
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H, W, 4]

    # ---- Objectness va class probability ----
    objectness = torch.sigmoid(t_obj)              # [B, H, W]
    class_probs = torch.softmax(t_cls, dim=-1)      # [B, H, W, C]

    return boxes, objectness, class_probs


if __name__ == "__main__":
    # ---- TEST DECODER BANG PREDICTION GIA (chua can model that) ----
    B, C, H, W = 1, 3, 20, 20
    stride = 32
    base_w = base_h = 32

    # Gia lap raw output tu Detection Head: [B, 5+C, H, W]
    raw_pred = torch.zeros(B, 5 + C, H, W)

    # Dat gia tri gia cho 1 o cu the: gy=5, gx=7 (giong Bai 2)
    gy, gx = 5, 7
    raw_pred[0, 0, gy, gx] = 0.2   # tx
    raw_pred[0, 1, gy, gx] = -0.5  # ty
    raw_pred[0, 2, gy, gx] = 0.1   # tw
    raw_pred[0, 3, gy, gx] = -0.3  # th
    raw_pred[0, 4, gy, gx] = 5.0   # objectness (raw, truoc sigmoid -> gan 1)
    raw_pred[0, 5, gy, gx] = 5.0   # class Person score cao

    boxes, objectness, class_probs = decode_predictions(
        raw_pred, stride=stride, base_w=base_w, base_h=base_h
    )

    print("Boxes shape      :", boxes.shape)
    print("Objectness shape :", objectness.shape)
    print("Class probs shape:", class_probs.shape)

    print("\n--- Ket qua tai o (gy=5, gx=7) ---")
    print("Box (x1,y1,x2,y2):", boxes[0, gy, gx].tolist())
    print("Objectness       :", objectness[0, gy, gx].item())
    print("Class probs      :", class_probs[0, gy, gx].tolist())

    print("\n--- Doi chieu voi tinh tay o Bai 3 ---")
    print("Expected box xap xi: [223.9, 160.4, 259.3, 184.1]")