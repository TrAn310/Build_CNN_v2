"""
target_assigner.py (multi-scale)
 
Output: 3 bộ target riêng, mỗi bộ cho 1 scale, format giống hệt bản cũ
(target_obj, target_box, target_cls) - để decoder.py, losses.py dùng lại
NGUYÊN VẸN không cần sửa gì.
"""
 
import torch
 
 
# Cấu hình cố định của model - PHẢI khớp với neck.py / detector.py
# (grid_h, grid_w, stride) theo đúng thứ tự P3, P4, P5
SCALE_CONFIGS = [
    {"name": "P3", "grid_h": 80, "grid_w": 80, "stride": 8},
    {"name": "P4", "grid_h": 40, "grid_w": 40, "stride": 16},
    {"name": "P5", "grid_h": 20, "grid_w": 20, "stride": 32},
]
 
# Ngưỡng kích thước để quyết định object thuộc scale nào
# object nhỏ hơn 64px -> P3, 64-128px -> P4, >=128px -> P5
SIZE_THRESHOLDS = [64, 128]
 
 
def assign_scale_index(box_w, box_h):
    """
    Input:
        box_w, box_h: kích thước box (pixel, trên ảnh gốc 640x640)
 
    Output:
        scale_idx: 0 (P3), 1 (P4), hoặc 2 (P5)
    """
    max_side = max(box_w, box_h)
 
    if max_side < SIZE_THRESHOLDS[0]:
        return 0  # P3 - object nhỏ
    elif max_side < SIZE_THRESHOLDS[1]:
        return 1  # P4 - object vừa
    else:
        return 2  # P5 - object lớn
 
 
def _encode_single_scale(cx, cy, w, h, stride):
    """
    Encode 1 box thành (tx*, ty*, tw*, th*) cho 1 scale cụ thể.
    Logic giống hệt bản single-scale cũ, chỉ tách ra thành hàm riêng
    để dùng lại cho cả 3 scale mà không lặp code.
    """
    base_w = base_h = stride  # base = chính stride của scale đó
 
    gx = int(cx // stride)
    gy = int(cy // stride)
 
    offset_x = cx / stride - gx
    offset_y = cy / stride - gy
 
    eps = 1e-6
    offset_x = offset_x.clamp(eps, 1 - eps)
    offset_y = offset_y.clamp(eps, 1 - eps)
 
    tx_star = torch.log(offset_x / (1 - offset_x))
    ty_star = torch.log(offset_y / (1 - offset_y))
    tw_star = torch.log(w / base_w)
    th_star = torch.log(h / base_h)
 
    return gx, gy, tx_star, ty_star, tw_star, th_star
 
 
def build_targets_multiscale(gt_boxes, gt_classes, num_classes):
    """
    Input:
        gt_boxes:   list[Tensor[N_i,4]]  (x1,y1,x2,y2) trên ảnh gốc 640x640
        gt_classes: list[Tensor[N_i]]
        num_classes: int (lấy động từ data.yaml)
 
    Output:
        targets: list gồm 3 phần tử, mỗi phần tử là tuple
                 (target_obj, target_box, target_cls) cho 1 scale,
                 theo đúng thứ tự [P3, P4, P5]
    """
    B = len(gt_boxes)
 
    # Tạo sẵn target rỗng cho cả 3 scale
    targets = []
    for cfg in SCALE_CONFIGS:
        gh, gw = cfg["grid_h"], cfg["grid_w"]
        target_obj = torch.zeros(B, gh, gw)
        target_box = torch.zeros(B, gh, gw, 4)
        target_cls = torch.zeros(B, gh, gw, dtype=torch.long)
        targets.append([target_obj, target_box, target_cls])
 
    for b in range(B):
        boxes_b = gt_boxes[b]
        classes_b = gt_classes[b]
 
        for n in range(boxes_b.shape[0]):
            x1, y1, x2, y2 = boxes_b[n]
            cls_id = classes_b[n]
 
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
 
            # ---- BƯỚC MỚI: xác định object này thuộc scale nào ----
            scale_idx = assign_scale_index(w.item(), h.item())
            cfg = SCALE_CONFIGS[scale_idx]
            gh, gw_, stride = cfg["grid_h"], cfg["grid_w"], cfg["stride"]
 
            gx, gy, tx_star, ty_star, tw_star, th_star = _encode_single_scale(
                cx, cy, w, h, stride
            )
 
            gx = min(max(gx, 0), gw_ - 1)
            gy = min(max(gy, 0), gh - 1)
 
            # CHỈ ghi vào target của đúng scale_idx, 2 scale kia không đổi
            target_obj, target_box, target_cls = targets[scale_idx]
            target_obj[b, gy, gx] = 1.0
            target_box[b, gy, gx, 0] = tx_star
            target_box[b, gy, gx, 1] = ty_star
            target_box[b, gy, gx, 2] = tw_star
            target_box[b, gy, gx, 3] = th_star
            target_cls[b, gy, gx] = cls_id
 
    # Trả về dạng tuple cho mỗi scale (giữ nguyên interface như bản cũ)
    return [tuple(t) for t in targets]
 
 
if __name__ == "__main__":
    # ---- TEST với 3 object kích thước KHÁC NHAU để rơi vào 3 scale khác nhau ----
    gt_boxes = [
        torch.tensor([
            [100.0, 100.0, 130.0, 130.0],   # w=h=30  -> nhỏ -> P3
            [300.0, 200.0, 380.0, 280.0],   # w=h=80  -> vừa -> P4
            [50.0, 50.0, 250.0, 250.0],     # w=h=200 -> lớn -> P5
        ])
    ]
    gt_classes = [torch.tensor([1, 0, 0])]  # Helmet, Person, Person
 
    targets = build_targets_multiscale(gt_boxes, gt_classes, num_classes=3)
 
    scale_names = ["P3", "P4", "P5"]
    for i, (t_obj, t_box, t_cls) in enumerate(targets):
        num_pos = t_obj.sum().item()
        print(f"{scale_names[i]}: shape={tuple(t_obj.shape)}  so_o_positive={num_pos}")
        if num_pos > 0:
            pos_idx = (t_obj[0] == 1).nonzero()
            for gy, gx in pos_idx.tolist():
                print(f"    -> cell (gy={gy}, gx={gx}), box={t_box[0, gy, gx].tolist()}, "
                      f"class={t_cls[0, gy, gx].item()}")
 
    print("\nExpected: P3 co 1 o positive, P4 co 1 o positive, P5 co 1 o positive")