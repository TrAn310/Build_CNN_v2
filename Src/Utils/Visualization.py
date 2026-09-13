"""
visualization.py
 
Ve bounding box + nhan (class + confidence) len anh, de KIEM TRA BANG
MAT prediction cua model co dung khong (PHAN 20 trong Prompt).
 
Theo dung yeu cau: mau sac KHONG QUAN TRONG. Quan trong la:
    - box dung vi tri
    - class dung
    - confidence dung
"""
 
import cv2
 
 
def draw_boxes(image, boxes, scores, classes, class_names,
               box_color=(0, 255, 0), thickness=2):
    """
    Args:
        image:   numpy array [H,W,3], uint8. Dung dung he mau ma anh
                 dang o (RGB hay BGR deu duoc, ham nay KHONG doi mau,
                 chi ve len tren).
        boxes:   [N,4]  (x1,y1,x2,y2) toa do PIXEL tren CHINH anh nay
                 (KHONG normalize) -- dung dinh dang postprocess.py tra ve.
        scores:  [N]    confidence, 0..1
        classes: [N]    class index (int)
        class_names: list[str], vd ['Person','Helmet','Vest'] -- lay tu
                     config_utils.load_dataset_config().
        box_color: mau (B,G,R) cho box va nen chu -- mac dinh xanh la,
                   nhung mau khong quan trong (theo dung Prompt).
        thickness: do day duong vien box.
 
    Returns:
        numpy array [H,W,3] -- ANH MOI da ve xong, KHONG sua anh goc
        (lam viec tren 1 ban .copy()).
    """
    out = image.copy()
 
    # Ho tro ca tensor (torch) lan list/numpy -- nguoi goi khong can tu
    # convert tay truoc khi truyen vao.
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if hasattr(classes, "tolist"):
        classes = classes.tolist()
 
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cls = int(cls)
        name = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
 
        # ---- Box ----
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
 
        # ---- Nhan: "Person 0.91" ----
        label = f"{name} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
 
        # Uu tien dat nhan NGAY TREN box; neu box sat mep tren anh
        # (khong du cho) thi dat XUONG DUOI canh tren cua box.
        label_top = y1 - text_h - baseline - 4
        if label_top < 0:
            label_top = y1
 
        # Nen mau dac sau lung chu de de doc tren moi nen anh.
        cv2.rectangle(
            out,
            (x1, label_top),
            (x1 + text_w + 4, label_top + text_h + baseline + 4),
            box_color,
            -1,
        )
        cv2.putText(
            out, label,
            (x1 + 2, label_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA,
        )
 
    return out
 
 
if __name__ == "__main__":
    # ---- TEST: ve tren 1 anh gia, dung DUNG dinh dang output cua
    # postprocess.py (boxes/scores/classes) + class_names cua PPE dataset ----
    import numpy as np
 
    image = np.full((400, 500, 3), 255, dtype=np.uint8)  # nen trang de de nhin
 
    class_names = ["Person", "Helmet", "Vest"]
 
    boxes = [
        [50, 50, 250, 350],    # Person, box lon
        [100, 60, 160, 110],   # Helmet, nam tren dau Person
        [10, 10, 90, 40],      # Vest, sat mep tren anh (test truong hop label bi lech)
    ]
    scores = [0.91, 0.87, 0.83]
    classes = [0, 1, 2]
 
    out = draw_boxes(image, boxes, scores, classes, class_names)
 
    cv2.imwrite("test_draw_boxes.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("Da luu test_draw_boxes.png -- mo anh de kiem tra bang mat:")
    print("  - 3 box dung vi tri nhu toa do khai bao")
    print("  - Nhan hien dung 'Person 0.91' / 'Helmet 0.87' / 'Vest 0.83'")
    print("  - Box Vest sat mep tren -> nhan phai tu dong lat xuong duoi box")
    print("    (khong bi cat mat ngoai bien anh)")
 