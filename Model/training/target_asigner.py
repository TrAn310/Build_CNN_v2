"""
target_assigner.py

Nhiệm vụ của file này: như đáp án đúng để so sánh với prediction để tính 

Ground Truth ban đầu có dạng:

    Box thật trên ảnh gốc:
    [x1, y1, x2, y2]

Ví dụ:

    Person:
    [210, 150, 270, 210]

Nhưng model của chúng ta không dự đoán trực tiếp theo dạng này.

Model dự đoán trên grid 20x20.

Vì vậy file này sẽ:

    Ground Truth
        ↓
    tìm tâm của object
        ↓
    tìm cell chứa tâm object
        ↓
    đánh dấu cell đó là positive
        ↓
    chuyển box thật sang dạng
    [tx*, ty*, tw*, th*]

Kết quả cuối cùng tạo ra 3 target:

    target_obj
    target_box
    target_cls

Sau này sẽ dùng 3 target này để so sánh
với prediction của model và tính LOSS.
"""

import torch
# Import thư viện PyTorch


def build_targets(
    gt_boxes,
    gt_classes,
    grid_h,
    grid_w,
    stride,
    base_w,
    base_h,
    num_classes
):
    """
    =========================
    INPUT
    =========================

    gt_boxes:

        Là list chứa ground-truth boxes của từng ảnh trong batch.

        Ví dụ batch có 2 ảnh:

        gt_boxes = [

            # Ảnh thứ 0 có 2 object
            tensor([
                [100, 100, 200, 200],
                [300, 200, 400, 500]
            ]),

            # Ảnh thứ 1 có 1 object
            tensor([
                [50, 50, 150, 200]
            ])
        ]

        Mỗi phần tử trong list tương ứng với 1 ảnh.

        Shape mỗi phần tử:

            [N_i, 4]

        Trong đó:

            N_i = số object trong ảnh thứ i

            4 = [x1, y1, x2, y2]


    gt_classes:

        Là list chứa class của từng object
        tương ứng với gt_boxes.

        Ví dụ:

        gt_classes = [

            tensor([0, 1]),

            tensor([2])
        ]

        Nghĩa là:

        Ảnh 0:
            Object 0 → class 0
            Object 1 → class 1

        Ảnh 1:
            Object 0 → class 2


    grid_h, grid_w:

        Kích thước grid.

        Trong model hiện tại:

            grid_h = 20
            grid_w = 20

        → Có tổng cộng:

            20 x 20 = 400 cell


    stride:

        Khoảng cách giữa feature map và ảnh gốc.

        Ví dụ:

            ảnh gốc = 640 x 640
            feature map = 20 x 20

        640 / 20 = 32

        nên:

            stride = 32


    base_w, base_h:

        Kích thước box cơ sở.

        Được dùng để encode chiều rộng và chiều cao.

        Trong phiên bản hiện tại:

            base_w = 32
            base_h = 32


    num_classes:

        Số lượng class.

        Ví dụ:

            0 = Person
            1 = Helmet
            2 = Vest

        → num_classes = 3


    =========================
    OUTPUT
    =========================

    target_obj:

        Shape:

            [B, H, W]

        Ví dụ:

            [16, 20, 20]

        Ý nghĩa:

            1 → cell positive, chịu trách nhiệm
                dự đoán object

            0 → cell negative


    target_box:

        Shape:

            [B, H, W, 4]

        Mỗi cell chứa:

            [tx*, ty*, tw*, th*]

        Tuy nhiên chỉ cell positive
        mới có box target thực sự.


    target_cls:

        Shape:

            [B, H, W]

        Chứa class index.

        Ví dụ:

            0 = Person
            1 = Helmet
            2 = Vest

        Chỉ có ý nghĩa tại positive cell.
    """


    # ============================================================
    # BƯỚC 0: XÁC ĐỊNH BATCH SIZE
    # ============================================================

    B = len(gt_boxes)

    # gt_boxes là list.
    #
    # Mỗi phần tử trong gt_boxes tương ứng với 1 ảnh.
    #
    # Ví dụ:
    #
    # gt_boxes = [
    #     tensor([...]),   # ảnh 0
    #     tensor([...]),   # ảnh 1
    #     tensor([...])    # ảnh 2
    # ]
    #
    # len(gt_boxes) = 3
    #
    # → Batch size B = 3
    #
    # LƯU Ý:
    #
    # B không phải số object.
    #
    # Một ảnh có thể có:
    #
    # 1 object
    # 2 object
    # 10 object
    #
    # nhưng vẫn chỉ tính là 1 ảnh trong batch.


    # ============================================================
    # BƯỚC 1: TẠO TARGET OBJECTNESS
    # ============================================================

    target_obj = torch.zeros(B, grid_h, grid_w)

    # Tạo tensor toàn số 0.
    #
    # Shape:
    #
    # [B, H, W]
    #
    # Ví dụ:
    #
    # [16, 20, 20]
    #
    # Nghĩa là:
    #
    # 16 ảnh
    # mỗi ảnh có grid 20x20
    #
    # Ban đầu tất cả cell đều bằng 0:
    #
    # 0 = negative
    #
    # Khi tìm được object, cell chứa tâm object
    # sẽ được đổi thành:
    #
    # 1 = positive


    # ============================================================
    # BƯỚC 2: TẠO TARGET BOX
    # ============================================================

    target_box = torch.zeros(B, grid_h, grid_w, 4)

    # Shape:
    #
    # [B, H, W, 4]
    #
    # Ví dụ:
    #
    # [16, 20, 20, 4]
    #
    # Mỗi cell có 4 giá trị:
    #
    # [tx*, ty*, tw*, th*]
    #
    # Ban đầu toàn bộ đều là 0.
    #
    # Sau đó positive cell sẽ được gán
    # giá trị box target thật.


    # ============================================================
    # BƯỚC 3: TẠO TARGET CLASS
    # ============================================================

    target_cls = torch.zeros(
        B,
        grid_h,
        grid_w,
        dtype=torch.long
    )

    # Shape:
    #
    # [B, H, W]
    #
    # Ví dụ:
    #
    # [16, 20, 20]
    #
    # dtype=torch.long
    #
    # Vì class index là số nguyên:
    #
    # 0 = Person
    # 1 = Helmet
    # 2 = Vest
    #
    # Không dùng float vì sau này
    # khi tính classification loss,
    # class target thường cần dạng số nguyên.


    # ============================================================
    # BƯỚC 4: DUYỆT QUA TỪNG ẢNH TRONG BATCH
    # ============================================================

    for b in range(B):

        # b chính là index của ảnh.
        #
        # Nếu B = 16:
        #
        # b chạy:
        #
        # 0, 1, 2, ..., 15


        # Lấy toàn bộ box của ảnh thứ b
        boxes_b = gt_boxes[b]

        # Ví dụ:
        #
        # boxes_b =
        #
        # tensor([
        #     [210, 150, 270, 210],
        #     [400, 300, 450, 350]
        # ])
        #
        # Shape:
        #
        # [N, 4]
        #
        # Trong đó N là số object
        # trong ảnh thứ b.


        # Lấy toàn bộ class tương ứng
        # với các box của ảnh thứ b
        classes_b = gt_classes[b]

        # Ví dụ:
        #
        # tensor([0, 1])
        #
        # Nghĩa là:
        #
        # box thứ 0 → Person
        # box thứ 1 → Helmet


        # ========================================================
        # BƯỚC 5: DUYỆT QUA TỪNG OBJECT TRONG ẢNH
        # ========================================================

        for n in range(boxes_b.shape[0]):

            # boxes_b.shape[0]
            #
            # = số lượng object trong ảnh.
            #
            # Ví dụ:
            #
            # boxes_b.shape = [3, 4]
            #
            # → Có 3 object.
            #
            # n sẽ chạy:
            #
            # 0, 1, 2


            # Lấy tọa độ box thứ n
            x1, y1, x2, y2 = boxes_b[n]

            # Ví dụ:
            #
            # boxes_b[n]
            #
            # = [210, 150, 270, 210]
            #
            # nên:
            #
            # x1 = 210
            # y1 = 150
            # x2 = 270
            # y2 = 210


            # Lấy class của object thứ n
            cls_id = classes_b[n]

            # Ví dụ:
            #
            # cls_id = 0
            #
            # → Person


            # ====================================================
            # BƯỚC 6: TÍNH TÂM BOX
            # ====================================================

            cx = (x1 + x2) / 2

            # Tọa độ tâm theo chiều x.
            #
            # Ví dụ:
            #
            # x1 = 210
            # x2 = 270
            #
            # cx = (210 + 270) / 2
            #
            # cx = 240


            cy = (y1 + y2) / 2

            # Tọa độ tâm theo chiều y.
            #
            # y1 = 150
            # y2 = 210
            #
            # cy = (150 + 210) / 2
            #
            # cy = 180


            # ====================================================
            # BƯỚC 7: TÍNH WIDTH VÀ HEIGHT CỦA BOX
            # ====================================================

            w = x2 - x1

            # Ví dụ:
            #
            # 270 - 210 = 60
            #
            # → chiều rộng box = 60 pixel


            h = y2 - y1

            # Ví dụ:
            #
            # 210 - 150 = 60
            #
            # → chiều cao box = 60 pixel


            # ====================================================
            # BƯỚC 8: TÌM GRID CELL CHỨA TÂM OBJECT
            # ====================================================

            gx = int(cx // stride)

            # // là phép chia lấy phần nguyên.
            #
            # Ví dụ:
            #
            # cx = 240
            # stride = 32
            #
            # 240 // 32 = 7
            #
            # → gx = 7
            #
            # Nghĩa là tâm object nằm ở
            # cột số 7 của grid.


            gy = int(cy // stride)

            # Ví dụ:
            #
            # cy = 180
            # stride = 32
            #
            # 180 // 32 = 5
            #
            # → gy = 5
            #
            # Nghĩa là tâm object nằm ở
            # hàng số 5 của grid.


            # ====================================================
            # BƯỚC 9: GIỚI HẠN gx, gy KHÔNG VƯỢT GRID
            # ====================================================

            gx = min(
                max(gx, 0),
                grid_w - 1
            )

            # Bảo đảm:
            #
            # 0 <= gx <= grid_w - 1
            #
            # Với grid_w = 20:
            #
            # gx chỉ được nằm trong:
            #
            # 0 → 19


            gy = min(
                max(gy, 0),
                grid_h - 1
            )

            # Tương tự với gy:
            #
            # 0 <= gy <= grid_h - 1


            # ====================================================
            # BƯỚC 10: ĐÁNH DẤU POSITIVE CELL
            # ====================================================

            target_obj[b, gy, gx] = 1.0

            # Ví dụ:
            #
            # b = 0
            # gy = 5
            # gx = 7
            #
            # target_obj[0, 5, 7] = 1
            #
            # Nghĩa là:
            #
            # Trong ảnh số 0,
            # cell hàng 5, cột 7
            # chịu trách nhiệm dự đoán object này.


            # ====================================================
            # BƯỚC 11: ENCODE TÂM BOX
            # ====================================================

            # Model decode bằng công thức:
            #
            # cx =
            #
            # (sigmoid(tx) + gx) * stride
            #
            # Bây giờ ta đang có cx thật.
            #
            # Ta cần tìm tx* để khi đưa vào
            # công thức trên thì lấy lại được cx.


            offset_x = cx / stride - gx

            # Ví dụ:
            #
            # cx = 240
            # stride = 32
            # gx = 7
            #
            # offset_x =
            #
            # 240 / 32 - 7
            #
            # = 7.5 - 7
            #
            # = 0.5
            #
            # Nghĩa là:
            #
            # tâm object nằm ở vị trí 50%
            # theo chiều ngang bên trong cell.


            offset_y = cy / stride - gy

            # Ví dụ:
            #
            # cy = 180
            # stride = 32
            # gy = 5
            #
            # offset_y =
            #
            # 180 / 32 - 5
            #
            # = 5.625 - 5
            #
            # = 0.625
            #
            # Nghĩa là:
            #
            # tâm object nằm ở 62.5%
            # theo chiều dọc trong cell.


            # ====================================================
            # BƯỚC 12: TRÁNH offset = 0 HOẶC = 1
            # ====================================================

            eps = 1e-6

            # eps = 0.000001
            #
            # Một số rất nhỏ.


            offset_x = offset_x.clamp(
                eps,
                1 - eps
            )

            offset_y = offset_y.clamp(
                eps,
                1 - eps
            )

            # clamp có nghĩa là giới hạn giá trị.
            #
            # Ta cần:
            #
            # eps < offset < 1 - eps
            #
            # Vì bước tiếp theo có phép:
            #
            # offset / (1 - offset)
            #
            # Nếu offset = 1:
            #
            # 1 - offset = 0
            #
            # → chia cho 0 → lỗi.
            #
            # Nếu offset = 0 hoặc 1,
            # logit cũng tiến tới vô cực.
            #
            # Vì vậy clamp để an toàn.


            # ====================================================
            # BƯỚC 13: INVERSE SIGMOID / LOGIT
            # ====================================================

            tx_star = torch.log(
                offset_x / (1 - offset_x)
            )

            # Vì:
            #
            # sigmoid(tx*) = offset_x
            #
            # Ta muốn tìm ngược tx*.
            #
            # Công thức nghịch đảo sigmoid là:
            #
            # logit(p) = ln(p / (1-p))
            #
            # Ví dụ:
            #
            # offset_x = 0.5
            #
            # tx* = ln(0.5 / 0.5)
            #
            # = ln(1)
            #
            # = 0
            #
            # Sau này:
            #
            # sigmoid(0) = 0.5
            #
            # → quay lại đúng offset ban đầu.


            ty_star = torch.log(
                offset_y / (1 - offset_y)
            )

            # Hoàn toàn tương tự tx_star
            # nhưng theo chiều y.


            # ====================================================
            # BƯỚC 14: ENCODE WIDTH
            # ====================================================

            tw_star = torch.log(
                w / base_w
            )

            # Decoder của chúng ta có:
            #
            # w = exp(tw) * base_w
            #
            # Bây giờ ta có w thật.
            #
            # Muốn tìm tw*:
            #
            # tw* = ln(w / base_w)
            #
            # Ví dụ:
            #
            # w = 60
            # base_w = 32
            #
            # tw* = ln(60 / 32)
            #
            # ≈ 0.6286


            # ====================================================
            # BƯỚC 15: ENCODE HEIGHT
            # ====================================================

            th_star = torch.log(
                h / base_h
            )

            # Hoàn toàn tương tự width.
            #
            # Decoder:
            #
            # h = exp(th) * base_h
            #
            # Encode ngược:
            #
            # th* = ln(h / base_h)


            # ====================================================
            # BƯỚC 16: GÁN BOX TARGET VÀO POSITIVE CELL
            # ====================================================

            target_box[b, gy, gx, 0] = tx_star

            # Lưu target tx*
            # vào cell positive.


            target_box[b, gy, gx, 1] = ty_star

            # Lưu target ty*.


            target_box[b, gy, gx, 2] = tw_star

            # Lưu target tw*.


            target_box[b, gy, gx, 3] = th_star

            # Lưu target th*.


            # Sau bước này:
            #
            # target_box[b, gy, gx]
            #
            # =
            #
            # [tx*, ty*, tw*, th*]


            # ====================================================
            # BƯỚC 17: GÁN CLASS TARGET
            # ====================================================

            target_cls[b, gy, gx] = cls_id

            # Ví dụ:
            #
            # cls_id = 0
            #
            # target_cls[0, 5, 7] = 0
            #
            # Nghĩa là:
            #
            # cell (5,7)
            #
            # phải dự đoán:
            #
            # Person


    # ============================================================
    # HOÀN THÀNH TẤT CẢ OBJECT TRONG BATCH
    # ============================================================

    return target_obj, target_box, target_cls


# =================================================================
# PHẦN BÊN DƯỚI CHỈ CHẠY KHI TA CHẠY TRỰC TIẾP FILE NÀY
# =================================================================

if __name__ == "__main__":

    # Nếu chạy:
    #
    # python target_assigner.py
    #
    # thì đoạn code này chạy.
    #
    # Nếu file này được import từ file khác:
    #
    # from target_assigner import build_targets
    #
    # thì đoạn test bên dưới KHÔNG chạy.


    # ============================================================
    # CẤU HÌNH TEST
    # ============================================================

    stride = 32

    # 640 / 20 = 32
    #
    # Mỗi bước grid cách nhau 32 pixel
    # trên hệ tọa độ ảnh gốc.


    base_w = base_h = 32

    # Box cơ sở:
    #
    # width cơ sở = 32
    # height cơ sở = 32


    grid_h = grid_w = 20

    # Feature map:
    #
    # 20 x 20
    #
    # = 400 cell


    num_classes = 3

    # 0 = Person
    # 1 = Helmet
    # 2 = Vest


    # ============================================================
    # TẠO 1 BATCH GIẢ
    # ============================================================

    gt_boxes = [
        torch.tensor([
            [210.0, 150.0, 270.0, 210.0]
        ])
    ]

    # Dấu [ ... ] bên ngoài nghĩa là:
    #
    # đây là LIST chứa dữ liệu của từng ảnh.
    #
    # Hiện tại chỉ có 1 phần tử.
    #
    # → batch size = 1
    #
    # Tensor bên trong:
    #
    # [[210, 150, 270, 210]]
    #
    # → ảnh này có 1 object.


    gt_classes = [
        torch.tensor([0])
    ]

    # Object duy nhất có class:
    #
    # 0 = Person


    # ============================================================
    # GỌI HÀM BUILD TARGET
    # ============================================================

    target_obj, target_box, target_cls = build_targets(

        gt_boxes,
        gt_classes,
        grid_h,
        grid_w,
        stride,
        base_w,
        base_h,
        num_classes
    )

    # Sau dòng này:
    #
    # target_obj:
    #
    # [1, 20, 20]
    #
    # target_box:
    #
    # [1, 20, 20, 4]
    #
    # target_cls:
    #
    # [1, 20, 20]


    # ============================================================
    # IN SHAPE CỦA CÁC TARGET
    # ============================================================

    print(
        "target_obj shape:",
        target_obj.shape
    )

    print(
        "target_box shape:",
        target_box.shape
    )

    print(
        "target_cls shape:",
        target_cls.shape
    )


    # ============================================================
    # ĐẾM SỐ POSITIVE CELL
    # ============================================================

    num_positive = target_obj.sum().item()

    # target_obj chỉ chứa:
    #
    # 0 hoặc 1
    #
    # Ví dụ:
    #
    # 0 0 0
    # 0 1 0
    # 0 0 0
    #
    # Tổng = 1
    #
    # → Có 1 positive cell.


    print(
        "\nSo o positive:",
        num_positive
    )


    # ============================================================
    # TÌM VỊ TRÍ POSITIVE CELL
    # ============================================================

    pos_idx = (
        target_obj[0] == 1
    ).nonzero()

    # target_obj[0]
    #
    # Lấy grid của ảnh đầu tiên.
    #
    # Shape:
    #
    # [20,20]
    #
    # target_obj[0] == 1
    #
    # Tạo mask:
    #
    # True tại positive cell.
    #
    # .nonzero()
    #
    # Trả về tọa độ những vị trí khác 0.
    #
    # Ví dụ:
    #
    # tensor([
    #     [5, 7]
    # ])
    #
    # Nghĩa là:
    #
    # gy = 5
    # gx = 7


    print(
        "Vi tri o positive (gy, gx):",
        pos_idx.tolist()
    )


    # ============================================================
    # LẤY TỌA ĐỘ POSITIVE CELL
    # ============================================================

    gy, gx = pos_idx[0].tolist()

    # pos_idx[0]
    #
    # = [5, 7]
    #
    # Sau đó:
    #
    # gy = 5
    # gx = 7


    print(
        f"\nTai o (gy={gy}, gx={gx}):"
    )


    # ============================================================
    # IN BOX TARGET TẠI POSITIVE CELL
    # ============================================================

    print(
        "  target_box:",
        target_box[0, gy, gx].tolist()
    )

    # Ví dụ kết quả:
    #
    # [0.0, 0.5108, 0.6286, 0.6286]
    #
    # Đây chính là:
    #
    # [tx*, ty*, tw*, th*]


    # ============================================================
    # IN CLASS TẠI POSITIVE CELL
    # ============================================================

    print(
        "  target_cls:",
        target_cls[0, gy, gx].item()
    )

    # Kết quả:
    #
    # 0
    #
    # → Person


    print(
        "\nExpected: o (gy=5, gx=7), class=0 (Person)"
    )