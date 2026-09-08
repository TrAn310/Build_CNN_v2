"""
losses.py

File này tính TOTAL LOSS cho detector, gồm 3 thành phần cộng lại:
  - box_loss  : đo sai số vị trí/kích thước box   -> chỉ tính tại ô CÓ object
  - obj_loss  : đo model có nhận ra "đây là object" không -> tính trên TOÀN BỘ lưới
  - cls_loss  : đo model đoán đúng class (person/helmet/vest) không -> chỉ tính tại ô CÓ object

Lý do tách 3 loss riêng: mỗi cái đo một loại sai số khác nhau về bản chất toán học
(box là hồi quy số thực, objectness là nhị phân có/không, class là phân loại đa lớp),
nên không thể gộp chung 1 công thức.
"""

import torch
import torch.nn.functional as F


def detection_loss(raw_pred, obj_mask, box_target, cls_target,
                    lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
    """
    Tham số đầu vào:
    ----------------
    raw_pred:   [B, 5+C, H, W]
                Output THÔ (chưa decode) đi thẳng ra từ Detection Head.
                5 kênh đầu = tx, ty, tw, th, objectness (chưa qua sigmoid/exp).
                C kênh sau = logits class (chưa qua softmax).

    obj_mask:   [B, H, W]
                Bản đồ đánh dấu ô nào có object, do target_assigner.py tạo ra.
                Giá trị 1 = ô này CÓ object (positive), 0 = ô này KHÔNG có (negative).

    box_target: [B, 4, H, W]
                Giá trị box đã encode (tx,ty,tw,th) tại các ô positive.
                Tại ô negative, giá trị ở đây là rác (không được dùng tới,
                vì ta sẽ lọc bằng obj_mask trước khi tính loss).

    cls_target: [B, H, W]  (kiểu long/int)
                Class index đúng (0=person, 1=helmet, 2=vest...) tại ô positive.
                Tại ô negative, giá trị cũng là rác, không dùng tới.

    lambda_box/obj/cls:
                Hệ số trọng số cho từng loss, để cân bằng độ lớn giữa 3 loss
                (nếu không có hệ số, loss nào có giá trị tự nhiên lớn hơn sẽ
                "lấn át" gradient của loss còn lại).

    Trả về:
    -------
    total, box_loss, obj_loss, cls_loss (để log riêng từng thành phần khi train)
    """

    # ---------------------------------------------------------
    # BƯỚC 1: TÁCH raw_pred THÀNH 3 NHÓM Ý NGHĨA RIÊNG
    # ---------------------------------------------------------
    # raw_pred có 5+C kênh gộp chung, ta cắt theo chiều channel (dim=1)
    # để lấy ra đúng phần nào là box, phần nào là objectness, phần nào là class.
    pred_box = raw_pred[:, 0:4, :, :]   # 4 kênh đầu: tx,ty,tw,th (còn thô, CHƯA decode)
    pred_obj = raw_pred[:, 4, :, :]     # kênh thứ 5: objectness logit (1 số / ô)
    pred_cls = raw_pred[:, 5:, :, :]    # các kênh còn lại: class logits (C số / ô)

    # ---------------------------------------------------------
    # BƯỚC 2: XÁC ĐỊNH Ô NÀO LÀ POSITIVE (có object)
    # ---------------------------------------------------------
    pos_mask = obj_mask.bool()          # đổi 0/1 (float) -> True/False để dùng làm mask lọc
    num_pos = pos_mask.sum().clamp(min=1)
    # num_pos = tổng số ô positive trong CẢ BATCH.
    # clamp(min=1): phòng trường hợp batch này không có object nào (num_pos=0)
    # thì tránh lỗi chia cho 0 ở bước chuẩn hóa bên dưới.

    # ---------------------------------------------------------
    # BƯỚC 3: BOX LOSS — CHỈ TÍNH TẠI Ô POSITIVE
    # ---------------------------------------------------------
    # Vì sao chỉ tính tại positive: ô negative không có ground-truth box thật,
    # nên box_target ở đó là rác -> nếu tính loss cả ở đó là ép model học sai.

    # pos_mask hiện có shape [B,H,W], còn pred_box có shape [B,4,H,W]
    # -> phải "nhân bản" mask ra 4 kênh để chỉ đúng vị trí cần lấy trên cả 4 kênh box.
    pos_mask_4 = pos_mask.unsqueeze(1).expand_as(pred_box)  # [B,1,H,W] -> [B,4,H,W]

    # pred_box[pos_mask_4] : lấy phẳng ra 1 vector 1-D, chỉ gồm giá trị
    # tại các vị trí positive (không quan tâm chúng nằm rải rác ở đâu trên lưới 20x20).
    box_loss = F.smooth_l1_loss(
        pred_box[pos_mask_4],
        box_target[pos_mask_4],
        reduction='sum'      # cộng dồn lỗi của tất cả giá trị lấy được
    ) / num_pos              # chia cho SỐ Ô positive (không phải tổng số ô)
    # -> lý do chia cho num_pos: nếu không, ảnh có nhiều object sẽ tự nhiên
    # có box_loss lớn hơn ảnh ít object, dù model dự đoán tốt/tệ như nhau.

    # ---------------------------------------------------------
    # BƯỚC 4: OBJECTNESS LOSS — TÍNH TRÊN TOÀN BỘ LƯỚI (cả positive lẫn negative)
    # ---------------------------------------------------------
    # Vì sao KHÔNG lọc mask ở đây: objectness cần học phân biệt
    # "đây là object" (target=1) VS "đây là background" (target=0).
    # Nếu chỉ tính tại positive, model sẽ không bao giờ học được cách
    # nhận diện background -> lúc suy luận sẽ báo object khắp mọi nơi.
    obj_loss = F.binary_cross_entropy_with_logits(
        pred_obj,            # logit thô, HÀM NÀY TỰ áp sigmoid bên trong, không cần làm tay
        obj_mask.float(),    # target: 1 = có object, 0 = không có
        reduction='mean'     # trung bình trên toàn bộ B*H*W ô
    )

    # ---------------------------------------------------------
    # BƯỚC 5: CLASS LOSS — CHỈ TÍNH TẠI Ô POSITIVE
    # ---------------------------------------------------------
    # Vì sao chỉ tính tại positive: ô negative (background) không thuộc
    # class nào cả (không phải person/helmet/vest) -> ép model phân loại
    # class cho background là vô nghĩa và gây nhiễu loss.
    if num_pos > 0:
        # permute(0,2,3,1): đổi thứ tự chiều từ [B,C,H,W] -> [B,H,W,C]
        # để C (số class) nằm ở chiều cuối -> khi lọc bằng pos_mask [B,H,W]
        # kết quả tự động ra shape [num_pos, C], đúng định dạng CrossEntropyLoss cần.
        pred_cls_pos = pred_cls.permute(0, 2, 3, 1)[pos_mask]   # [num_pos, C]
        cls_target_pos = cls_target[pos_mask]                  # [num_pos]

        cls_loss = F.cross_entropy(
            pred_cls_pos,       # logits (CHƯA softmax — hàm này tự làm)
            cls_target_pos,     # class index đúng, kiểu long
            reduction='mean'
        )
    else:
        # Không có object nào trong cả batch -> không có gì để tính class loss.
        # Trả về 0 nhưng vẫn phải nằm đúng device (GPU/CPU) để không lỗi khi cộng tổng.
        cls_loss = torch.tensor(0.0, device=raw_pred.device)

    # ---------------------------------------------------------
    # BƯỚC 6: CỘNG 3 LOSS THEO TRỌNG SỐ λ
    # ---------------------------------------------------------
    total = lambda_box * box_loss + lambda_obj * obj_loss + lambda_cls * cls_loss

    # trả thêm .detach() cho 3 loss thành phần: để log/in ra theo dõi training
    # mà KHÔNG kéo theo đồ thị gradient (tránh giữ bộ nhớ không cần thiết).
    return total, box_loss.detach(), obj_loss.detach(), cls_loss.detach()