"""
predict.py

Pipeline suy luan (inference) hoan chinh tren 1 anh:

    load image -> resize -> normalize -> tensor -> model -> decode
    -> threshold -> NMS -> ve box -> luu ket qua

THIET KE DE "VIET 1 LAN, DUNG MAI": moi thu tin quan trong (so class,
ten class, kich thuoc anh model can) deu duoc DOC TU CHECKPOINT, khong
hard-code trong file nay. Neu sau nay ban doi dataset (so class khac,
ten class khac) hay train lai model voi cau hinh khac, CHI CAN train.py
luu dung checkpoint moi - predict.py khong can sua gi ca.

CAU TRUC THU MUC: file nay dat o Src/inference/ (cung cho voi decoder.py,
postprocess.py). Can them Src/model/ vao sys.path de import duoc detector.py.
"""

import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # Src/inference
_SRC_DIR = os.path.dirname(_CURRENT_DIR)                     # Src/

for _sub in ("model", os.path.join("model", "utils")):
    _path = os.path.normpath(os.path.join(_SRC_DIR, _sub))
    if _path not in sys.path:
        sys.path.append(_path)

import cv2
import torch
import numpy as np

from detector import MiniPPEDetector
from postprocess import decode_and_filter


# Mau co dinh cho tung class de ve box - khong quan trong dung mau gi
# (theo dung PHAN 29 trong Prompt: "Mau sac khong quan trong"), chi can
# NHAT QUAN giua cac lan chay va DU nhieu mau de phan biet cac class.
_COLOR_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
    (128, 128, 0), (0, 128, 255),
]


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """
    Doc checkpoint (da luu boi train.py) va dung lai model.

    Input:
        checkpoint_path: duong dan file .pth

    Output:
        model:       MiniPPEDetector, da load trong so, o che do eval()
        class_names: list[str]
        img_size:    int - kich thuoc anh vuong model can (vd 640)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 640)

    model = MiniPPEDetector(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()   # QUAN TRONG: tat Dropout/doi cach hoat dong cua BatchNorm

    return model, class_names, img_size


def preprocess_image(image_path, img_size):
    """
    Doc anh tu file, resize ve hinh vuong img_size x img_size, chuan hoa
    ve [0,1], doi BGR (OpenCV mac dinh) sang RGB, doi HWC -> CHW, them
    chieu batch.

    LUU Y (gioi han hien tai, khong anh huong tinh dung cua pipeline):
    resize truc tiep KHONG giu ty le khung hinh goc (khong letterbox).
    Neu anh goc khong vuong, object se bi meo nhe theo 1 chieu. Day la
    lua chon don gian hoa co chu dich (dung tinh than "code toi thieu
    truoc, toi uu sau" cua roadmap) - co the nang cap letterbox sau ma
    KHONG anh huong toi giao dien ham nay (van nhan image_path, tra ve
    tensor + kich thuoc goc).

    Output:
        tensor:        [1,3,img_size,img_size], dtype float32
        original_bgr:  anh goc (BGR, chua resize) de ve box len sau nay
        orig_h, orig_w: kich thuoc anh GOC, dung de quy doi box ve dung
                        ty le khi resize nguoc lai
    """
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    orig_h, orig_w = original_bgr.shape[:2]

    resized = cv2.resize(original_bgr, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)   # them chieu batch -> [1,3,H,W]

    return tensor, original_bgr, orig_h, orig_w


def scale_boxes_to_original(boxes, img_size, orig_h, orig_w):
    """
    Box hien dang o he toa do anh da resize (img_size x img_size).
    Quy doi ve he toa do anh GOC (orig_w x orig_h) de ve dung vi tri.
    """
    if boxes.shape[0] == 0:
        return boxes

    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    boxes = boxes.clone()
    boxes[:, 0] *= scale_x  # x1
    boxes[:, 2] *= scale_x  # x2
    boxes[:, 1] *= scale_y  # y1
    boxes[:, 3] *= scale_y  # y2

    return boxes


def draw_boxes(image_bgr, boxes, scores, classes, class_names):
    """
    Ve box + nhan class + confidence len anh (dung OpenCV, tu code -
    khong dung thu vien detection nao).
    """
    image = image_bgr.copy()

    for box, score, cls_id in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = _COLOR_PALETTE[cls_id % len(_COLOR_PALETTE)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_names[cls_id]} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def predict(image_path, checkpoint_path, output_path,
            conf_threshold=0.3, nms_iou_threshold=0.5, device="cpu"):
    """
    Ham tong hop toan bo pipeline - day la ham ban se goi tu ben ngoai.
    """
    model, class_names, img_size = load_model_from_checkpoint(checkpoint_path, device)

    tensor, original_bgr, orig_h, orig_w = preprocess_image(image_path, img_size)
    tensor = tensor.to(device)

    with torch.no_grad():   # khong can tinh gradient luc inference -> tiet kiem bo nho
        preds = model(tensor)

    results = decode_and_filter(preds, conf_threshold=conf_threshold,
                                 nms_iou_threshold=nms_iou_threshold)
    boxes, scores, classes = results[0]   # batch size = 1 (chi 1 anh)

    boxes = scale_boxes_to_original(boxes, img_size, orig_h, orig_w)

    result_image = draw_boxes(original_bgr, boxes, scores, classes, class_names)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result_image)

    print(f"Phat hien {boxes.shape[0]} object.")
    for box, score, cls_id in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        print(f"  {class_names[cls_id]:15s} {score:.2f}  {[round(v, 1) for v in box]}")
    print(f"Da luu ket qua: {output_path}")

    return boxes, scores, classes


if __name__ == "__main__":
    # ---- TEST predict.py bang checkpoint vua luu tu train.py (overfit test) ----
    # LUU Y: checkpoint nay chi hoc tu 1 ANH GIA NGAU NHIEN, nen ket qua
    # detect tren anh THAT se KHONG co y nghia - muc dich cua test nay
    # CHI la xac nhan TOAN BO PIPELINE CODE chay dung, khong loi shape/import.
    _PROJECT_ROOT = os.path.dirname(_SRC_DIR)
    _CKPT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints", "last_model.pth")

    # Doi thanh duong dan anh that cua ban de test
    _TEST_IMAGE = os.path.join(_SRC_DIR, "Data", "valid", "images")
    # Neu ban chua co duong dan anh cu the, comment 2 dong tren va tu
    # dien duong dan 1 anh .jpg/.png that vao _TEST_IMAGE ben duoi:
    # _TEST_IMAGE = r"F:\duong\dan\toi\anh_test.jpg"

    _OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "predictions", "result.jpg")

    if not os.path.exists(_CKPT_PATH):
        print(f"Chua co checkpoint tai {_CKPT_PATH}.")
        print("Chay train.py truoc de tao checkpoint, roi chay lai file nay.")
    elif not os.path.isfile(_TEST_IMAGE):
        print(f"Duong dan anh test khong hop le: {_TEST_IMAGE}")
        print("Sua bien _TEST_IMAGE trong file nay thanh duong dan 1 anh that (.jpg/.png).")
    else:
        predict(_TEST_IMAGE, _CKPT_PATH, _OUTPUT_PATH)