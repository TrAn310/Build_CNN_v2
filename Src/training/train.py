"""
train.py

BUOC 1 (Bai 10 - Training Loop):
    Vong lap training CO BAN cho detector: forward (3 scale) -> build
    targets multi-scale -> tinh loss (box+obj+cls) -> backward ->
    optimizer.step(). Ban da biet training loop tu classification roi
    nen khong nhac lai phan do -- chi giai thich phan RIENG cua
    detector (targets duoc xay dung the nao, loss gom may thanh phan).

    Test BAT BUOC cua buoc nay (PHAN 22 trong Prompt): OVERFIT 1 ANH.
    Neu model KHONG overfit duoc 1 anh don gian voi vai object, thi
    CHUA duoc phep dung dataset that -- loi chac chan nam o CODE,
    khong phai o DU LIEU.

BUOC 2 (Bai 11 - Training that + Validation):
    Ham run_training() o duoi day la training loop DAY DU tren dataset
    YOLO THAT: DataLoader (dataset.py + collate.py) -> train 1 epoch ->
    validate (model.eval() -> decode_and_filter -> evaluator.calculate_map)
    -> MetricsLogger quyet dinh is_best DUA TREN VALIDATION mAP (khong
    dua train loss) -> luu last_model.pth moi epoch, best_model.pth chi
    khi is_best -> luu lich su ra CSV.

    Khac voi overfit test o __main__ (PHAN 22 - van GIU NGUYEN de lam
    bai kiem tra code doc lap, khong phu thuoc dataset that): run_training()
    moi la ham dung de train THAT SU tren du lieu YOLO cua ban.

CAU TRUC THU MUC THAT:
    Src/
    ├── dataset/       (collate.py, dataset.py, parser.py)
    ├── evaluation/    (evaluator.py)
    ├── inference/     (decoder.py, postprocess.py)
    ├── model/         (Backbone.py, block.py, detector.py, head.py, neck.py)
    │   └── utils/     (config_utils.py)
    ├── training/      (losses.py, target_assigner.py, train.py)  <- file nay o day
    └── Utils/         (Metrics.py, Visualization.py)

    train.py va cac thu muc kia la ANH EM (cung nam trong Src/).
    Python KHONG tu tim module o thu muc anh em, nen phai tu them
    TUNG thu muc con chua module can dung vao sys.path TRUOC khi import.
"""

import os
import sys

# Src/training/train.py -> _SRC_DIR se la .../Src
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)

# Cac file trong du an dung IMPORT PHANG (vd "from Backbone import Backbone"
# trong detector.py), nen phai them TUNG THU MUC CON chua module can dung.
# LUU Y: utils nam LONG BEN TRONG model/ (Src/model/utils/), khong phai
# Src/utils/ truc tiep -> phai noi duong dan "model/utils" chinh xac.
_SUBFOLDERS = [
    "model",
    os.path.join("model", "utils"),
    "dataset",
    "inference",
    "evaluation",
    "Utils",   # Metrics.py, Visualization.py nam o day (Src/Utils/)
]

for _sub in _SUBFOLDERS:
    _path = os.path.normpath(os.path.join(_SRC_DIR, _sub))
    if _path not in sys.path:
        sys.path.append(_path)

import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from detector import MiniPPEDetector
from losses import DetectionLoss, compute_multiscale_loss
from config_utils import load_dataset_config
from dataset import PPEDetectionDataset
from collate import collate_fn
from postprocess import decode_and_filter
from evaluator import match_predictions_to_ground_truth, summarize, calculate_map
from Metrics import AverageMeter, MetricsLogger


def save_checkpoint(model, num_classes, class_names, save_path, img_size=640):
    """
    Luu checkpoint kem THEO CA METADATA (num_classes, class_names, img_size),
    khong chi trong so thuan tuy. Day la diem mau chot de predict.py sau
    nay KHONG BAO GIO can sua code cung du doi dataset/so class khac nhau -
    predict.py se tu doc lai dung thong tin nay tu file checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
        "img_size": img_size,
    }, save_path)
    print(f"Da luu checkpoint: {save_path}")


def train_one_step(model, images, gt_boxes, gt_classes, loss_fn, optimizer):
    """
    1 BUOC training duy nhat -- day la toan bo phan MOI so voi
    training loop classification ban da biet:

        predictions = model(images)                # 3 tensor (P3,P4,P5)
        loss = compute_multiscale_loss(...)         # tu build targets ben trong
        loss.backward(); optimizer.step()           # giong het classification

    Args:
        model:      MiniPPEDetector
        images:     tensor [B,3,H,W]
        gt_boxes:   list[Tensor[N_i,4]]   -- dung format dataset.py/collate.py
        gt_classes: list[Tensor[N_i]]
        loss_fn:    instance DetectionLoss (tao 1 lan, tai su dung xuyen suot)
        optimizer:  torch optimizer

    Returns:
        loss_dict: dict cac thanh phan loss cua tung scale (de log/theo doi)
    """
    model.train()

    preds = model(images)   # (pred_p3, pred_p4, pred_p5)

    total_loss, loss_dict = compute_multiscale_loss(
        preds, gt_boxes, gt_classes, model.num_classes, loss_fn
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return loss_dict


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, log_every=10):
    """
    Chay 1 EPOCH train that su (nhieu batch, tu DataLoader that -- khac
    voi overfit test chi lap lai DUNG 1 anh gia 200 lan).

    Chi la vong lap goi lai train_one_step() cho tung batch + cong don
    loss bang AverageMeter -- KHONG dinh nghia logic train moi.

    In tien do MOI log_every batch (kem s/batch + ETA con lai cua epoch)
    -- de tren man hinh LUON CO gi do chay, tranh nhin giong bi "dung
    hinh" khi dataset lon / chay tren CPU (moi epoch co the mat vai
    chuc phut, ma neu khong in gi ca thi de tuong nham la treo may).

    Returns:
        train_loss: float, trung binh total_loss CA EPOCH (co trong so
            theo batch_size, dung AverageMeter.update(..., n=batch_size))
    """
    import time

    model.train()
    meter = AverageMeter()
    num_batches = len(train_loader)
    start_time = time.time()

    for batch_idx, (images, gt_boxes, gt_classes) in enumerate(train_loader, start=1):
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_classes = [c.to(device) for c in gt_classes]

        loss_dict = train_one_step(model, images, gt_boxes, gt_classes, loss_fn, optimizer)
        meter.update(loss_dict["total_loss"], n=images.shape[0])

        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
            elapsed = time.time() - start_time
            sec_per_batch = elapsed / batch_idx
            eta_sec = sec_per_batch * (num_batches - batch_idx)
            print(
                f"  [train] batch {batch_idx:4d}/{num_batches} | "
                f"loss={loss_dict['total_loss']:.4f} | "
                f"{sec_per_batch:.2f}s/batch | ETA epoch nay: {eta_sec/60:.1f} phut",
                flush=True,
            )

    return meter.avg


def validate(model, val_loader, num_classes, loss_fn, device,
             conf_threshold=0.3, nms_iou_threshold=0.5, map_iou_threshold=0.5,
             log_every=10):
    """
    Validation DAY DU cho 1 epoch -- day la phan MOI cua Buoc 2, khac
    han train_one_epoch():

        model.eval() + torch.no_grad()  (khong hoc, chi danh gia)
        -> forward -> decode_and_filter (postprocess.py: decode + NMS,
           GIONG HET pipeline predict.py that su dung khi inference)
        -> gom prediction + ground-truth CUA CA TAP VALID lai
        -> evaluator.calculate_map() (Buoc 2 vua viet o evaluator.py)

    val_loss o day CHI de theo doi/debug (xem model co overfit khong),
    KHONG dung de chon best model -- best model PHAI dua tren mAP, dung
    yeu cau PHAN 23 (xem MetricsLogger.log_epoch).

    NMS trong postprocess.py la vong lap Python thuan (khong vector
    hoa), nen validate co the CHAM HON train ro rang tren so luong anh
    it hon -- vi vay cung in tien do MOI log_every batch giong
    train_one_epoch(), tranh nhin giong treo may.

    Returns:
        val_loss:      float
        precision:     float, TP/(TP+FP) cong don TREN CA TAP VALID
        recall:        float, TP/(TP+FN) cong don TREN CA TAP VALID
        map_score:      float, mAP@map_iou_threshold
        ap_per_class:  dict {class_id: ap}
    """
    import time

    model.eval()
    loss_meter = AverageMeter()

    all_predictions = []
    all_ground_truths = []

    num_batches = len(val_loader)
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, gt_boxes, gt_classes) in enumerate(val_loader, start=1):
            images = images.to(device)
            gt_boxes_dev = [b.to(device) for b in gt_boxes]
            gt_classes_dev = [c.to(device) for c in gt_classes]

            preds = model(images)

            # val_loss: dung LAI DUNG cong thuc loss cua luc train (chi
            # khac o cho khong backward), de con so co the so sanh duoc
            # voi train_loss tren cung 1 thang do.
            total_loss, _ = compute_multiscale_loss(
                preds, gt_boxes_dev, gt_classes_dev, model.num_classes, loss_fn
            )
            loss_meter.update(total_loss.item(), n=images.shape[0])

            # decode_and_filter dung CHINH XAC pipeline postprocess.py
            # (decode -> threshold -> NMS rieng tung class) ma predict.py
            # dang dung khi inference that -- khong viet lai logic khac.
            results = decode_and_filter(
                preds, conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold
            )

            for b in range(images.shape[0]):
                boxes_b, scores_b, classes_b = results[b]
                all_predictions.append((boxes_b.cpu(), scores_b.cpu(), classes_b.cpu()))
                all_ground_truths.append((gt_boxes[b], gt_classes[b]))

            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
                elapsed = time.time() - start_time
                sec_per_batch = elapsed / batch_idx
                eta_sec = sec_per_batch * (num_batches - batch_idx)
                print(
                    f"  [valid] batch {batch_idx:4d}/{num_batches} | "
                    f"{sec_per_batch:.2f}s/batch | ETA validate: {eta_sec/60:.1f} phut",
                    flush=True,
                )

    map_score, ap_per_class = calculate_map(
        all_predictions, all_ground_truths, num_classes, iou_threshold=map_iou_threshold
    )

    # Precision/recall CONG DON tren CA TAP VALID (khong phai tung anh
    # rieng le) -- tai su dung match_predictions_to_ground_truth() +
    # summarize() cua evaluator.py cho tung anh roi cong TP/FP/FN lai.
    total_tp = total_fp = total_fn = 0
    for (boxes_b, scores_b, classes_b), (gt_boxes_b, gt_classes_b) in zip(
        all_predictions, all_ground_truths
    ):
        if boxes_b.shape[0] == 0:
            total_fn += gt_boxes_b.shape[0]
            continue
        _, is_tp, num_gt = match_predictions_to_ground_truth(
            boxes_b, scores_b, classes_b, gt_boxes_b, gt_classes_b, map_iou_threshold
        )
        s = summarize(is_tp, num_gt)
        total_tp += s["TP"]
        total_fp += s["FP"]
        total_fn += s["FN"]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return loss_meter.avg, precision, recall, map_score, ap_per_class


def run_training(train_img_dir, train_label_dir, val_img_dir, val_label_dir,
                  data_yaml_path, epochs=50, batch_size=8, lr=1e-3, img_size=640,
                  conf_threshold=0.3, nms_iou_threshold=0.5, map_iou_threshold=0.5,
                  device=None):
    """
    TRAINING THAT SU tren dataset YOLO that -- day la ham ban goi de
    train tren du lieu cua minh (khac voi overfit test o __main__, von
    chi la bai KIEM TRA CODE tren 1 anh gia).

    Args:
        train_img_dir, train_label_dir: thu muc anh/label YOLO cua tap train
        val_img_dir, val_label_dir:     thu muc anh/label YOLO cua tap valid
        data_yaml_path:                 duong dan data.yaml (nc, names)
        epochs, batch_size, lr, img_size: sieu tham so training co ban
        conf_threshold, nms_iou_threshold: dung trong buoc decode+NMS luc validate
        map_iou_threshold:              nguong IoU de tinh mAP (mac dinh mAP@0.5)
        device:                         "cuda"/"cpu", None -> tu dong chon

    Output (ghi ra dia, dung cau truc thu muc PHAN 32):
        outputs/checkpoints/last_model.pth  -- ghi de MOI epoch
        outputs/checkpoints/best_model.pth  -- CHI ghi de khi validation
                                                mAP cai thien (is_best=True)
        outputs/logs/train_history.csv      -- lich su train_loss/val_loss/
                                                precision/recall/mAP tung epoch
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    num_classes, class_names = load_dataset_config(data_yaml_path)

    train_dataset = PPEDetectionDataset(train_img_dir, train_label_dir, img_size=img_size)
    val_dataset = PPEDetectionDataset(val_img_dir, val_label_dir, img_size=img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = MiniPPEDetector(num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = DetectionLoss()
    logger = MetricsLogger()

    _PROJECT_ROOT = os.path.dirname(_SRC_DIR)
    _CKPT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints")
    _LOG_CSV_PATH = os.path.join(_PROJECT_ROOT, "outputs", "logs", "train_history.csv")

    print(f"Thiet bi          : {device}")
    print(f"So anh train/valid: {len(train_dataset)} / {len(val_dataset)}")
    print(f"So batch train/valid (batch_size={batch_size}): "
          f"{len(train_loader)} / {len(val_loader)}")
    print(f"num_classes       : {num_classes}  class_names: {class_names}\n")

    for epoch in range(1, epochs + 1):
        print(f"--- Epoch {epoch}/{epochs}: bat dau train ---")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        print(f"--- Epoch {epoch}/{epochs}: bat dau validate ---")
        val_loss, precision, recall, map_score, ap_per_class = validate(
            model, val_loader, num_classes, loss_fn, device,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            map_iou_threshold=map_iou_threshold,
        )

        is_best = logger.log_epoch(epoch, train_loss, val_loss, precision, recall, map_score)

        flag = "  <-- BEST (luu best_model.pth)" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} | precision={precision:.3f} "
            f"recall={recall:.3f} | mAP@{map_iou_threshold}={map_score:.4f}{flag}"
        )

        # last_model.pth: LUON ghi de moi epoch (de biet trang thai train gan nhat)
        save_checkpoint(
            model, num_classes, class_names,
            save_path=os.path.join(_CKPT_DIR, "last_model.pth"), img_size=img_size,
        )
        # best_model.pth: CHI ghi de khi validation mAP epoch nay cao nhat
        # tu truoc den gio (dung PHAN 23 - khong dua train loss)
        if is_best:
            save_checkpoint(
                model, num_classes, class_names,
                save_path=os.path.join(_CKPT_DIR, "best_model.pth"), img_size=img_size,
            )

        logger.save_csv(_LOG_CSV_PATH)

    print(f"\nHoan tat training. Best epoch = {logger.best_epoch} (mAP={logger.best_map:.4f})")
    print(f"Checkpoint luu tai : {_CKPT_DIR}")
    print(f"Lich su train (CSV): {_LOG_CSV_PATH}")

    return model, logger


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Khong truyen tham so nao -> chay OVERFIT TEST (PHAN 22) tren 1 anh gia. "
                     "Truyen --train-img-dir (va cac tham so con lai) -> train THAT tren dataset YOLO."
    )
    _parser.add_argument("--train-img-dir", type=str, default=None)
    _parser.add_argument("--train-label-dir", type=str, default=None)
    _parser.add_argument("--val-img-dir", type=str, default=None)
    _parser.add_argument("--val-label-dir", type=str, default=None)
    _parser.add_argument("--data-yaml", type=str, default=None,
                          help="Duong dan data.yaml. Mac dinh: Src/Data/data.yaml")
    _parser.add_argument("--epochs", type=int, default=50)
    _parser.add_argument("--batch-size", type=int, default=8)
    _parser.add_argument("--lr", type=float, default=1e-3)
    _parser.add_argument("--img-size", type=int, default=640)
    _args = _parser.parse_args()

    if _args.train_img_dir is not None:
        # ==============================================================
        # TRAIN THAT tren dataset YOLO
        # ==============================================================
        _data_yaml = _args.data_yaml or os.path.normpath(
            os.path.join(_SRC_DIR, "Data", "data.yaml")
        )
        run_training(
            train_img_dir=_args.train_img_dir,
            train_label_dir=_args.train_label_dir,
            val_img_dir=_args.val_img_dir,
            val_label_dir=_args.val_label_dir,
            data_yaml_path=_data_yaml,
            epochs=_args.epochs,
            batch_size=_args.batch_size,
            lr=_args.lr,
            img_size=_args.img_size,
        )
        sys.exit(0)

    # ==================================================================
    # PHAN 22 - OVERFIT 1 ANH
    # Bai test bat buoc TRUOC KHI dung dataset that.
    # ==================================================================
    torch.manual_seed(0)

    # ---- Lay num_classes, class_names DONG tu data.yaml (khong hard-code) ----
    # Neu chua co data.yaml dung duong dan, fallback ve 3 class mac dinh de
    # overfit test van chay duoc doc lap, khong phu thuoc dataset that.
    _yaml_path = os.path.normpath(os.path.join(_SRC_DIR, "Data", "data.yaml"))
    try:
        num_classes, class_names = load_dataset_config(_yaml_path)
    except FileNotFoundError:
        print(f"Khong tim thay {_yaml_path}, dung fallback 3 class mac dinh.")
        num_classes = 3
        class_names = ["Person", "Helmet", "Vest"]

    model = MiniPPEDetector(num_classes=num_classes)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = DetectionLoss()

    # 1 anh gia CO DINH xuyen suot qua trinh train (dac diem cua overfit test:
    # KHONG doi anh moi step, de xem model co "thuoc long" duoc 1 anh khong).
    image = torch.randn(1, 3, 640, 640)

    # 2 object gia, co chu dinh de roi vao 2 scale khac nhau:
    #   Person: w=200,h=300 (canh lon nhat 300 >=128) -> P5
    #   Helmet: w=40, h=40  (canh lon nhat 40  <64)   -> P3
    gt_boxes = [torch.tensor([
        [200., 150., 400., 450.],
        [500., 100., 540., 140.],
    ])]
    gt_classes = [torch.tensor([0, 1])]

    print("Bat dau overfit 1 anh -- neu code dung, total_loss phai giam manh.\n")

    history = []
    for step in range(1, 201):
        loss_dict = train_one_step(model, image, gt_boxes, gt_classes, loss_fn, optimizer)
        history.append(loss_dict["total_loss"])

        if step == 1 or step % 20 == 0:
            print(
                f"Step {step:3d} | total={loss_dict['total_loss']:.4f} | "
                f"box(P3/P4/P5)={loss_dict['loss_P3_box']:.3f}/{loss_dict['loss_P4_box']:.3f}/{loss_dict['loss_P5_box']:.3f} | "
                f"obj(P3/P4/P5)={loss_dict['loss_P3_obj']:.3f}/{loss_dict['loss_P4_obj']:.3f}/{loss_dict['loss_P5_obj']:.3f} | "
                f"cls(P3/P5)={loss_dict['loss_P3_cls']:.3f}/{loss_dict['loss_P5_cls']:.3f}"
            )

    print(f"\ntotal_loss step 1   : {history[0]:.4f}")
    print(f"total_loss step 200 : {history[-1]:.4f}")
    print("\nExpected: total_loss giam manh (ly tuong ve gan 0). Neu total_loss")
    print("KHONG giam hoac dung yen tu step 1 -> co bug o build_targets/loss/")
    print("backward -- PHAI sua truoc khi dung dataset that (dung do loi cho du lieu).")

    # ---- LUU CHECKPOINT ----
    # outputs/checkpoints/ nam o CAP GOC project (ngang hang voi Src/),
    # dung PHAN 32 trong Prompt goc.
    _PROJECT_ROOT = os.path.dirname(_SRC_DIR)
    _CKPT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints")

    save_checkpoint(
        model, num_classes, class_names,
        save_path=os.path.join(_CKPT_DIR, "last_model.pth"),
    )
    # Overfit test khong co validation that nen tam thoi luu ca best = last.
    # Khi viet training loop that (dataset that + validation mAP), best_model.pth
    # se chi duoc ghi de khi validation mAP cai thien - chua lam trong file nay.
    save_checkpoint(
        model, num_classes, class_names,
        save_path=os.path.join(_CKPT_DIR, "best_model.pth"),
    )