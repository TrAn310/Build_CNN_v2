"""
run_train.py

Entry point kieu YOLO. 2 CACH DUNG:

  1) DON GIAN NHAT (khuyen dung, KHONG can dong vao data.yaml):
     chi truyen THU MUC GOC dataset (co san train/, valid/, data.yaml
     dung dinh dang Roboflow) qua --dataset-dir:

         python run_train.py --dataset-dir Data --epochs 100 --batch 16

     Code se TU SUY RA train/images, train/labels, valid/images,
     valid/labels tu chinh thu muc do, KHONG dong cham gi den noi
     dung file data.yaml (chi doc rieng nc/names trong do).

  2) NANG CAO (khi dataset khong theo dung quy uoc train/valid+images,
     hoac muon kiem soat duong dan chi tiet qua chinh data.yaml):
     truyen thang file data.yaml chuan YOLO (co day du path/train/val)
     qua --data:

         python run_train.py --data Data/data.yaml --epochs 100 --batch 16

     Xem config_utils.resolve_dataset_paths() de biet dinh dang
     data.yaml can co cho cach nay.

Chi can dung 1 trong 2 (--dataset-dir HOAC --data). Neu khong truyen
gi ca, mac dinh dung --dataset-dir = Src/Data (dung setup pho bien
nhat: dataset Roboflow giai nen thang vao Src/Data/).

File nay CHI la 1 LOP VO CLI: suy ra 4 duong dan anh/label (train +
valid) -> goi lai run_training() trong training/train.py. KHONG dinh
nghia lai logic training o day (viet 1 lan, dung lai).
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Src/

# ======================================================================
# CAU HINH MAC DINH -- SUA TRUC TIEP O DAY CHO KHOP VOI MAY CUA BAN.
#
# Neu ban KHONG truyen --dataset-dir (hoac --data) khi chay lenh, script
# se TU DONG dung DATASET_DIR ben duoi -- tuc la chi can go:
#     python run_train.py
# la chay thang, khong can go them gi ca.
#
# Van co the GHI DE tung gia tri nay bang tham so dong lenh neu muon,
# vi du: python run_train.py --epochs 30 (dung DATASET_DIR mac dinh,
# nhung doi so epoch thanh 30).
# ======================================================================
DATASET_DIR = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Src\Data"  # <-- SUA DUONG DAN NAY
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
LR = 1e-3
# ======================================================================

# Chi can 2 thu muc de import duoc config_utils.py va training/train.py.
# (training/train.py, khi duoc import, se TU NO them cac thu muc con
# khac no can - dataset/, inference/, evaluation/, Utils/ - vao sys.path,
# xem lai dau file train.py).
_SUBFOLDERS = [
    os.path.join("model", "utils"),
    "training",
]
for _sub in _SUBFOLDERS:
    _path = os.path.normpath(os.path.join(_CURRENT_DIR, _sub))
    if _path not in sys.path:
        sys.path.append(_path)

from config_utils import resolve_dataset_paths, resolve_dataset_paths_from_folder
from train import run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train MiniPPEDetector tren dataset dinh dang YOLO. "
            "Khong truyen gi ca -> dung DATASET_DIR/EPOCHS/... khai bao san o dau file nay. "
            "Truyen tham so -> GHI DE gia tri mac dinh do."
        )
    )
    parser.add_argument("--dataset-dir", type=str, default=None,
                         help=f"[DON GIAN NHAT] Thu muc GOC dataset, co san train/, valid/, data.yaml "
                              f"(quy uoc Roboflow) -- KHONG can sua data.yaml. "
                              f"Mac dinh (khai bao trong code): {DATASET_DIR}")
    parser.add_argument("--data", type=str, default=None,
                         help="[NANG CAO] Duong dan truc tiep toi 1 file data.yaml CHUAN YOLO "
                              "(co day du key path/train/val). Dung khi --dataset-dir khong hop "
                              "voi cau truc dataset cua ban.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="So epoch train")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Kich thuoc anh vuong dua vao model")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate cho Adam")
    parser.add_argument("--conf-thres", type=float, default=0.3,
                         help="Nguong confidence khi decode luc validate (postprocess.py)")
    parser.add_argument("--iou-thres", type=float, default=0.5,
                         help="Nguong IoU dung trong NMS luc validate")
    parser.add_argument("--map-iou", type=float, default=0.5,
                         help="Nguong IoU de tinh mAP (mac dinh mAP@0.5)")
    parser.add_argument("--device", type=str, default=None,
                         help="'cuda' hoac 'cpu'. Mac dinh: tu dong chon GPU neu co")

    args = parser.parse_args()

    # Neu nguoi dung KHONG truyen ca --dataset-dir lan --data -> fallback
    # ve DATASET_DIR khai bao san trong code o dau file.
    if args.dataset_dir is None and args.data is None:
        args.dataset_dir = DATASET_DIR

    return args


def main():
    args = parse_args()

    if args.dataset_dir is not None:
        # ---- CACH 1: DON GIAN NHAT, chi can 1 thu muc ----
        dataset_dir = os.path.abspath(args.dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Khong tim thay thu muc dataset: {dataset_dir}")

        paths = resolve_dataset_paths_from_folder(dataset_dir)
        data_yaml_path = os.path.join(dataset_dir, "data.yaml")
        source_desc = f"--dataset-dir {dataset_dir}"
    else:
        # ---- CACH 2: NANG CAO, truyen thang file data.yaml ----
        data_yaml_path = os.path.abspath(args.data)
        if not os.path.isfile(data_yaml_path):
            raise FileNotFoundError(f"Khong tim thay file data.yaml: {data_yaml_path}")

        paths = resolve_dataset_paths(data_yaml_path)
        source_desc = f"--data {data_yaml_path}"

    for name, d in [
        ("train images", paths["train_img_dir"]),
        ("train labels", paths["train_label_dir"]),
        ("val images", paths["val_img_dir"]),
        ("val labels", paths["val_label_dir"]),
    ]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Thu muc {name} khong ton tai: {d}\n"
                f"(Dang doc dataset tu: {source_desc}. Kiem tra lai cau truc thu muc dataset, "
                f"hoac dung --data de tu khai bao duong dan trong data.yaml neu cau truc dataset "
                f"khong theo quy uoc train/valid + images/.)"
            )

    print("=" * 60)
    print("CAU HINH TRAINING")
    print("=" * 60)
    print(f"Nguon dataset: {source_desc}")
    print(f"num_classes  : {paths['num_classes']}")
    print(f"class_names  : {paths['class_names']}")
    print(f"train images : {paths['train_img_dir']}")
    print(f"train labels : {paths['train_label_dir']}")
    print(f"val images   : {paths['val_img_dir']}")
    print(f"val labels   : {paths['val_label_dir']}")
    print(f"epochs       : {args.epochs}")
    print(f"batch size   : {args.batch}")
    print(f"img size     : {args.imgsz}")
    print(f"learning rate: {args.lr}")
    print("=" * 60 + "\n")

    run_training(
        train_img_dir=paths["train_img_dir"],
        train_label_dir=paths["train_label_dir"],
        val_img_dir=paths["val_img_dir"],
        val_label_dir=paths["val_label_dir"],
        data_yaml_path=data_yaml_path,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_size=args.imgsz,
        conf_threshold=args.conf_thres,
        nms_iou_threshold=args.iou_thres,
        map_iou_threshold=args.map_iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()