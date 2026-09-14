"""
run_train.py

Entry point kieu YOLO: chi can 1 lenh voi --data la du, KHONG phai tu
go --train-img-dir/--train-label-dir/--val-img-dir/--val-label-dir
rieng le nhu train.py (dung khi ban goi "python train.py --train-img-dir ...").

Vi du dung (giong het cach dung "python train.py --data data.yaml
--epochs 100 --batch 16" cua YOLO that):

    python run_train.py --data Data/data.yaml --epochs 100 --batch 16 --imgsz 640

File nay CHI la 1 LOP VO CLI: doc data.yaml -> suy ra 4 duong dan
anh/label (train + valid) theo dung quy uoc YOLO -> goi lai
run_training() trong training/train.py. KHONG dinh nghia lai logic
training o day (viet 1 lan, dung lai, tranh code trung lap 2 noi).

data.yaml ky vong dinh dang CHUAN YOLO:

    path: ../Data          # (tuy chon) thu muc goc dataset
    train: train/images    # thu muc ANH cua tap train
    val: valid/images      # thu muc ANH cua tap valid
    nc: 3
    names: ['Person', 'Helmet', 'Vest']

Thu muc LABEL (train/labels, valid/labels) duoc TU DONG suy ra tu thu
muc anh (thay thanh phan "images" -> "labels" trong duong dan) --
KHONG can khai bao rieng, giong het quy uoc that cua YOLO.
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Src/

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

from config_utils import resolve_dataset_paths
from train import run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train MiniPPEDetector tren dataset dinh dang YOLO. "
            "Cach dung: python run_train.py --data Data/data.yaml --epochs 100 --batch 16 --imgsz 640"
        )
    )
    parser.add_argument("--data", type=str, required=True,
                         help="Duong dan data.yaml (chuan YOLO: path/train/val/nc/names)")
    parser.add_argument("--epochs", type=int, default=100, help="So epoch train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Kich thuoc anh vuong dua vao model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho Adam")
    parser.add_argument("--conf-thres", type=float, default=0.3,
                         help="Nguong confidence khi decode luc validate (postprocess.py)")
    parser.add_argument("--iou-thres", type=float, default=0.5,
                         help="Nguong IoU dung trong NMS luc validate")
    parser.add_argument("--map-iou", type=float, default=0.5,
                         help="Nguong IoU de tinh mAP (mac dinh mAP@0.5)")
    parser.add_argument("--device", type=str, default=None,
                         help="'cuda' hoac 'cpu'. Mac dinh: tu dong chon GPU neu co")
    return parser.parse_args()


def main():
    args = parse_args()

    data_yaml_path = os.path.abspath(args.data)
    if not os.path.isfile(data_yaml_path):
        raise FileNotFoundError(f"Khong tim thay file data.yaml: {data_yaml_path}")

    # Suy ra 4 duong dan anh/label + num_classes/class_names TU 1 FILE
    # data.yaml duy nhat -- day la diem khac biet chinh so voi train.py
    # (train.py can nguoi dung tu go tung duong dan rieng).
    paths = resolve_dataset_paths(data_yaml_path)

    for name, d in [
        ("train images", paths["train_img_dir"]),
        ("train labels", paths["train_label_dir"]),
        ("val images", paths["val_img_dir"]),
        ("val labels", paths["val_label_dir"]),
    ]:
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Thu muc {name} khong ton tai: {d}\n"
                f"(Kiem tra lai key 'path'/'train'/'val' trong {data_yaml_path}, "
                f"va dam bao anh nam trong 1 thu muc ten 'images' de tu dong suy "
                f"ra thu muc 'labels' tuong ung.)"
            )

    print("=" * 60)
    print("CAU HINH TRAINING")
    print("=" * 60)
    print(f"data.yaml    : {data_yaml_path}")
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