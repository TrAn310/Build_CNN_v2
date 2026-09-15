import os
import sys

# debug_scores.py nam TRUC TIEP trong Src/, nen day chinh la _SRC_DIR
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)   # Build_CNN_v2/

_SUBFOLDERS = [
    "model",
    os.path.join("model", "utils"),
    "dataset",
    "inference",
    "evaluation",
    "Utils",
]
for _sub in _SUBFOLDERS:
    _path = os.path.normpath(os.path.join(_SRC_DIR, _sub))
    if _path not in sys.path:
        sys.path.append(_path)

import torch
from torch.utils.data import DataLoader

from detector import MiniPPEDetector
from postprocess import decode_and_filter
from dataset import PPEDetectionDataset
from collate import collate_fn
from config_utils import resolve_dataset_paths_from_folder

# ---- SUA 2 DUONG DAN NAY NEU KHAC VOI may ban ----
DATASET_DIR = os.path.join(_SRC_DIR, "Data")   # mac dinh giong run_train.py
CKPT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints", "last_model.pth")

paths = resolve_dataset_paths_from_folder(DATASET_DIR)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = MiniPPEDetector(num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

val_ds = PPEDetectionDataset(
    paths["val_img_dir"], paths["val_label_dir"], img_size=ckpt.get("img_size", 640)
)
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

with torch.no_grad():
    images, gt_boxes, gt_classes = next(iter(val_loader))
    preds = model(images)
    results = decode_and_filter(preds, conf_threshold=0.0, nms_iou_threshold=0.5)
    for b, (boxes, scores, classes) in enumerate(results):
        if len(scores) == 0:
            print(f"Anh {b}: khong co box nao (bat thuong, kiem tra lai decode)")
            continue
        print(f"Anh {b}: max_score={scores.max().item():.4f}  "
              f"mean={scores.mean().item():.4f}  n_box={len(scores)}")