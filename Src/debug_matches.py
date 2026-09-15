import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

_SUBFOLDERS = ["model", os.path.join("model", "utils"), "dataset", "inference", "evaluation", "Utils"]
for _sub in _SUBFOLDERS:
    _path = os.path.normpath(os.path.join(_SRC_DIR, _sub))
    if _path not in sys.path:
        sys.path.append(_path)

import torch
from torch.utils.data import DataLoader

from detector import MiniPPEDetector
from postprocess import decode_and_filter, compute_iou
from evaluator import match_predictions_to_ground_truth, summarize
from dataset import PPEDetectionDataset
from collate import collate_fn
from config_utils import resolve_dataset_paths_from_folder

DATASET_DIR = os.path.join(_SRC_DIR, "Data")
CKPT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints", "last_model.pth")

paths = resolve_dataset_paths_from_folder(DATASET_DIR)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
class_names = ckpt["class_names"]

model = MiniPPEDetector(num_classes=ckpt["num_classes"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

val_ds = PPEDetectionDataset(paths["val_img_dir"], paths["val_label_dir"], img_size=ckpt.get("img_size", 640))
val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

with torch.no_grad():
    images, gt_boxes, gt_classes = next(iter(val_loader))
    preds = model(images)

    # DUNG DUNG threshold 0.3 nhu validate() that su dung
    results = decode_and_filter(preds, conf_threshold=0.3, nms_iou_threshold=0.5)

    for b, (boxes, scores, classes) in enumerate(results):
        gtb, gtc = gt_boxes[b], gt_classes[b]
        print(f"\n=== Anh {b} === n_pred(conf>0.3)={len(scores)}  n_gt={gtb.shape[0]}")

        if gtb.shape[0] > 0:
            gt_names = [class_names[c] for c in gtc.tolist()]
            gt_print = list(zip(gt_names, [[round(v, 1) for v in bx] for bx in gtb.tolist()]))
            print(f"  GT: {gt_print}")

        if len(scores) == 0:
            print("  -> Khong co prediction nao vuot 0.3")
            continue

        order, is_tp, num_gt = match_predictions_to_ground_truth(
            boxes, scores, classes, gtb, gtc, iou_threshold=0.5
        )
        s = summarize(is_tp, num_gt)
        print(f"  TP={s['TP']} FP={s['FP']} FN={s['FN']}")

        top_idx = torch.argsort(scores, descending=True)[:5]
        for i in top_idx.tolist():
            pred_box = boxes[i]
            pred_cls = classes[i].item()
            same_cls_mask = gtc == pred_cls
            if same_cls_mask.sum() > 0:
                best_iou = compute_iou(pred_box, gtb[same_cls_mask]).max().item()
            else:
                best_iou = -1.0  # khong co GT nao cung class de so IoU
            print(f"  pred: {class_names[pred_cls]:15s} score={scores[i]:.3f} "
                  f"box={[round(v, 1) for v in pred_box.tolist()]} "
                  f"best_iou_same_class={best_iou:.3f}")