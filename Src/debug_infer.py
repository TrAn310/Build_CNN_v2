"""
debug_infer.py

DEBUG 1 ẢNH VALIDATION

Pipeline:
    checkpoint
        ↓
    MiniPPEDetector
        ↓
    raw P3/P4/P5
        ↓
    decode_predictions()
        ↓
    objectness × class probability
        ↓
    threshold
        ↓
    NMS
        ↓
    predictions cuối

Mục tiêu:
    Tìm xem model thực sự có dự đoán object hay không.
"""

import os
import sys
import shutil

import torch
import cv2
import numpy as np


# ============================================================
# 0. PATH SETUP
# ============================================================

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Project root:
# Build_CNN_v2/
PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_DIR = os.path.join(
    SRC_DIR,
    "model"
)

MODEL_UTILS_DIR = os.path.join(
    SRC_DIR,
    "model",
    "utils"
)

INFERENCE_DIR = os.path.join(
    SRC_DIR,
    "inference"
)

# detector.py dùng import phẳng:
# from Backbone import Backbone
# from neck import Neck
# from head import DetectionHead
#
# Vì vậy phải thêm model/ vào sys.path.

for path in [
    MODEL_DIR,
    MODEL_UTILS_DIR,
    INFERENCE_DIR,
]:
    if path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# 1. IMPORT PROJECT CODE
# ============================================================

from detector import MiniPPEDetector
from decoder import decode_predictions
from postprocess import decode_and_filter


# ============================================================
# 2. CONFIG DATASET
# ============================================================

NUM_CLASSES = 10

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]


# ============================================================
# 3. PATH
# ============================================================

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "last_model.pth"
)

DEBUG_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "checkpoints",
    "debug_model.pth"
)

IMAGE_PATH = os.path.join(
    SRC_DIR,
    "Data",
    "valid",
    "images",
    "-1079-_png_jpg.rf.19092a3937930012f9fd9c1ce57f5a7b.jpg"
)

DEBUG_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "debug_prediction.jpg"
)


# ============================================================
# 4. CONFIG INFERENCE
# ============================================================

IMG_SIZE = 640

CONF_THRESHOLD = 0.30

NMS_IOU_THRESHOLD = 0.50

DEVICE = torch.device("cpu")


# ============================================================
# 5. CHECK PATH
# ============================================================

def check_paths():

    print("\n========== CHECK PATHS ==========\n")

    print("SRC:")
    print(SRC_DIR)

    print("\nCheckpoint:")
    print(CHECKPOINT_PATH)

    print("\nImage:")
    print(IMAGE_PATH)

    print()

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            "\nKHONG TIM THAY CHECKPOINT:\n"
            + CHECKPOINT_PATH
        )

    print("✓ Checkpoint tồn tại")

    if not os.path.exists(IMAGE_PATH):

        raise FileNotFoundError(
            "\nKHONG TIM THAY ANH VALIDATION:\n"
            + IMAGE_PATH
        )

    print("✓ Ảnh validation tồn tại")


# ============================================================
# 6. LOAD CHECKPOINT
# ============================================================

def load_model():

    print("\n========== LOAD MODEL ==========\n")

    # --------------------------------------------------------
    # Copy checkpoint
    # --------------------------------------------------------

    print("Đang tạo snapshot checkpoint...")

    shutil.copy2(
        CHECKPOINT_PATH,
        DEBUG_CHECKPOINT_PATH
    )

    print(
        "Snapshot:",
        DEBUG_CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        DEBUG_CHECKPOINT_PATH,
        map_location=DEVICE
    )

    print(
        "\nCheckpoint type:",
        type(checkpoint)
    )

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        print(
            "\nCheckpoint keys:"
        )

        print(
            list(checkpoint.keys())
        )

        checkpoint_num_classes = checkpoint.get(
            "num_classes",
            None
        )

        checkpoint_class_names = checkpoint.get(
            "class_names",
            None
        )

        checkpoint_img_size = checkpoint.get(
            "img_size",
            None
        )

        print(
            "\nCheckpoint metadata:"
        )

        print(
            "num_classes:",
            checkpoint_num_classes
        )

        print(
            "img_size:",
            checkpoint_img_size
        )

        if checkpoint_class_names is not None:

            print(
                "class_names:",
                checkpoint_class_names
            )

        if checkpoint_num_classes is not None:

            if checkpoint_num_classes != NUM_CLASSES:

                raise RuntimeError(
                    "\nLOI NUM_CLASSES!\n"
                    f"Config debug = {NUM_CLASSES}\n"
                    f"Checkpoint = {checkpoint_num_classes}"
                )

    # --------------------------------------------------------
    # Extract state dict
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nKhởi tạo MiniPPEDetector..."
    )

    model = MiniPPEDetector(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing:

        print(
            "\n⚠️ Missing keys:"
        )

        for key in missing:

            print(
                "   ",
                key
            )

    if unexpected:

        print(
            "\n⚠️ Unexpected keys:"
        )

        for key in unexpected:

            print(
                "   ",
                key
            )

    if not missing and not unexpected:

        print(
            "\n✓ Load state_dict hoàn toàn khớp"
        )

    model.to(DEVICE)

    model.eval()

    print(
        "✓ Model ở chế độ eval()"
    )

    return model


# ============================================================
# 7. LOAD IMAGE
# ============================================================

def load_image():

    print("\n========== LOAD IMAGE ==========\n")

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        raise RuntimeError(
            "OpenCV không đọc được ảnh."
        )

    print(
        "Tên ảnh:",
        os.path.basename(IMAGE_PATH)
    )

    print(
        "Original shape:",
        image.shape
    )

    print(
        "Original dtype:",
        image.dtype
    )

    return image


# ============================================================
# 8. PREPROCESS
# ============================================================

def preprocess(image):

    print("\n========== PREPROCESS ==========\n")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # uint8 -> float32
    resized = (
        resized.astype(np.float32)
        / 255.0
    )

    # HWC -> CHW
    tensor = torch.from_numpy(
        resized
    ).permute(2, 0, 1)

    # CHW -> BCHW
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    print(
        "Tensor shape:",
        tensor.shape
    )

    print(
        "Tensor dtype:",
        tensor.dtype
    )

    print(
        "Tensor min:",
        tensor.min().item()
    )

    print(
        "Tensor max:",
        tensor.max().item()
    )

    print(
        "Tensor mean:",
        tensor.mean().item()
    )

    return tensor


# ============================================================
# 9. FORWARD
# ============================================================

def run_forward(
    model,
    image_tensor
):

    print("\n========== FORWARD ==========\n")

    with torch.no_grad():

        preds = model(
            image_tensor
        )

    if not isinstance(
        preds,
        (tuple, list)
    ):

        raise RuntimeError(
            "Model không trả tuple/list P3/P4/P5."
        )

    if len(preds) != 3:

        raise RuntimeError(
            f"Model trả {len(preds)} outputs thay vì 3."
        )

    pred_p3, pred_p4, pred_p5 = preds

    print(
        "P3 shape:",
        pred_p3.shape
    )

    print(
        "P4 shape:",
        pred_p4.shape
    )

    print(
        "P5 shape:",
        pred_p5.shape
    )

    expected_channels = (
        5 + NUM_CLASSES
    )

    for name, pred, expected_hw in [
        ("P3", pred_p3, 80),
        ("P4", pred_p4, 40),
        ("P5", pred_p5, 20),
    ]:

        if pred.shape[1] != expected_channels:

            raise RuntimeError(
                f"{name}: channel sai. "
                f"Expected {expected_channels}, "
                f"got {pred.shape[1]}"
            )

        if pred.shape[2] != expected_hw:

            raise RuntimeError(
                f"{name}: H sai."
            )

        if pred.shape[3] != expected_hw:

            raise RuntimeError(
                f"{name}: W sai."
            )

    print(
        "\n✓ P3/P4/P5 shape hoàn toàn đúng"
    )

    return preds


# ============================================================
# 10. RAW PREDICTION ANALYSIS
# ============================================================

def analyze_raw_predictions(preds):

    print(
        "\n========== RAW PREDICTION ANALYSIS ==========\n"
    )

    all_scores = []

    for name, pred in zip(
        ["P3", "P4", "P5"],
        preds
    ):

        print(
            f"\n----- {name} -----"
        )

        # ----------------------------------------------------
        # Raw objectness
        # ----------------------------------------------------

        obj_raw = pred[
            :, 4, :, :
        ]

        # ----------------------------------------------------
        # Raw classes
        # ----------------------------------------------------

        cls_raw = pred[
            :, 5:, :, :
        ]

        print(
            "raw objectness:"
        )

        print(
            f"  min  = {obj_raw.min().item():.6f}"
        )

        print(
            f"  max  = {obj_raw.max().item():.6f}"
        )

        print(
            f"  mean = {obj_raw.mean().item():.6f}"
        )

        # ----------------------------------------------------
        # Decode objectness
        # ----------------------------------------------------

        objectness = torch.sigmoid(
            obj_raw
        )

        print(
            "\nsigmoid(objectness):"
        )

        print(
            f"  min  = {objectness.min().item():.6f}"
        )

        print(
            f"  max  = {objectness.max().item():.6f}"
        )

        print(
            f"  mean = {objectness.mean().item():.6f}"
        )

        # ----------------------------------------------------
        # Decode classes
        # ----------------------------------------------------

        class_probs = torch.softmax(
            cls_raw,
            dim=1
        )

        max_class_prob, class_ids = torch.max(
            class_probs,
            dim=1
        )

        print(
            "\nclass probability:"
        )

        print(
            f"  max = {max_class_prob.max().item():.6f}"
        )

        print(
            f"  mean = {max_class_prob.mean().item():.6f}"
        )

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        scores = (
            objectness
            * max_class_prob
        )

        all_scores.append(
            scores.flatten()
        )

        print(
            "\nFINAL SCORE:"
        )

        print(
            f"  max  = {scores.max().item():.6f}"
        )

        print(
            f"  mean = {scores.mean().item():.6f}"
        )

        # ----------------------------------------------------
        # Threshold distribution
        # ----------------------------------------------------

        for threshold in [
            0.01,
            0.05,
            0.10,
            0.20,
            0.30,
            0.50,
            0.70,
        ]:

            count = (
                scores > threshold
            ).sum().item()

            print(
                f"  score > {threshold:.2f}: "
                f"{count}"
            )

        # ----------------------------------------------------
        # Highest scoring cell
        # ----------------------------------------------------

        flat_index = scores.flatten().argmax()

        _, _, H, W = pred.shape

        y = (
            flat_index // W
        ).item()

        x = (
            flat_index % W
        ).item()

        best_score = scores[
            0,
            y,
            x
        ].item()

        best_obj = objectness[
            0,
            y,
            x
        ].item()

        best_class_prob = max_class_prob[
            0,
            y,
            x
        ].item()

        best_class = class_ids[
            0,
            y,
            x
        ].item()

        print(
            "\nBEST CELL:"
        )

        print(
            f"  grid = ({x}, {y})"
        )

        print(
            f"  objectness = {best_obj:.6f}"
        )

        print(
            f"  class_prob = {best_class_prob:.6f}"
        )

        print(
            f"  final_score = {best_score:.6f}"
        )

        print(
            f"  class = {best_class} "
            f"({CLASS_NAMES[best_class]})"
        )

    all_scores = torch.cat(
        all_scores
    )

    print(
        "\n========== GLOBAL SCORE ==========\n"
    )

    print(
        f"Global max score  = "
        f"{all_scores.max().item():.6f}"
    )

    print(
        f"Global mean score = "
        f"{all_scores.mean().item():.6f}"
    )

    for threshold in [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
        0.70,
    ]:

        count = (
            all_scores > threshold
        ).sum().item()

        print(
            f"Global score > {threshold:.2f}: "
            f"{count}"
        )


# ============================================================
# 11. DECODE EACH SCALE
# ============================================================

def decode_predictions_debug(
    preds
):

    print(
        "\n========== DECODE ==========\n"
    )

    decoded_all = []

    scale_configs = [
        ("P3", preds[0], 8),
        ("P4", preds[1], 16),
        ("P5", preds[2], 32),
    ]

    for name, pred, stride in scale_configs:

        boxes, objectness, class_probs = decode_predictions(
            pred,
            stride=stride,
            base_w=stride,
            base_h=stride
        )

        print(
            f"\n{name}:"
        )

        print(
            "  boxes:",
            boxes.shape
        )

        print(
            "  objectness:",
            objectness.shape
        )

        print(
            "  class_probs:",
            class_probs.shape
        )

        print(
            "  box min:",
            boxes.min().item()
        )

        print(
            "  box max:",
            boxes.max().item()
        )

        decoded_all.append(
            (
                boxes,
                objectness,
                class_probs
            )
        )

    return decoded_all


# ============================================================
# 12. POSTPROCESS / NMS
# ============================================================

def run_postprocess(
    preds
):

    print(
        "\n========== POSTPROCESS / NMS ==========\n"
    )

    with torch.no_grad():

        results = decode_and_filter(
            preds,
            conf_threshold=CONF_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD
        )

    if len(results) != 1:

        raise RuntimeError(
            f"Expected batch size 1, got {len(results)}"
        )

    boxes, scores, classes = results[0]

    print(
        "Confidence threshold:",
        CONF_THRESHOLD
    )

    print(
        "NMS IoU threshold:",
        NMS_IOU_THRESHOLD
    )

    print(
        "\nFinal detections:",
        boxes.shape[0]
    )

    if boxes.shape[0] == 0:

        print(
            "\n⚠️ KHÔNG CÓ BOX SAU THRESHOLD + NMS"
        )

        return boxes, scores, classes

    print(
        "\nTop detections:"
    )

    order = torch.argsort(
        scores,
        descending=True
    )

    for rank, idx in enumerate(
        order[:20]
    ):

        idx = idx.item()

        cls_id = classes[
            idx
        ].item()

        print(
            f"{rank+1:02d}. "
            f"score={scores[idx].item():.6f} "
            f"class={cls_id} "
            f"({CLASS_NAMES[cls_id]}) "
            f"box={boxes[idx].tolist()}"
        )

    return boxes, scores, classes


# ============================================================
# 13. DRAW RESULT
# ============================================================

def draw_predictions(
    image,
    boxes,
    scores,
    classes
):

    print(
        "\n========== DRAW RESULT ==========\n"
    )

    output = image.copy()

    for box, score, cls_id in zip(
        boxes,
        scores,
        classes
    ):

        x1, y1, x2, y2 = [
            int(v)
            for v in box.tolist()
        ]

        cls_id = int(
            cls_id.item()
        )

        score = float(
            score.item()
        )

        label = (
            f"{CLASS_NAMES[cls_id]} "
            f"{score:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        DEBUG_OUTPUT_PATH,
        output
    )

    print(
        "Đã lưu ảnh debug:"
    )

    print(
        DEBUG_OUTPUT_PATH
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "\n"
        "==============================================\n"
        "       DEBUG INFERENCE - REAL IMAGE\n"
        "=============================================="
    )

    print(
        "\nDevice:",
        DEVICE
    )

    print(
        "Num classes:",
        NUM_CLASSES
    )

    print(
        "Classes:",
        CLASS_NAMES
    )

    # --------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------

    check_paths()

    # --------------------------------------------------------
    # 2. Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # 3. Image
    # --------------------------------------------------------

    image = load_image()

    # --------------------------------------------------------
    # 4. Preprocess
    # --------------------------------------------------------

    image_tensor = preprocess(
        image
    )

    # --------------------------------------------------------
    # 5. Forward
    # --------------------------------------------------------

    preds = run_forward(
        model,
        image_tensor
    )

    # --------------------------------------------------------
    # 6. Raw analysis
    # --------------------------------------------------------

    analyze_raw_predictions(
        preds
    )

    # --------------------------------------------------------
    # 7. Decode
    # --------------------------------------------------------

    decode_predictions_debug(
        preds
    )

    # --------------------------------------------------------
    # 8. Postprocess + NMS
    # --------------------------------------------------------

    boxes, scores, classes = run_postprocess(
        preds
    )

    # --------------------------------------------------------
    # 9. Draw
    # --------------------------------------------------------

    draw_predictions(
        image,
        boxes,
        scores,
        classes
    )

    print(
        "\n"
        "==============================================\n"
        "             DEBUG FINISHED\n"
        "=============================================="
    )


if __name__ == "__main__":

    main()