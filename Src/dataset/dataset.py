# dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from parser import parse_yolo_label

class PPEDetectionDataset(Dataset):
    """
    Moi item tra ve:
        image:   tensor [3, img_size, img_size], gia tri [0,1]
        boxes:   tensor [N, 4]  (x1,y1,x2,y2) pixel TREN ANH DA RESIZE
        classes: tensor [N]     class index, tach rieng khoi boxes

    N khac nhau tuy anh -- can collate.py de gop batch.

    Tham so augment:
        True  -> bat augmentation (dung cho tap TRAIN)
        False -> tat augmentation (dung cho tap VALID/TEST, giu nguyen anh goc)
    """

    def __init__(self, img_dir, label_dir, img_size=640, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment  # MOI: True cho train, False cho valid

        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        # ---- Doc anh ----
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")

        # Phai lay kich thuoc GOC truoc khi resize -- can de doi normalize -> pixel
        orig_h, orig_w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        # LUU Y: KHONG tao image_tensor o day nua -- doi xuong duoi cung,
        # sau khi da augment xong (neu co), de tranh phai convert lai.

        # ---- Doc label bang parser.py ----
        boxes = []
        classes = []

        if os.path.exists(label_path):
            objects = parse_yolo_label(label_path)  # goi lai file parser.py

            for class_id, xc, yc, w, h in objects:
                # Buoc 1: normalize -> pixel tren ANH GOC
                xc_px = xc * orig_w
                yc_px = yc * orig_h
                w_px = w * orig_w
                h_px = h * orig_h

                # Buoc 2: scale theo ti le resize ve img_size
                scale_x = self.img_size / orig_w
                scale_y = self.img_size / orig_h

                xc_px *= scale_x
                w_px *= scale_x
                yc_px *= scale_y
                h_px *= scale_y

                # Buoc 3: center,w,h -> x1,y1,x2,y2
                x1 = xc_px - w_px / 2
                y1 = yc_px - h_px / 2
                x2 = xc_px + w_px / 2
                y2 = yc_px + h_px / 2

                boxes.append([x1, y1, x2, y2])
                classes.append(class_id)

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            classes_tensor = torch.tensor(classes, dtype=torch.long)
        else:
            # Anh khong co object (background image)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            classes_tensor = torch.zeros((0,), dtype=torch.long)

        # ============================================================
        # MOI: AUGMENTATION -- CHI ap dung khi self.augment=True (tap train)
        # Lam SAU KHI da co boxes_tensor (toa do tren anh da resize),
        # de khong phai tinh lai tu normalize.
        # ============================================================
        if self.augment:
            # ---- 1. Random horizontal flip (50%) ----
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1].copy()  # lat theo truc W, image dang (C,H,W)
                if boxes_tensor.shape[0] > 0:
                    x1 = boxes_tensor[:, 0].clone()
                    x2 = boxes_tensor[:, 2].clone()
                    boxes_tensor[:, 0] = self.img_size - x2
                    boxes_tensor[:, 2] = self.img_size - x1

            # ---- 2. Random brightness nhe (50%) ----
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0.0, 1.0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image))

        return image_tensor, boxes_tensor, classes_tensor


if __name__ == "__main__":

    # ĐƯỜNG DẪN TỚI THƯ MỤC, KHÔNG PHẢI FILE
    img_dir = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Src\Data\train\images"
    label_dir = r"F:\NCKH_2026\lapTrinhPy\CNN_v2\Src\Data\train\labels"

    # Test dataset
    dataset = PPEDetectionDataset(
        img_dir,
        label_dir,
        img_size=640,
        augment=True
    )

    print("So luong anh:", len(dataset))

    # Test 3 lần cùng một ảnh
    for i in range(3):

        image, boxes, classes = dataset[0]

        print(f"\n--- Lan goi {i+1} (index 0) ---")
        print("Image shape :", image.shape)
        print("Boxes shape :", boxes.shape)
        print("Classes     :", classes.tolist())
        print("Boxes       :", boxes.tolist())