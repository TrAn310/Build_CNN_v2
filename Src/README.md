# Custom PPE Detector

Object detector tự xây bằng PyTorch để phát hiện PPE (Person / Helmet / Vest) trên công trường.
**Không dùng YOLO, Ultralytics, Detectron2, MMDetection hay bất kỳ thư viện detection nào.**

## Kiến trúc

```
Input [B, 3, 640, 640]
  ↓
Backbone (custom CNN)
  ↓
P3 [80x80], P4 [40x40], P5 [20x20]
  ↓
Neck (FPN-like top-down fusion)
  ↓
F3, F4, F5
  ↓
3 Detection Heads (anchor-free, dense prediction)
  ↓
Raw predictions [B, 5+C, H, W]
  ↓
Decode (tx,ty,tw,th → x1,y1,x2,y2)
  ↓
Confidence threshold
  ↓
NMS tự code
  ↓
Final detections [N, 6] = [x1,y1,x2,y2,score,class]
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dataset (YOLO format)

```
data/detection/
├── images/
│   ├── train/*.jpg
│   ├── val/*.jpg
│   └── test/*.jpg
└── labels/
    ├── train/*.txt
    ├── val/*.txt
    └── test/*.txt
```

Mỗi file `.txt` chứa các dòng:
```
class x_center y_center width height
```
với tọa độ normalized [0, 1]. `class`: 0=person, 1=helmet, 2=vest.

## Training

**Bước 1: Test overfit 1 ảnh (BẮT BUỘC)**

```bash
python overfit_test.py --img path/to/img.jpg --lbl path/to/img.txt --epochs 300
```

Nếu model không overfit được 1 ảnh → debug trước, KHÔNG train dataset lớn.

**Bước 2: Train full dataset**

```bash
python train.py \
    --train_img data/detection/images/train \
    --train_lbl data/detection/labels/train \
    --val_img data/detection/images/val \
    --val_lbl data/detection/labels/val \
    --epochs 50 --batch_size 8 --lr 1e-3 --img_size 640
```

Checkpoint lưu tại `outputs/checkpoints/best_model.pth`.

## Inference

**1 ảnh:**

```bash
python predict.py \
    --weights outputs/checkpoints/best_model.pth \
    --image test.jpg \
    --out outputs/predictions/result.jpg
```

**Video:**

```bash
python video_inference.py \
    --weights outputs/checkpoints/best_model.pth \
    --source path/to/video.mp4 \
    --output outputs/videos/result.mp4 \
    --save_csv outputs/logs/violations.csv
```

**Webcam:**

```bash
python video_inference.py --weights outputs/checkpoints/best_model.pth --source 0
```

## Evaluation

```bash
python evaluate.py \
    --weights outputs/checkpoints/best_model.pth \
    --test_img data/detection/images/test \
    --test_lbl data/detection/labels/test
```

Output: precision / recall / F1 / AP / mAP từng class.

## PPE Association

Sau khi detector trả về person/helmet/vest, module `ppe_association.py` ghép PPE vào từng người:
- Head ROI: 0-30% chiều cao box person → check helmet
- Torso ROI: 20-70% chiều cao box person → check vest
- Output: `SAFE / NO_HELMET / NO_VEST / NO_HELMET_NO_VEST`

## Các bước debug bắt buộc

1. **Forward random input**
   ```python
   import torch
   from src.models.detector import CustomPPEDetector
   m = CustomPPEDetector()
   x = torch.randn(2, 3, 640, 640)
   outs = m(x)
   for o in outs: print(o.shape)  # [2,8,80,80], [2,8,40,40], [2,8,20,20]
   ```

2. **Overfit 1 ảnh** — `python overfit_test.py --img ... --lbl ...`

3. **Train 10-20 ảnh** — dùng `train.py` với subset nhỏ

4. **Train full dataset**

## Failure analysis (cần làm trong báo cáo)

Lưu các case lỗi:
- False Positive / False Negative
- Missed Helmet (vi phạm an toàn → quan trọng)
- Wrong Class / Wrong Box / Duplicate Box
- Small Object / Occlusion

## Giới hạn

- Backbone nhỏ, không mạnh bằng YOLO.
- Head anchor-free đơn giản, assignment dựa vào kích thước object.
- Chưa có tracking → identity giữa các frame không ổn định.
- Chưa có temporal smoothing → cảnh báo có thể nhấp nháy.

## Hướng phát triển

- Thêm tracking (ByteTrack đơn giản) để giữ ID person.
- Temporal smoothing (N-of-M frames).
- Domain adaptation khi chuyển camera.
- Export ONNX/TensorRT để tăng FPS.
