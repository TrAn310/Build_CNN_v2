import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image, ImageDraw
import cv2
import numpy as np
from Src.models.detector import CustomPPEDetector
from Src.inference.postprocess import postprocess

model = CustomPPEDetector(num_classes=10)
model.load_state_dict(torch.load('Src/outputs/checkpoints/best_model.pth'))
model.eval().cuda()

# Load 1 ảnh test
img_path = 'Src/Data/test/images/' + os.listdir('Src/Data/test/images')[0]
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0).cuda() / 255.0

with torch.no_grad():
    raw = model(img_tensor)
    dets = postprocess(raw, model.strides, 10, 640, 0.5, 0.5)

print(f'Số box dự đoán: {dets[0].shape[0]}')
for d in dets[0][:20]:  # in 20 box đầu
    print(f'  box={d[:4].tolist()} score={d[4].item():.3f} cls={int(d[5].item())}')