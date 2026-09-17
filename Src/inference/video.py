"""
video.py
Video/webcam inference + FPS + logging vi phạm.
"""

import os
import csv
import time
import cv2
import torch

from .postprocess import postprocess
from ..utils.visualization import draw_boxes
from ..association.ppe_association import associate_ppe


@torch.no_grad()
def run_video(model, source, output_path=None, device='cpu',
              img_size=640, conf_thresh=0.3, iou_thresh=0.5,
              num_classes=3, save_csv=None, save_snapshots=None,
              show=True, max_frames=None):
    """
    source: đường dẫn video (str) hoặc 0 (int) cho webcam.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được source: {source}')

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[Video] {w}x{h} @ {fps_src:.1f} FPS')

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    csv_file = None
    csv_writer = None
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or '.', exist_ok=True)
        csv_file = open(save_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'person_id', 'status'])

    if save_snapshots:
        os.makedirs(save_snapshots, exist_ok=True)

    frame_idx = 0
    fps_list = []
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t0 = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = cv2.resize(img_rgb, (img_size, img_size))
        tensor = tensor.astype('float32') / 255.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

        raw = model(tensor)
        dets = postprocess(raw, model.strides, num_classes,
                           img_size, conf_thresh, iou_thresh)[0]

        # Rescale về ảnh gốc
        if dets.shape[0] > 0:
            dets[:, [0, 2]] *= w / img_size
            dets[:, [1, 3]] *= h / img_size

        # Association
        statuses = associate_ppe(dets)

        # Log vi phạm
        for s in statuses:
            if s['status'] != 'SAFE':
                if csv_writer:
                    csv_writer.writerow([frame_idx, s['id'], s['status']])
                if save_snapshots:
                    snap = os.path.join(
                        save_snapshots,
                        f'f{frame_idx:06d}_id{s["id"]}_{s["status"]}.jpg')
                    cv2.imwrite(snap, frame)

        # Vẽ
        vis = draw_boxes(img_rgb, dets)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        for s in statuses:
            x1, y1 = int(s['box'][0]), int(s['box'][1])
            color = (0, 255, 0) if s['status'] == 'SAFE' else (0, 0, 255)
            cv2.putText(vis, s['status'], (x1, max(20, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        dt = time.time() - t0
        fps_list.append(1.0 / max(dt, 1e-6))
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(vis, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer:
            writer.write(vis)
        if show:
            cv2.imshow('PPE Detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()

    avg = sum(fps_list) / max(len(fps_list), 1)
    print(f'[Video] Done. Frames={frame_idx} AvgFPS={avg:.2f}')
    return avg