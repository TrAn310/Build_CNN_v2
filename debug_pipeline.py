"""
debug_pipeline.py  --  Chẩn đoán độc lập pipeline inference (ảnh tĩnh + frame video).

KHÔNG sửa code dự án. Chỉ ĐỌC checkpoint, chạy model, in thống kê từng giai đoạn.

Chạy từ thư mục gốc dự án (thư mục chứa Src/), ví dụ:

  # 1) Một ảnh tĩnh
  python debug_pipeline.py --weights Src\\outputs\\checkpoints\\best_model.pth --image test.jpg

  # 2) Một frame cụ thể của video (so sánh đường ảnh vs đường video trên CÙNG frame)
  python debug_pipeline.py --weights Src\\outputs\\checkpoints\\best_model.pth --video clip.mp4 --frame-idx 120

  # 3) Quét 20 frame đều nhau để xem video có thật sự "không ra gì" không
  python debug_pipeline.py --weights Src\\outputs\\checkpoints\\best_model.pth --video clip.mp4 --scan 20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

THRESHOLDS = (0.01, 0.05, 0.1, 0.25, 0.5)


# ----------------------------------------------------------------------
# Import project (thử Src rồi src)
# ----------------------------------------------------------------------
def import_project(root):
    sys.path.insert(0, os.path.abspath(root))
    for pkg in ('Src', 'src'):
        try:
            det = __import__(f'{pkg}.models.detector', fromlist=['CustomPPEDetector'])
            post = __import__(f'{pkg}.inference.postprocess', fromlist=['postprocess', 'nms'])
            print(f'[import] OK với package "{pkg}"')
            return det.CustomPPEDetector, post
        except Exception as e:  # noqa
            print(f'[import] package "{pkg}" lỗi: {type(e).__name__}: {e}')
    raise SystemExit('Không import được project. Chạy từ thư mục gốc chứa Src/.')


# ----------------------------------------------------------------------
# A. Load checkpoint — KHÔNG nuốt lỗi
# ----------------------------------------------------------------------
def load_model(CustomPPEDetector, weights, nc, width_mult, device):
    print('\n' + '=' * 70 + '\n[A] CHECKPOINT\n' + '=' * 70)
    weights = os.path.abspath(weights)
    if not os.path.isfile(weights):
        raise SystemExit(f'Không thấy checkpoint: {weights}')
    st = os.stat(weights)
    import time
    print(f'file     : {weights}')
    print(f'size     : {st.st_size / 1e6:.2f} MB')
    print(f'modified : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}')

    ckpt = torch.load(weights, map_location='cpu')
    sd = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                print(f'unwrap key "{k}"')
                break
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    print(f'số tensor trong checkpoint: {len(sd)}')

    cls_out = [tuple(v.shape)[0] for k, v in sd.items() if k.endswith('pred_cls.weight')]
    print(f'num_classes suy ra từ checkpoint (pred_cls.weight): {cls_out}')
    if nc is None:
        nc = cls_out[0]
    elif cls_out and nc != cls_out[0]:
        print(f'!!! --nc={nc} KHÁC checkpoint ({cls_out[0]}).')

    model = CustomPPEDetector(num_classes=nc, width_mult=width_mult)
    msd = model.state_dict()
    missing = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]
    mism = [(k, tuple(sd[k].shape), tuple(msd[k].shape))
            for k in msd if k in sd and sd[k].shape != msd[k].shape]
    print(f'missing keys   : {len(missing)}  {missing[:5]}')
    print(f'unexpected keys: {len(unexpected)}  {unexpected[:5]}')
    print(f'shape mismatch : {len(mism)}  {mism[:5]}')
    if missing or unexpected or mism:
        raise SystemExit('Checkpoint KHÔNG khớp kiến trúc (kiểm tra --nc / --width-mult / phiên bản code).')
    model.load_state_dict(sd, strict=True)
    print('load_state_dict(strict=True): OK')

    nbt = [int(v.item()) for k, v in sd.items() if k.endswith('num_batches_tracked')]
    if nbt:
        print(f'BatchNorm num_batches_tracked max = {max(nbt)}  '
              f'(~số iteration đã train; =0 nghĩa là checkpoint CHƯA train)')
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    print(f'tensor có NaN/Inf: {bad if bad else "không"}')

    model.to(device).eval()
    print(f'device={device}  training_mode={model.training}  strides={model.strides}  nc={nc}')
    for n in ('head_p3', 'head_p4', 'head_p5'):
        b = getattr(model, n).pred_box.bias.detach().cpu().tolist()
        print(f'  {n}.pred_box.bias = {[round(x, 3) for x in b]}  (obj bias = {b[4]:.3f})')
    return model, nc


# ----------------------------------------------------------------------
# B. Tiền xử lý
# ----------------------------------------------------------------------
def prep_stretch(bgr, size):
    """GIỐNG train (dataset.py): BGR->RGB, resize thẳng, /255, CHW."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    t = torch.from_numpy(rgb.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.contiguous()


def prep_letterbox(bgr, size):
    """GIỐNG Src/predict.py: letterbox giữ tỉ lệ."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    t = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t.contiguous()


# ----------------------------------------------------------------------
# C. Thống kê raw output
# ----------------------------------------------------------------------
def raw_stats(raw, strides, nc):
    print('\n--- [C] RAW OUTPUT ---')
    for i, (p, s) in enumerate(zip(raw, strides)):
        p = p[0].float().cpu()
        C, H, W = p.shape
        print(f'\nP{i+3}  stride={s}  shape={tuple(p.shape)}  (kỳ vọng C={5+nc}, HxW={640//s})')
        print(f'  NaN={torch.isnan(p).any().item()}  Inf={torch.isinf(p).any().item()}')
        for c, name in enumerate(['tx', 'ty', 'tw', 'th', 'obj']):
            ch = p[c]
            print(f'  ch{c} {name:<4} min={ch.min():+8.3f} max={ch.max():+8.3f} mean={ch.mean():+8.3f}')
        obj_l = p[4]
        obj_p = torch.sigmoid(obj_l)
        print(f'  obj sigmoid: max={obj_p.max():.4f} mean={obj_p.mean():.5f}')
        print('  #cell obj>thr: ' + '  '.join(f'>{t}:{int((obj_p > t).sum())}' for t in THRESHOLDS))
        cl = p[5:]
        cp = torch.softmax(cl, 0)
        cmax, cidx = cp.max(0)
        print(f'  cls logits   : min={cl.min():+.3f} max={cl.max():+.3f} mean={cl.mean():+.3f}')
        print(f'  cls softmax  : max-prob mean={cmax.mean():.3f}  max={cmax.max():.3f}')
        print(f'  cls sigmoid  : max={torch.sigmoid(cl).max():.3f}  (nếu loss dùng BCE, đây mới là xác suất)')
        top = obj_p > 0.25
        if top.sum() > 0:
            hist = torch.bincount(cidx[top], minlength=nc).tolist()
            print(f'  argmax class tại cell obj>0.25: {hist}')
        else:
            print('  argmax class tại cell obj>0.25: (không có cell nào)')


# ----------------------------------------------------------------------
# D/E. Decode từng giai đoạn (bản sao logic decoder.py để đếm)
# ----------------------------------------------------------------------
def staged_decode(raw, strides, nc, img_size, conf, obj_thr, post, nms_iou, topk=10, min_box=2):
    print('\n--- [D/E] DECODE + LỌC TỪNG GIAI ĐOẠN ---')
    rows = []
    for si, (pred, stride) in enumerate(zip(raw, strides)):
        p = pred[0].float()
        _, H, W = p.shape
        gy, gx = torch.meshgrid(torch.arange(H, device=p.device),
                                torch.arange(W, device=p.device), indexing='ij')
        gx, gy = gx.float(), gy.float()
        xc = (gx + torch.sigmoid(p[0])) * stride
        yc = (gy + torch.sigmoid(p[1])) * stride
        w = torch.exp(p[2].clamp(-10, 10)) * stride
        h = torch.exp(p[3].clamp(-10, 10)) * stride
        obj = torch.sigmoid(p[4])
        cp = torch.softmax(p[5:], 0)
        cmax, cidx = cp.max(0)
        score = obj * cmax
        cols = torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2,
                            obj, cmax, score, cidx.float(),
                            torch.full_like(obj, si), gy, gx], 0)
        rows.append(cols.reshape(11, -1).t())
    a = torch.cat(rows, 0).cpu()   # [N, 11]: x1 y1 x2 y2 obj cmax score cls scale gy gx
    n_raw = a.shape[0]
    print(f'1. prediction thô (mọi cell, mọi scale)     : {n_raw}')
    print('   #obj>thr   : ' + '  '.join(f'>{t}:{int((a[:, 4] > t).sum())}' for t in THRESHOLDS))
    print('   #score>thr : ' + '  '.join(f'>{t}:{int((a[:, 6] > t).sum())}' for t in THRESHOLDS))
    print(f'   max obj={a[:, 4].max():.4f}  max cls-prob={a[:, 5].max():.4f}  max score={a[:, 6].max():.4f}')

    m1 = a[:, 4] > obj_thr
    print(f'2. sau lọc objectness > {obj_thr}                : {int(m1.sum())}')
    m2 = m1 & (a[:, 6] > conf)
    print(f'3. sau lọc score(obj*cls) > {conf}              : {int(m2.sum())}')
    d = a[m2].clone()
    d[:, 0:4] = d[:, 0:4].clamp(0, img_size)
    valid = (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])
    d = d[valid]
    print(f'4. sau clamp + lọc box không hợp lệ          : {d.shape[0]}')
    det6 = d[:, [0, 1, 2, 3, 6, 7]]
    out = post.nms(det6, nms_iou, img_size) if det6.shape[0] else det6
    print(f'5. sau NMS(iou={nms_iou})                        : {out.shape[0]}')

    order = a[:, 6].argsort(descending=True)[:topk]
    print(f'\nTop-{topk} ứng viên theo score (KHÔNG áp ngưỡng) -> kiểm tra model có prediction hợp lý không:')
    print('   scale cell(gy,gx)  obj    clsP   score  cls   box(x1,y1,x2,y2)')
    for r in a[order]:
        print(f'   P{int(r[8])+3}   ({int(r[9]):2d},{int(r[10]):2d})     {r[4]:.3f}  {r[5]:.3f}  {r[6]:.3f}  {int(r[7]):2d}   '
              f'({r[0]:.0f},{r[1]:.0f},{r[2]:.0f},{r[3]:.0f})')
    return out, a


def run_one(name, model, tensor, nc, args, post, device):
    print('\n' + '#' * 70 + f'\n# {name}\n' + '#' * 70)
    t = tensor.to(device)
    print(f'tensor: shape={tuple(t.shape)} dtype={t.dtype} min={t.min():.3f} max={t.max():.3f} mean={t.mean():.4f}')
    with torch.inference_mode():
        raw = model(t)
    raw_stats(raw, model.strides, nc)
    staged, _ = staged_decode(raw, model.strides, nc, args.img_size, args.conf, args.obj_thresh, post, args.nms_iou)
    with torch.inference_mode():
        real = post.postprocess(raw, model.strides, nc, args.img_size, args.conf, args.nms_iou,
                                obj_thresh=args.obj_thresh)[0]
    print(f'\n6. postprocess() thật của dự án trả về          : {real.shape[0]} box')
    if real.shape[0] != staged.shape[0]:
        print('   !!! KHÁC với bản đếm từng giai đoạn -> decoder/postprocess thật có hành vi khác dự kiến')
    return raw, real


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--nc', type=int, default=None, help='mặc định: suy ra từ checkpoint')
    ap.add_argument('--width-mult', type=float, default=1.0)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--obj-thresh', type=float, default=0.5, help='mặc định cứng trong decoder.py là 0.5')
    ap.add_argument('--nms-iou', type=float, default=0.5)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image')
    ap.add_argument('--video')
    ap.add_argument('--frame-idx', type=int, default=0)
    ap.add_argument('--scan', type=int, default=0)
    ap.add_argument('--out-dir', default='debug_out')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    CustomPPEDetector, post = import_project(args.root)
    model, nc = load_model(CustomPPEDetector, args.weights, args.nc, args.width_mult, device)

    # ---------------- ảnh tĩnh ----------------
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f'cv2.imread trả None: {args.image}')
        print(f'\nẢnh {args.image}: shape={bgr.shape} dtype={bgr.dtype}')
        t1 = prep_stretch(bgr, args.img_size)
        t2 = prep_letterbox(bgr, args.img_size)
        
        print(f"stretch shape = {tuple(t1.shape)}")
        print(f"letterbox shape = {tuple(t2.shape)}")

        if t1.shape == t2.shape:
            print(f"|stretch - letterbox| max = {(t1 - t2).abs().max().item():.3f}")
        else:
            print("Hai tensor khác kích thước, không thể trừ trực tiếp.")

        run_one('ẢNH - LETTERBOX (giống Src/predict.py)', model, t2, nc, args, post, device)

        print('\n--- [F] Tính xác định: chạy lại 3 lần cùng tensor ---')
        with torch.inference_mode():
            outs = [model(t1.to(device)) for _ in range(3)]
        for k in range(1, 3):
            diff = max((a - b).abs().max().item() for a, b in zip(outs[0], outs[k]))
            print(f'  lần 1 vs lần {k+1}: max|diff| raw = {diff:.2e}')
        model.train()
        with torch.no_grad():
            tr = model(t1.to(device))
        model.eval()
        d_train = max((a - b).abs().max().item() for a, b in zip(outs[0], tr))
        print(f'  eval vs train-mode(BN dùng batch stats): max|diff| = {d_train:.3f}  '
              f'(nếu lớn, quên gọi model.eval() sẽ làm kết quả lệch mạnh)')

    # ---------------- video ----------------
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f'\nVideo {args.video}: isOpened={cap.isOpened()}')
        if not cap.isOpened():
            raise SystemExit('Không mở được video (codec/đường dẫn).')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'frames={total}  fps={cap.get(cv2.CAP_PROP_FPS):.2f}  '
              f'size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

        if args.scan > 0:
            print(f'\n--- Quét {args.scan} frame ---')
            print(' idx   mean_px  max_obj  max_score  #obj>thr  #score>conf  #final')
            idxs = np.linspace(0, max(total - 1, 0), args.scan).astype(int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if not ok or fr is None:
                    print(f'{i:5d}  đọc frame thất bại')
                    continue
                with torch.inference_mode():
                    raw = model(prep_stretch(fr, args.img_size).to(device))
                    fin = post.postprocess(raw, model.strides, nc, args.img_size, args.conf,
                                           args.nms_iou, obj_thresh=args.obj_thresh)[0]
                obj = torch.cat([torch.sigmoid(r[0, 4]).flatten() for r in raw])
                sc = torch.cat([(torch.sigmoid(r[0, 4]) * torch.softmax(r[0, 5:], 0).max(0)[0]).flatten() for r in raw])
                print(f'{i:5d}  {fr.mean():7.1f}  {obj.max():.4f}   {sc.max():.4f}    '
                      f'{int((obj > args.obj_thresh).sum()):6d}    {int((sc > args.conf).sum()):6d}      {fin.shape[0]}')

        if args.scan == 0 or args.frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise SystemExit(f'Không đọc được frame {args.frame_idx}')
            print(f'\nframe {args.frame_idx}: shape={frame.shape} dtype={frame.dtype} mean={frame.mean():.1f}')
            png = os.path.join(args.out_dir, f'frame_{args.frame_idx}.png')
            jpg = os.path.join(args.out_dir, f'frame_{args.frame_idx}.jpg')
            cv2.imwrite(png, frame)
            cv2.imwrite(jpg, frame)
            print(f'đã lưu {png} (lossless) và {jpg}')

            t_vid = prep_stretch(frame, args.img_size)             # đường video: frame trong RAM
            t_png = prep_stretch(cv2.imread(png), args.img_size)   # đường ảnh, lossless
            t_jpg = prep_stretch(cv2.imread(jpg), args.img_size)   # đường ảnh, nén JPG
            print(f'|video - png| max = {(t_vid - t_png).abs().max():.4f}  (phải = 0)')
            print(f'|video - jpg| max = {(t_vid - t_jpg).abs().max():.4f}  (nhỏ là bình thường)')
            run_one('FRAME - đường VIDEO (RAM, stretch)', model, t_vid, nc, args, post, device)
            run_one('FRAME - đường ẢNH (đọc lại từ jpg, stretch)', model, t_jpg, nc, args, post, device)
            run_one('FRAME - letterbox (như Src/predict.py)', model, prep_letterbox(frame, args.img_size), nc, args, post, device)

    print('\nXong. Gửi lại TOÀN BỘ log này.')


if __name__ == '__main__':
    main()
