"""
check_dataset.py -- Kiểm tra dataset/label/class mapping (phần H).

Chạy từ thư mục gốc dự án:
  python check_dataset.py --root Src\\Data --yaml Src\\Data\\data.yaml
"""

import argparse
import os
import re
from collections import Counter

IMG_EXT = ('.jpg', '.jpeg', '.png')
BUCKETS = [('<48 (chỉ P3)', 0, 48), ('48-96 (P3+P4)', 48, 96),
           ('96-160 (P4+P5)', 96, 160), ('>=160 (chỉ P5)', 160, 1e9)]


def read_yaml(path):
    txt = open(path, encoding='utf-8').read()
    nc = int(re.search(r'^\s*nc:\s*(\d+)', txt, re.M).group(1))
    m = re.search(r'names:\s*\[(.*?)\]', txt, re.S)
    names = [s.strip().strip("'\"") for s in m.group(1).split(',')] if m else None
    return nc, names, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--img-size', type=int, default=640)
    ap.add_argument('--check-images', type=int, default=300, help='số ảnh mỗi split để thử cv2.imread/đọc kích thước')
    args = ap.parse_args()

    nc, names, raw = read_yaml(args.yaml)
    print(f'data.yaml: nc={nc}  names({len(names) if names else None})={names}')
    if names is None or len(names) != nc:
        print('!!! names thiếu hoặc không khớp nc (parser thủ công chỉ đọc dạng names: [..] một dòng)')
    for i, n in enumerate(names or []):
        print(f'  id {i}: {n}')

    try:
        import cv2
    except ImportError:
        cv2 = None

    for split in ('train', 'valid', 'val', 'test'):
        idir = os.path.join(args.root, split, 'images')
        ldir = os.path.join(args.root, split, 'labels')
        if not os.path.isdir(idir):
            continue
        files = sorted(f for f in os.listdir(idir) if f.lower().endswith(IMG_EXT))
        print('\n' + '=' * 60 + f'\nSplit {split}: {len(files)} ảnh')
        cls_cnt = Counter()
        bucket = Counter()
        n_missing = n_empty = n_bad_cls = n_bad_coord = n_bad_fmt = 0
        tiny = 0
        for f in files:
            lp = os.path.join(ldir, os.path.splitext(f)[0] + '.txt')
            if not os.path.exists(lp):
                n_missing += 1
                continue
            lines = [l.split() for l in open(lp).read().strip().splitlines() if l.strip()]
            if not lines:
                n_empty += 1
            for p in lines:
                if len(p) != 5:
                    n_bad_fmt += 1   # segmentation polygon hoặc sai format
                    continue
                c = float(p[0])
                xc, yc, w, h = map(float, p[1:])
                if c != int(c) or not (0 <= int(c) < nc):
                    n_bad_cls += 1
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    n_bad_coord += 1
                    continue
                cls_cnt[int(c)] += 1
                s = max(w, h) * args.img_size
                if s < 8:
                    tiny += 1
                for nm, lo, hi in BUCKETS:
                    if lo <= s < hi:
                        bucket[nm] += 1
        print(f'ảnh thiếu label: {n_missing} | label rỗng: {n_empty} | dòng sai format(!=5 cột): {n_bad_fmt}')
        print(f'class id ngoài [0,{nc-1}]: {n_bad_cls} | tọa độ ngoài miền: {n_bad_coord} | box <8px: {tiny}')
        print('số instance mỗi class:')
        for c in range(nc):
            nm = names[c] if names and c < len(names) else '?'
            print(f'  {c:2d} {nm:<16} {cls_cnt.get(c, 0)}')
        print('phân bố kích thước box (max(w,h) trên ảnh 640) -> scale được gán:')
        for nm, _, _ in BUCKETS:
            print(f'  {nm:<18} {bucket.get(nm, 0)}')

        if cv2 is not None:
            bad = 0
            ratios = Counter()
            for f in files[:args.check_images]:
                im = cv2.imread(os.path.join(idir, f))
                if im is None:
                    bad += 1
                    continue
                h, w = im.shape[:2]
                ratios[round(w / h, 2)] += 1
            print(f'{min(len(files), args.check_images)} ảnh đầu: lỗi đọc={bad}')
            print(f'tỉ lệ w/h phổ biến (train resize THẲNG nên !=1.0 là bị méo): {ratios.most_common(5)}')


if __name__ == '__main__':
    main()
