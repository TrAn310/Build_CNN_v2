"""
ppe_association.py
Ghép helmet/vest với person dựa trên ROI.
"""

CLASS_PERSON = 0
CLASS_HELMET = 1
CLASS_VEST = 2


def make_head_roi(box):
    """Head ROI = 0-30% chiều cao box person."""
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + 0.30 * (y2 - y1))


def make_torso_roi(box):
    """Torso ROI = 20-70% chiều cao box person."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + 0.20 * h, x2, y1 + 0.70 * h)


def center_inside(inner_box, roi):
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def associate_ppe(dets):
    """
    dets: [N,6] = [x1,y1,x2,y2,score,class]
    return: list dict {id, box, status, has_helmet, has_vest}
    """
    if dets.shape[0] == 0:
        return []
    dets_np = dets.cpu().numpy() if hasattr(dets, 'cpu') else dets

    persons, helmets, vests = [], [], []
    for d in dets_np:
        x1, y1, x2, y2, score, cls = d
        box = [float(x1), float(y1), float(x2), float(y2)]
        cls = int(cls)
        if cls == CLASS_PERSON:
            persons.append(box)
        elif cls == CLASS_HELMET:
            helmets.append(box)
        elif cls == CLASS_VEST:
            vests.append(box)

    results = []
    for pid, p in enumerate(persons):
        head_roi = make_head_roi(p)
        torso_roi = make_torso_roi(p)
        has_helmet = any(center_inside(h, head_roi) for h in helmets)
        has_vest = any(center_inside(v, torso_roi) for v in vests)

        if not has_helmet and not has_vest:
            status = 'NO_HELMET_NO_VEST'
        elif not has_helmet:
            status = 'NO_HELMET'
        elif not has_vest:
            status = 'NO_VEST'
        else:
            status = 'SAFE'

        results.append({
            'id': pid, 'box': p, 'status': status,
            'has_helmet': has_helmet, 'has_vest': has_vest,
        })
    return results
