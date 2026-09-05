from iou import compute_iou 

def nms(boxes, scores, iou_threshold=0.5):
    indices = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]

        remaining = []
        for idx in indices:
            iou = compute_iou(boxes[current], boxes[idx])
            if iou <= iou_threshold:
                remaining.append(idx)
        indices = remaining
    return keep

if __name__ == "__main__":
    boxes = [
        (100,100,200,200), #A
        (105,102,203,198), #B
        (98,105,198,205), #C
        (400,400,450,450), #D
        (110,98,210,202), #E
    ]
    scores = [0.95, 0.9, 0.85, 0.8, 0.6]
    lables = ["A", "B", "C", "D", "E"]

    kept = nms(boxes, scores, iou_threshold=0.5)
    print("Giu lai:", [lables[i] for i in kept])
