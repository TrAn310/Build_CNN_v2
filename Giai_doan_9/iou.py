def compute_iou(boxA, boxB):
    xA1 , yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    # Bước 1: Tọa độ vùng giao 
    x1_inter = max (xA1, xB1)
    y1_inter = max (yA1, yB1)
    x2_inter = min (xA2, xB2) 
    y2_inter = min (yA2, yB2) 

    # Bước 2: Diện tích giao (nếu không giao nhau, width/height sẽ <0)
    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    intersection = inter_w * inter_h
    #Bước 3: tính diện tích từng box 
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1) 

    #Bước 4: Union
    union = areaA + areaB - intersection 

    #Bước 5: IoU 
    iou = intersection / union if union >0 else 0.0
    return iou 

if __name__ == "__main__":
    gt = (100,100,300,300)
    pred = (150,150,350,350)
    print(f"IoU(gt,pred)= {compute_iou(gt,pred):.3f}") 