"""
metrics.py
 
Cac tien ich ho tro qua trinh TRAINING VA THEO DOI KET QUA -- KHONG
trung lap voi evaluator.py (evaluator.py da lo IoU-matching/AP/mAP roi,
file nay khong dinh nghia lai).
 
Gom 2 thu:
    AverageMeter  -- theo doi trung binh CHAY (running average) cua 1
                     dai luong qua nhieu step/batch (vd: loss trong 1
                     epoch).
    MetricsLogger -- gom metric CUA TUNG EPOCH (train_loss, val_loss,
                     precision, recall, mAP) thanh 1 lich su, luu CSV,
                     va tu dong xac dinh epoch nao la "best" DUA TREN
                     VALIDATION mAP (dung PHAN 23: "Best model dua tren
                     validation mAP, khong dua train loss don thuan").
"""
 
import csv
import os
 
 
class AverageMeter:
    """
    Vi sao can: log tung step (nhu "total_loss step 200 = 0.05" o
    train.py) chi la loss cua RIENG step do, dao dong len xuong tung
    batch. Muon biet loss TRUNG BINH CA EPOCH (on dinh hon, dung de so
    sanh giua cac epoch voi nhau) thi phai cong don roi chia so lan --
    day chinh la viec AverageMeter lam.
    """
 
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.sum = 0.0
        self.count = 0
 
    def update(self, value, n=1):
        """
        value: gia tri moi (vd: total_loss cua 1 batch)
        n:     trong so cua gia tri nay (thuong la batch_size, mac
               dinh 1 neu moi lan update ung voi 1 don vi)
        """
        self.sum += value * n
        self.count += n
 
    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0
 
 
class MetricsLogger:
    """
    Gom metric CUA TUNG EPOCH lai, dung cho 2 viec:
      1. Luu ra file CSV -- xem lai duoc lich su training sau khi train
         xong (ve bieu do loss/mAP theo epoch, v.v).
      2. Bao cho train.py biet epoch NAY co phai la "best" khong, de
         quyet dinh co ghi de best_model.pth hay khong -- dua tren
         VALIDATION mAP, dung yeu cau PHAN 23, khong dua train loss.
    """
 
    def __init__(self):
        self.history = []       # list[dict], moi phan tu la 1 epoch
        self.best_map = -1.0
        self.best_epoch = -1
 
    def log_epoch(self, epoch, train_loss, val_loss, precision, recall, map_score):
        """
        Ghi lai 1 epoch va TU DONG cap nhat best_map/best_epoch neu
        map_score cua epoch nay cao hon moi epoch truoc do.
 
        Returns:
            is_best: bool -- True neu epoch nay la epoch tot nhat TINH
                     DEN THOI DIEM NAY. train.py se dung gia tri nay de
                     quyet dinh co luu best_model.pth o epoch nay khong.
        """
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "mAP": map_score,
        }
        self.history.append(entry)
 
        is_best = map_score > self.best_map
        if is_best:
            self.best_map = map_score
            self.best_epoch = epoch
 
        return is_best
 
    def save_csv(self, path):
        """Luu toan bo lich su ra file CSV."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
 
        fieldnames = ["epoch", "train_loss", "val_loss", "precision", "recall", "mAP"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.history:
                writer.writerow(entry)
 
 
if __name__ == "__main__":
    # ---- TEST 1: AverageMeter ----
    print("=== TEST AverageMeter ===")
    meter = AverageMeter()
    for value in (10.0, 6.0, 2.0):   # gia lap loss 3 batch, cung batch_size
        meter.update(value)
    print("avg sau 3 gia tri (10,6,2):", meter.avg)
    print("Expected: 6.0 ((10+6+2)/3)")
 
    meter.reset()
    meter.update(9.0, n=2)   # batch_size=2
    meter.update(3.0, n=1)   # batch_size=1
    print("\navg co trong so batch_size (9.0 x2, 3.0 x1):", meter.avg)
    print("Expected: 7.0 ((9*2 + 3*1) / (2+1) = 21/3)")
 
    # ---- TEST 2: MetricsLogger ----
    print("\n=== TEST MetricsLogger ===")
    logger = MetricsLogger()
 
    fake_epochs = [
        (1, 50.0, 45.0, 0.40, 0.30, 0.35),
        (2, 20.0, 18.0, 0.60, 0.55, 0.58),   # mAP tang -> best moi
        (3, 15.0, 25.0, 0.50, 0.50, 0.45),   # val_loss tang nhung mAP GIAM -> KHONG phai best
        (4, 10.0, 12.0, 0.70, 0.65, 0.68),   # mAP tang lai -> best moi
    ]
 
    for epoch, tl, vl, p, r, m in fake_epochs:
        is_best = logger.log_epoch(epoch, tl, vl, p, r, m)
        print(f"Epoch {epoch}: mAP={m:.2f} -> is_best={is_best}")
 
    print(f"\nBest epoch: {logger.best_epoch} (mAP={logger.best_map:.2f})")
    print("Expected: is_best = [True, True, False, True], best_epoch=4")
    print("(Epoch 3 KHONG phai best du val_loss giam so epoch 1 -- vi mAP")
    print(" moi la tieu chi quyet dinh, dung yeu cau PHAN 23)")
 
    csv_path = "test_metrics_history.csv"
    logger.save_csv(csv_path)
    print(f"\nDa luu lich su ra {csv_path}, noi dung:")
    with open(csv_path, "r", encoding="utf-8") as f:
        print(f.read())