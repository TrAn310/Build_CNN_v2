"""
train.py
 
BUOC 1 (Bai 10 - Training Loop):
    Vong lap training CO BAN cho detector: forward (3 scale) -> build
    targets multi-scale -> tinh loss (box+obj+cls) -> backward ->
    optimizer.step(). Ban da biet training loop tu classification roi
    nen khong nhac lai phan do -- chi giai thich phan RIENG cua
    detector (targets duoc xay dung the nao, loss gom may thanh phan).
 
    Test BAT BUOC cua buoc nay (PHAN 22 trong Prompt): OVERFIT 1 ANH.
    Neu model KHONG overfit duoc 1 anh don gian voi vai object, thi
    CHUA duoc phep dung dataset that -- loi chac chan nam o CODE,
    khong phai o DU LIEU.
 
BUOC 2 (lam sau, CHUA co trong file nay):
    Validation loop day du (decode -> NMS -> evaluator -> mAP) +
    luu best_model.pth / last_model.pth theo validation mAP.
 
LUU Y: file nay gia dinh 2 bug da duoc sua trong repo that cua ban:
    1. target_asigner.py -> doi ten thanh target_assigner.py
    2. detector.py -> chuyen "from utils.config_utils import ..."
       xuong trong khoi if __name__ == "__main__"
 
CAU TRUC THU MUC THAT (xem PHAN 36 trong Prompt) dat train.py o
Src/training/, con detector.py o Src/models/ -- 2 thu muc ANH EM.
Python KHONG tu tim module o thu muc anh em, nen phai tu them cac
thu muc do vao sys.path TRUOC KHI import. Day la ly do co doan code
sys.path ngay ben duoi, TRUOC moi import khac.
"""
 
import os
import sys
 
# Src/training/train.py -> _SRC_DIR se la .../Src
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_CURRENT_DIR)
 
# Cac file trong du an dung IMPORT PHANG (vd "from Backbone import Backbone"
# trong detector.py), nen phai them TUNG THU MUC CON chua module can dung --
# them mai "Src/" thoi la KHONG DU, vi ben trong detector.py van con
# "from Backbone import Backbone" tro toi Src/models/Backbone.py.
for _sub in ("models", "dataset", "inference", "evaluation", "utils"):
    _path = os.path.join(_SRC_DIR, _sub)
    if _path not in sys.path:
        sys.path.append(_path)
 
import torch
from torch.optim import Adam
 
from detector import MiniPPEDetector
from losses import DetectionLoss, compute_multiscale_loss
 
 
def train_one_step(model, images, gt_boxes, gt_classes, loss_fn, optimizer):
    """
    1 BUOC training duy nhat -- day la toan bo phan MOI so voi
    training loop classification ban da biet:
 
        predictions = model(images)                # 3 tensor (P3,P4,P5)
        loss = compute_multiscale_loss(...)         # tu build targets ben trong
        loss.backward(); optimizer.step()           # giong het classification
 
    Args:
        model:      MiniPPEDetector
        images:     tensor [B,3,H,W]
        gt_boxes:   list[Tensor[N_i,4]]   -- dung format dataset.py/collate.py
        gt_classes: list[Tensor[N_i]]
        loss_fn:    instance DetectionLoss (tao 1 lan, tai su dung xuyen suot)
        optimizer:  torch optimizer
 
    Returns:
        loss_dict: dict cac thanh phan loss cua tung scale (de log/theo doi)
    """
    model.train()
 
    preds = model(images)   # (pred_p3, pred_p4, pred_p5)
 
    total_loss, loss_dict = compute_multiscale_loss(
        preds, gt_boxes, gt_classes, model.num_classes, loss_fn
    )
 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
 
    return loss_dict
 
 
if __name__ == "__main__":
    # ==================================================================
    # PHAN 22 - OVERFIT 1 ANH
    # Bai test bat buoc TRUOC KHI dung dataset that.
    # ==================================================================
    torch.manual_seed(0)
 
    num_classes = 3   # Person, Helmet, Vest
    model = MiniPPEDetector(num_classes=num_classes)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = DetectionLoss()
 
    # 1 anh gia CO DINH xuyen suot qua trinh train (dac diem cua overfit test:
    # KHONG doi anh moi step, de xem model co "thuoc long" duoc 1 anh khong).
    image = torch.randn(1, 3, 640, 640)
 
    # 2 object gia, co chu dinh de roi vao 2 scale khac nhau:
    #   Person: w=200,h=300 (canh lon nhat 300 >=128) -> P5
    #   Helmet: w=40, h=40  (canh lon nhat 40  <64)   -> P3
    gt_boxes = [torch.tensor([
        [200., 150., 400., 450.],
        [500., 100., 540., 140.],
    ])]
    gt_classes = [torch.tensor([0, 1])]
 
    print("Bat dau overfit 1 anh -- neu code dung, total_loss phai giam manh.\n")
 
    history = []
    for step in range(1, 201):
        loss_dict = train_one_step(model, image, gt_boxes, gt_classes, loss_fn, optimizer)
        history.append(loss_dict["total_loss"])
 
        if step == 1 or step % 20 == 0:
            print(
                f"Step {step:3d} | total={loss_dict['total_loss']:.4f} | "
                f"box(P3/P4/P5)={loss_dict['loss_P3_box']:.3f}/{loss_dict['loss_P4_box']:.3f}/{loss_dict['loss_P5_box']:.3f} | "
                f"obj(P3/P4/P5)={loss_dict['loss_P3_obj']:.3f}/{loss_dict['loss_P4_obj']:.3f}/{loss_dict['loss_P5_obj']:.3f} | "
                f"cls(P3/P5)={loss_dict['loss_P3_cls']:.3f}/{loss_dict['loss_P5_cls']:.3f}"
            )
 
    print(f"\ntotal_loss step 1   : {history[0]:.4f}")
    print(f"total_loss step 200 : {history[-1]:.4f}")
    print("\nExpected: total_loss giam manh (ly tuong ve gan 0). Neu total_loss")
    print("KHONG giam hoac dung yen tu step 1 -> co bug o build_targets/loss/")
    print("backward -- PHAI sua truoc khi dung dataset that (dung do loi cho du lieu).")