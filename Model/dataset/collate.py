
import torch


def collate_fn(batch):
    """
    Args:
        batch: list cac tuple (image, boxes, classes) tu __getitem__

    Returns:
        images:  tensor [B, 3, H, W]
        boxes:   list gom B tensor [N_i, 4]
        classes: list gom B tensor [N_i]
    """
    images, boxes_list, classes_list = zip(*batch)

    images = torch.stack(images, dim=0)  # cung shape -> stack binh thuong duoc

    boxes = list(boxes_list)
    classes = list(classes_list)

    return images, boxes, classes


if __name__ == "__main__":
    # ---- TEST BANG DU LIEU GIA ----
    from torch.utils.data import DataLoader

    class FakeDataset:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            image = torch.randn(3, 640, 640)
            if idx == 0:
                boxes = torch.tensor([[10., 10., 50., 50.], [100., 100., 150., 150.]])
                classes = torch.tensor([5, 0])
            elif idx == 1:
                boxes = torch.zeros((0, 4))
                classes = torch.zeros((0,), dtype=torch.long)
            else:
                boxes = torch.tensor([[20., 20., 80., 80.]])
                classes = torch.tensor([7])
            return image, boxes, classes

    loader = DataLoader(FakeDataset(), batch_size=3, collate_fn=collate_fn)
    images, boxes, classes = next(iter(loader))

    print("Images shape:", images.shape)
    print("So anh trong boxes list:", len(boxes))
    for i, (b, c) in enumerate(zip(boxes, classes)):
        print(f"  Anh {i}: boxes.shape={tuple(b.shape)}, classes={c.tolist()}")

    print("\nExpected: 3 anh, so object lan luot la 2, 0, 1")