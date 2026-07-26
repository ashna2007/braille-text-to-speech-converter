import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

COCO_ROOT = Path("braille detection.coco")
NUM_CLASSES = 27  # 26 letters + background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")


class CocoBrailleDataset(Dataset):
    def __init__(self, split: str):
        print(f"loading '{split}' split from {COCO_ROOT / split}")
        self.split_dir = COCO_ROOT / split
        coco = json.loads((self.split_dir / "_annotations.coco.json").read_text())

        self.images = {img["id"]: img["file_name"] for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        self.annotations_by_image = {img_id: [] for img_id in self.image_ids}
        for ann in coco["annotations"]:
            if ann["category_id"] == 0:  # "Braille" supercategory placeholder
                continue
            self.annotations_by_image[ann["image_id"]].append(ann)

        print(
            f"  loaded {len(self.image_ids)} images, "
            f"{sum(len(a) for a in self.annotations_by_image.values())} annotations"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.open(self.split_dir / self.images[image_id]).convert("RGB")
        image = F.to_tensor(image)

        anns = self.annotations_by_image[image_id]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    print("building Faster R-CNN model (ResNet50-FPN backbone, pretrained on COCO)")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    print(f"  replaced box predictor head for {NUM_CLASSES} classes")
    return model


@torch.no_grad()
def evaluate(model, valid_loader) -> float:
    print("running validation...")
    model.eval()
    metric = MeanAveragePrecision(backend="faster_coco_eval")
    for batch_idx, (images, targets) in enumerate(valid_loader):
        print(f"  eval batch {batch_idx + 1}/{len(valid_loader)}")
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        preds = model(images)
        metric.update(preds, targets)

    print("  computing mAP...")
    val_map = metric.compute()["map"].item()
    print(f"  val_mAP: {val_map:.4f}")
    return val_map


def train(num_epochs: int = 30):
    print(f"starting training for {num_epochs} epochs")

    train_loader = DataLoader(
        CocoBrailleDataset("train"),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        CocoBrailleDataset("valid"),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    print("optimizer: SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)")

    milestones = [int(num_epochs * 0.6), int(num_epochs * 0.85)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print(f"lr_scheduler: MultiStepLR (milestones={milestones}, gamma=0.1)")

    best_map = -1.0
    for epoch in range(num_epochs):
        print(f"--- epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"  batch {batch_idx + 1}/{len(train_loader)} - loss: {loss.item():.4f}")

        lr_scheduler.step()

        val_map = evaluate(model, valid_loader)
        print(
            f"epoch {epoch + 1}/{num_epochs} summary - "
            f"avg_loss: {epoch_loss / len(train_loader):.4f} - val_mAP: {val_map:.4f} - "
            f"lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), "fasterrcnn_braille_detection_best.pth")
            print(f"  new best (val_mAP={val_map:.4f}), saved to fasterrcnn_braille_detection_best.pth")
        else:
            print(f"  no improvement (best val_mAP so far: {best_map:.4f})")

    print("training complete")


if __name__ == "__main__":
    train()
