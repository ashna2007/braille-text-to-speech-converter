import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_B_16_Weights, vit_b_16

DATA_ROOT = "datasets/braille-classification"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model(num_classes: int):
    print("building ViT-B/16 model (pretrained on ImageNet)")
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    print(f"  replaced classification head for {num_classes} classes")
    return model


@torch.no_grad()
def evaluate(model, loader) -> float:
    print("running validation...")
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"  eval batch {batch_idx + 1}/{len(loader)}")
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    val_acc = correct / total
    print(f"  val_acc: {val_acc:.4f}")
    return val_acc


def train(num_epochs: int = 50, batch_size: int = 32, lr: float = 1e-4):
    print(f"starting training for {num_epochs} epochs (batch_size={batch_size}, lr={lr})")

    print(f"loading train split from {DATA_ROOT}/train")
    train_set = ImageFolder(f"{DATA_ROOT}/train", transform=TRANSFORM)
    print(f"  loaded {len(train_set)} images, {len(train_set.classes)} classes")

    print(f"loading valid split from {DATA_ROOT}/valid")
    valid_set = ImageFolder(f"{DATA_ROOT}/valid", transform=TRANSFORM)
    print(f"  loaded {len(valid_set)} images, {len(valid_set.classes)} classes")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = build_model(num_classes=len(train_set.classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print("optimizer: AdamW")

    best_acc = -1.0
    for epoch in range(num_epochs):
        print(f"--- epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"  batch {batch_idx + 1}/{len(train_loader)} - loss: {loss.item():.4f}")

        val_acc = evaluate(model, valid_loader)
        print(
            f"epoch {epoch + 1}/{num_epochs} summary - "
            f"avg_loss: {epoch_loss / len(train_loader):.4f} - val_acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "vit_braille_classification_best.pth")
            print(f"  new best (val_acc={val_acc:.4f}), saved to vit_braille_classification_best.pth")
        else:
            print(f"  no improvement (best val_acc so far: {best_acc:.4f})")

    print("training complete")


if __name__ == "__main__":
    train()
