import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = Path("braille detection.yolov11/data.yaml")
NUM_CLASSES = 26  # 26 letters, no background class in YOLO
DEVICE = "0" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

RUNS_PROJECT = Path("runs/detect")
RUN_NAME = "braille_yolo11"
BEST_WEIGHTS_OUT = Path("yolo11_braille_detection_best.pt")


def build_model():
    print("building YOLO11 model (yolo11n, pretrained on COCO)")
    model = YOLO("yolo11n.pt")
    print(f"  will fine-tune for {NUM_CLASSES} classes using {DATA_YAML}")
    return model


def evaluate(model) -> float:
    print("running validation...")
    metrics = model.val(data=str(DATA_YAML), device=DEVICE, split="val")
    val_map = metrics.box.map
    print(f"  val_mAP50-95: {val_map:.4f}")
    return val_map


def train(num_epochs: int = 50):
    print(f"starting training for {num_epochs} epochs")

    model = build_model()

    results = model.train(
        data=str(DATA_YAML),
        epochs=num_epochs,
        batch=4,
        device=DEVICE,
        project=str(RUNS_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        save_period=-1,  # only keep last.pt + best.pt, best.pt refreshed whenever fitness improves
        val=True,
        plots=False,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"training complete - ultralytics saved best weights to {best_weights}")

    best_model = YOLO(str(best_weights))
    val_map = evaluate(best_model)
    print(f"final best model val_mAP50-95: {val_map:.4f}")

    shutil.copy(best_weights, BEST_WEIGHTS_OUT)
    print(f"copied best epoch weights to {BEST_WEIGHTS_OUT}")


if __name__ == "__main__":
    train(num_epochs=50)
