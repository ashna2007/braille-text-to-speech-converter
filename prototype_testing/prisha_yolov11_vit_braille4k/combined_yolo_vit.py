import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16
from ultralytics import YOLO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")

DETECTOR_WEIGHTS = "yolo11_braille_detection_best.pt"
CLASSIFIER_WEIGHTS = "vit_braille_classification_best.pth"
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]  # matches ImageFolder's alphabetical class order

DETECTION_SCORE_THRESHOLD = 0.3
DETECTION_NMS_IOU_THRESHOLD = 0.3  # class-agnostic, merges duplicate boxes the model's per-class NMS misses

COCO_ROOT = Path("braille detection.coco")
COCO_SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_OUTPUT_DIR = Path("combined_outputs")

CLASSIFY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_detector():
    print(f"loading YOLO11 detector weights from {DETECTOR_WEIGHTS}")
    model = YOLO(DETECTOR_WEIGHTS)
    model.to(DEVICE)
    return model


def build_classifier():
    print("building ViT-B/16 model (pretrained on ImageNet)")
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, len(CLASS_NAMES))
    print(f"  replaced classification head for {len(CLASS_NAMES)} classes")
    print(f"loading classifier weights from {CLASSIFIER_WEIGHTS}")
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def detect_boxes(detector, image):
    result = detector.predict(
        image,
        conf=DETECTION_SCORE_THRESHOLD,
        iou=DETECTION_NMS_IOU_THRESHOLD,
        agnostic_nms=True,
        device=DEVICE,
        verbose=False,
    )[0]
    return result.boxes.xyxy.cpu()


def sort_reading_order(boxes):
    """Groups boxes into rows by y-overlap, then orders each row left-to-right."""
    heights = boxes[:, 3] - boxes[:, 1]
    row_threshold = heights.mean().item() * 0.6

    rows = []
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i, 1].item()):
        y_center = (boxes[i, 1] + boxes[i, 3]).item() / 2
        for row in rows:
            if abs(row["y_center"] - y_center) <= row_threshold:
                row["indices"].append(i)
                row["y_center"] = (row["y_center"] * (len(row["indices"]) - 1) + y_center) / len(row["indices"])
                break
        else:
            rows.append({"y_center": y_center, "indices": [i]})

    ordered_indices = []
    for row in rows:
        ordered_indices.extend(sorted(row["indices"], key=lambda i: boxes[i, 0].item()))
    return ordered_indices


@torch.no_grad()
def classify_crop(classifier, image, box):
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    crop = image.crop((x1, y1, x2, y2))
    tensor = CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    pred = classifier(tensor).argmax(dim=1).item()
    return CLASS_NAMES[pred]


def process_image(detector, classifier, image_path: Path, output_path: Path) -> str:
    print(f"loading image {image_path}")
    image = Image.open(image_path).convert("RGB")

    print("running detection...")
    boxes = detect_boxes(detector, image)
    print(f"  found {len(boxes)} boxes above score {DETECTION_SCORE_THRESHOLD}")

    if len(boxes) == 0:
        print("no boxes detected, nothing to classify")
        return ""

    order = sort_reading_order(boxes)

    print("classifying crops...")
    letters = []
    draw = ImageDraw.Draw(image)
    for i in order:
        box = boxes[i]
        letter = classify_crop(classifier, image, box)
        letters.append(letter)
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), letter, fill="red")

    text = "".join(letters)
    print(f"decoded text: {text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"saved annotated image to {output_path}")
    return text


def find_coco_images():
    for split in COCO_SPLITS:
        split_dir = COCO_ROOT / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, path


def run_single(image_path: str, output_path: str):
    detector = build_detector()
    classifier = build_classifier()
    process_image(detector, classifier, Path(image_path), Path(output_path))


def run_all(output_dir: Path = DEFAULT_OUTPUT_DIR):
    images = list(find_coco_images())
    print(f"found {len(images)} images across {COCO_SPLITS} splits under {COCO_ROOT}")

    detector = build_detector()
    classifier = build_classifier()

    results = []
    for split, image_path in images:
        print(f"--- {split}/{image_path.name} ---")
        output_path = output_dir / split / image_path.name
        text = process_image(detector, classifier, image_path, output_path)
        results.append((split, image_path.name, text))

    print("\nsummary:")
    for split, name, text in results:
        print(f"  {split}/{name}: {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect braille characters and classify each detected crop.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help=f"path to a single input image. If omitted, processes every image under {COCO_ROOT}/{{{','.join(COCO_SPLITS)}}}",
    )
    parser.add_argument("--output", default="combined_output.png", help="output path (single-image mode only)")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="output directory (batch mode only)",
    )
    args = parser.parse_args()

    if args.image_path is None:
        run_all(Path(args.output_dir))
    else:
        run_single(args.image_path, args.output)
