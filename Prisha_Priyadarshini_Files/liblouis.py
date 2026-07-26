import os
import string
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from ultralytics import YOLO

LIBLOUIS_HOME = Path(os.environ.get("LIBLOUIS_HOME", Path(__file__).parent / "liblouis-bin"))
LOU_TRANSLATE = LIBLOUIS_HOME / "bin" / "lou_translate.exe"
TABLE_DIR = LIBLOUIS_HOME / "share" / "liblouis" / "tables"

DISPLAY_TABLE = "unicode.dis"
GRADE1_TABLE = "en-us-g1.ctb"  # uncontracted: one braille cell per letter
GRADE2_TABLE = "en-us-g2.ctb"  # contracted: whole-word/whole-syllable signs, numbers, punctuation

SPACE_CELL = "⠀"  # blank braille cell (word separator)


WEIGHTS_DIR = Path(__file__).parent / "yoonha_yolo11_efficientnet_weights_30_40_50"
DETECTOR_WEIGHTS = WEIGHTS_DIR / "yolo11" / "best.pt"
CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "efficientnet_b0" / "best_model.pt"
CLASS_NAMES = [chr(ord("A") + i) for i in range(26)]  # matches ImageFolder's alphabetical class order

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTION_SCORE_THRESHOLD = 0.3
DETECTION_NMS_IOU_THRESHOLD = 0.3

# full multi-cell scene images, checked in as a sibling of this repo rather than inside it
COCO_ROOT = Path(__file__).parent.parent.parent / "braille detection.coco"
COCO_SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

CLASSIFY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _run(args: list[str], text: str) -> str:
    env = {**os.environ, "LOUIS_TABLEPATH": str(TABLE_DIR)}
    result = subprocess.run(
        [str(LOU_TRANSLATE), *args],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=True,
    )
    return result.stdout


def text_to_braille(text: str, table: str = GRADE2_TABLE) -> str:
    return _run(["-d", DISPLAY_TABLE, table], text)


def braille_to_text(braille: str, table: str = GRADE2_TABLE) -> str:
    return _run(["-b", "-d", DISPLAY_TABLE, table], braille)


# Built once from the grade-1 table instead of hand-transcribed, so it can't drift
# from whatever liblouis actually considers the letter->cell mapping to be.
_LETTER_TO_CELL = dict(zip(string.ascii_uppercase, text_to_braille(string.ascii_lowercase, GRADE1_TABLE)))


def letters_to_braille_cells(letters: str) -> str:
    return "".join(_LETTER_TO_CELL[c] if c != " " else SPACE_CELL for c in letters.upper())


def build_detector() -> YOLO:
    print(f"loading yolo11 detector weights from {DETECTOR_WEIGHTS}")
    model = YOLO(str(DETECTOR_WEIGHTS))
    model.to(DEVICE)
    return model


def build_classifier() -> nn.Module:
    print("building efficientnet_b0 model")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    print(f"  replaced classification head for {len(CLASS_NAMES)} classes")
    print(f"loading classifier weights from {CLASSIFIER_WEIGHTS}")
    model.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def detect_boxes(detector: YOLO, image: Image.Image):
    result = detector.predict(
        image,
        conf=DETECTION_SCORE_THRESHOLD,
        iou=DETECTION_NMS_IOU_THRESHOLD,
        agnostic_nms=True,
        device=DEVICE,
        verbose=False,
    )[0]
    return result.boxes.xyxy.cpu()


def sort_reading_order(boxes) -> list[int]:
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
def classify_crop(classifier: nn.Module, image: Image.Image, box) -> str:
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    crop = image.crop((x1, y1, x2, y2))
    tensor = CLASSIFY_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    pred = classifier(tensor).argmax(dim=1).item()
    return CLASS_NAMES[pred]


def image_to_letters(detector: YOLO, classifier: nn.Module, image_path: Path | str) -> str:
    image = Image.open(image_path).convert("RGB")

    boxes = detect_boxes(detector, image)
    if len(boxes) == 0:
        return ""

    order = sort_reading_order(boxes)
    return "".join(classify_crop(classifier, image, boxes[i]) for i in order)


def image_to_text(detector: YOLO, classifier: nn.Module, image_path: Path | str, table: str = GRADE2_TABLE) -> str:
    letters = image_to_letters(detector, classifier, image_path)
    cells = letters_to_braille_cells(letters)
    return braille_to_text(cells, table)


def find_coco_images():
    for split in COCO_SPLITS:
        split_dir = COCO_ROOT / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, path


def run_all():
    images = list(find_coco_images())
    print(f"found {len(images)} images across {COCO_SPLITS} splits under {COCO_ROOT}")

    detector = build_detector()
    classifier = build_classifier()

    results = []
    for i, (split, image_path) in enumerate(images, start=1):
        print(f"--- [{i}/{len(images)}] {split}/{image_path.name} ---")
        text = image_to_text(detector, classifier, image_path)
        print(f"  text: {text!r}")
        results.append((split, image_path.name, text))

    print("\nsummary:")
    for split, name, text in results:
        print(f"  {split}/{name}: {text!r}")


def run_single(image_path: str):
    detector = build_detector()
    classifier = build_classifier()

    letters = image_to_letters(detector, classifier, image_path)
    cells = letters_to_braille_cells(letters)
    text = braille_to_text(cells)

    print(f"image:    {image_path}")
    print(f"letters:  {letters}")
    print(f"braille:  {cells}")
    print(f"text (grade 2): {text!r}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print braille cells

    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_all()
