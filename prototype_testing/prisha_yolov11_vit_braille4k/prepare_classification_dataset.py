import json
from pathlib import Path

from PIL import Image

COCO_ROOT = Path("Braille.coco")
OUTPUT_ROOT = Path("datasets/braille-classification")
SPLITS = ["train", "valid", "test"]


def crop_split(split: str):
    split_dir = COCO_ROOT / split
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    for ann in coco["annotations"]:
        class_name = categories[ann["category_id"]]
        if class_name == "Braille":  # supercategory placeholder, not a real class
            continue

        file_name = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]

        out_dir = OUTPUT_ROOT / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(split_dir / file_name) as im:
            crop = im.convert("RGB").crop((x, y, x + w, y + h))
            crop.save(out_dir / f"{ann['image_id']}_{ann['id']}.jpg")


if __name__ == "__main__":
    for split in SPLITS:
        crop_split(split)
        print(f"{split}: done")
