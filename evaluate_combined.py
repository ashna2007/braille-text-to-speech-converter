import json

import torch
from PIL import Image

from combined import (
    COCO_ROOT,
    COCO_SPLITS,
    build_classifier,
    build_detector,
    classify_crop,
    detect_boxes,
    sort_reading_order,
)

CATEGORY_ID_TO_LETTER = {i: chr(ord("A") + i - 1) for i in range(1, 27)}  # id 1='A' ... 26='Z', id 0 is the unused "Braille" supercategory

BOX_MATCH_IOU_THRESHOLD = 0.5


def load_ground_truth_boxes(split: str):
    """Returns {file_name: (boxes, labels)} with boxes as [x1, y1, x2, y2] lists, unsorted."""
    coco = json.loads((COCO_ROOT / split / "_annotations.coco.json").read_text())
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    anns_by_image = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        if ann["category_id"] == 0:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    ground_truth = {}
    for image_id, anns in anns_by_image.items():
        if not anns:
            continue
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(CATEGORY_ID_TO_LETTER[ann["category_id"]])
        ground_truth[images[image_id]] = (boxes, labels)

    return ground_truth


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=BOX_MATCH_IOU_THRESHOLD):
    """Greedy best-IoU-first matching. Returns (matches, false_positive_idxs, false_negative_idxs)."""
    candidates = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gbox in enumerate(gt_boxes):
            iou = box_iou(pbox, gbox)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_pred, matched_gt = set(), set()
    matches = []
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi))

    false_positives = [pi for pi in range(len(pred_boxes)) if pi not in matched_pred]
    false_negatives = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    return matches, false_positives, false_negatives


def main():
    detector = build_detector()
    classifier = build_classifier()

    total_edits = 0
    total_chars = 0
    exact_matches = 0
    total_images = 0

    total_gt_boxes = 0
    total_pred_boxes = 0
    total_matched_boxes = 0
    total_correct_given_match = 0

    for split in COCO_SPLITS:
        split_dir = COCO_ROOT / split
        if not split_dir.is_dir():
            continue

        print(f"--- evaluating {split} split ---")
        ground_truth = load_ground_truth_boxes(split)

        for file_name, (gt_boxes, gt_labels) in ground_truth.items():
            image_path = split_dir / file_name
            image = Image.open(image_path).convert("RGB")

            pred_boxes_tensor = detect_boxes(detector, image)
            pred_boxes = pred_boxes_tensor.tolist()
            pred_labels = [classify_crop(classifier, image, box) for box in pred_boxes_tensor]

            # reading-order string, for the CER/exact-match view from before
            gt_order = sort_reading_order(torch.tensor(gt_boxes, dtype=torch.float32))
            expected = "".join(gt_labels[i] for i in gt_order)
            if len(pred_boxes) == 0:
                predicted = ""
            else:
                pred_order = sort_reading_order(pred_boxes_tensor)
                predicted = "".join(pred_labels[i] for i in pred_order)

            edits = edit_distance(expected, predicted)
            cer = edits / len(expected)
            is_exact = predicted == expected
            total_edits += edits
            total_chars += len(expected)
            exact_matches += int(is_exact)
            total_images += 1

            # box-level localization vs classification split
            matches, false_positives, false_negatives = match_boxes(pred_boxes, gt_boxes)
            correct_given_match = sum(1 for pi, gi in matches if pred_labels[pi] == gt_labels[gi])

            total_gt_boxes += len(gt_boxes)
            total_pred_boxes += len(pred_boxes)
            total_matched_boxes += len(matches)
            total_correct_given_match += correct_given_match

            status = "OK" if is_exact else "MISMATCH"
            print(f"  [{status}] {file_name}")
            print(f"    expected:  {expected}")
            print(f"    predicted: {predicted}")
            print(f"    CER: {cer:.3f}")
            print(
                f"    boxes: gt={len(gt_boxes)} pred={len(pred_boxes)} matched={len(matches)} "
                f"missed={len(false_negatives)} extra={len(false_positives)} "
                f"| classification given match: {correct_given_match}/{len(matches)}"
            )

    print("\n=== overall: end-to-end text decoding ===")
    print(f"images: {total_images}")
    print(f"exact match accuracy: {exact_matches / total_images:.4f}")
    print(f"overall character error rate: {total_edits / total_chars:.4f}")

    print("\n=== overall: detector localization (IoU >= {:.1f}) ===".format(BOX_MATCH_IOU_THRESHOLD))
    print(f"ground-truth boxes: {total_gt_boxes}")
    print(f"predicted boxes: {total_pred_boxes}")
    print(f"recall (matched / gt): {total_matched_boxes / total_gt_boxes:.4f}")
    print(f"precision (matched / pred): {total_matched_boxes / total_pred_boxes:.4f}")

    print("\n=== overall: classifier accuracy given a correctly localized box ===")
    print(f"matched boxes: {total_matched_boxes}")
    print(f"correct classification: {total_correct_given_match}")
    print(f"accuracy: {total_correct_given_match / total_matched_boxes:.4f}")


if __name__ == "__main__":
    main()
