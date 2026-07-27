from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from backend.model_loader import ModelBundle
from backend.reading_order import order_predictions


def crop_from_predicted_box(
    image: Image.Image,
    box: np.ndarray,
    padding_ratio: float = 0.03,
) -> Image.Image:
    x1, y1, x2, y2 = map(float, box)
    padding = max(x2 - x1, y2 - y1) * padding_ratio

    left = max(0, int(x1 - padding))
    top = max(0, int(y1 - padding))
    right = min(image.width, int(x2 + padding))
    bottom = min(image.height, int(y2 + padding))

    return image.crop((left, top, right, bottom))


def _annotate_image(
    image: Image.Image,
    predictions: list[dict[str, Any]],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for prediction in predictions:
        x1, y1, x2, y2 = map(int, prediction["box"])
        label = (
            f"{prediction['reading_index']}. {prediction['letter']} "
            f"cls:{prediction['classifier_confidence']:.2f} "
            f"det:{prediction['detector_confidence']:.2f}"
        )
        text_y = max(0, y1 - 14)
        text_box = draw.textbbox((x1, text_y), label, font=font)

        draw.rectangle((x1, y1, x2, y2), outline="#0068c9", width=2)
        draw.rectangle(text_box, fill="white")
        draw.text((x1, text_y), label, fill="#0068c9", font=font)

    return annotated


def run_pipeline(
    image: Image.Image,
    models: ModelBundle,
    detector_confidence: float = 0.25,
    detector_iou: float = 0.7,
    max_detections: int = 500,
) -> dict[str, Any]:
    started_at = perf_counter()
    original_image = ImageOps.exif_transpose(image).convert("RGB")

    with models.inference_lock:
        detector_result = models.detector.predict(
            source=original_image,
            imgsz=640,
            conf=detector_confidence,
            iou=detector_iou,
            max_det=max_detections,
            device=models.yolo_device,
            verbose=False,
        )[0]

        if detector_result.boxes is None or len(detector_result.boxes) == 0:
            predicted_boxes = np.empty((0, 4), dtype=float)
            detector_scores = np.empty(0, dtype=float)
        else:
            predicted_boxes = detector_result.boxes.xyxy.cpu().numpy()
            detector_scores = detector_result.boxes.conf.cpu().numpy()

        crop_tensors = [
            models.classifier_transform(
                crop_from_predicted_box(original_image, box)
            )
            for box in predicted_boxes
        ]

        if crop_tensors:
            classifier_batch = torch.stack(crop_tensors).to(
                models.torch_device
            )
            with torch.inference_mode():
                probabilities = models.classifier(classifier_batch).softmax(
                    dim=1
                )
            classifier_scores, classifier_indices = probabilities.max(dim=1)
            classifier_scores_array = classifier_scores.cpu().numpy()
            classifier_indices_array = classifier_indices.cpu().numpy()
        else:
            classifier_scores_array = np.empty(0, dtype=float)
            classifier_indices_array = np.empty(0, dtype=int)

    predictions: list[dict[str, Any]] = []
    for box, detector_score, classifier_score, classifier_index in zip(
        predicted_boxes,
        detector_scores,
        classifier_scores_array,
        classifier_indices_array,
    ):
        predictions.append(
            {
                "letter": models.idx_to_class[int(classifier_index)],
                "detector_confidence": float(detector_score),
                "classifier_confidence": float(classifier_score),
                "box": [int(value) for value in box],
            }
        )

    ordered_predictions, recognized_text = order_predictions(predictions)
    annotated_image = _annotate_image(original_image, ordered_predictions)

    return {
        "annotated_image": annotated_image,
        "predictions": ordered_predictions,
        "recognized_text": recognized_text,
        "elapsed_seconds": perf_counter() - started_at,
        "device": models.yolo_device,
    }
