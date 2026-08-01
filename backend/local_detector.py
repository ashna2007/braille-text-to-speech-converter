"""Local YOLO bounding-box detection for Braille cells."""

from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_CACHE = PROJECT_ROOT / ".cache" / "ultralytics"
ULTRALYTICS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CACHE))

from ultralytics import YOLO

from backend.detection_types import BrailleDetection


LOCAL_DETECTOR_PATH = PROJECT_ROOT / "models" / "yolo11_best.pt"

_INFERENCE_LOCK = Lock()


def _validate_detector_file() -> None:
    if not LOCAL_DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing local YOLO model file: {LOCAL_DETECTOR_PATH}"
        )

    if LOCAL_DETECTOR_PATH.stat().st_size < 1024:
        content = LOCAL_DETECTOR_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if content.startswith("version https://git-lfs"):
            raise RuntimeError(
                f"{LOCAL_DETECTOR_PATH.name} is a Git LFS pointer. "
                "Run `git lfs pull`."
            )


def _yolo_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _detector() -> YOLO:
    """Load local YOLO weights only when local detection is first selected."""

    _validate_detector_file()
    return YOLO(str(LOCAL_DETECTOR_PATH))


def detect_braille(
    image: Image.Image,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 500,
) -> list[BrailleDetection]:
    """Return local YOLO detections using the shared corner-box contract."""

    if not isinstance(image, Image.Image):
        raise TypeError("detect_braille expects a PIL.Image.Image.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    with _INFERENCE_LOCK:
        result = _detector().predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            device=_yolo_device(),
            agnostic_nms=True,
            verbose=False,
        )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    detections: list[BrailleDetection] = []

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = (float(value) for value in box)
        x1 = max(0.0, min(float(image.width), x1))
        y1 = max(0.0, min(float(image.height), y1))
        x2 = max(0.0, min(float(image.width), x2))
        y2 = max(0.0, min(float(image.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(score),
            }
        )

    return detections
