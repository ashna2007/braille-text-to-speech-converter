"""Roboflow-hosted bounding-box detection for Braille cells."""

from functools import lru_cache
import math
import os
from pathlib import Path
import re
from typing import Any, TypedDict

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_MODEL_ID = os.environ.get(
    "ROBOFLOW_MODEL_ID",
    "braille-detection-f0rb5/10",
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[1-9][0-9]*$")


class BrailleDetection(TypedDict):
    box: list[float]
    confidence: float


class RoboflowConfigurationError(RuntimeError):
    """Raised when required Roboflow configuration is missing or invalid."""


class RoboflowRequestError(RuntimeError):
    """Raised when the hosted inference request fails."""


class RoboflowResponseError(RuntimeError):
    """Raised when Roboflow returns an unexpected response."""


def _api_key() -> str:
    # Loading a local .env is a development convenience. Existing environment
    # variables always take precedence and the key is never logged.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RoboflowConfigurationError(
            "ROBOFLOW_API_KEY is not set. Add it to your environment or a "
            "local .env file."
        )
    return api_key


def _validate_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not _MODEL_ID_PATTERN.fullmatch(model_id):
        raise RoboflowConfigurationError(
            "Invalid Roboflow model ID. Expected the format "
            "'project-name/version'."
        )
    return model_id


def _validate_confidence_threshold(confidence_threshold: float) -> float:
    threshold = float(confidence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    return threshold


@lru_cache(maxsize=1)
def _client() -> InferenceHTTPClient:
    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=_api_key(),
    )


def _image_size(image: Image.Image) -> tuple[int, int]:
    if not isinstance(image, Image.Image):
        raise TypeError("detect_braille expects a PIL.Image.Image.")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The input image must have positive dimensions.")
    return image.width, image.height


def _number(prediction: dict[str, Any], field: str) -> float:
    value = prediction.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoboflowResponseError(
            f"Roboflow prediction is missing numeric field {field!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise RoboflowResponseError(
            f"Roboflow prediction field {field!r} is not finite."
        )
    return number


def _normalise_prediction(
    prediction: dict[str, Any],
    image_width: int,
    image_height: int,
) -> BrailleDetection | None:
    confidence = _number(prediction, "confidence")
    center_x = _number(prediction, "x")
    center_y = _number(prediction, "y")
    width = _number(prediction, "width")
    height = _number(prediction, "height")

    if width <= 0 or height <= 0:
        return None

    x1 = max(0.0, min(float(image_width), center_x - width / 2.0))
    y1 = max(0.0, min(float(image_height), center_y - height / 2.0))
    x2 = max(0.0, min(float(image_width), center_x + width / 2.0))
    y2 = max(0.0, min(float(image_height), center_y + height / 2.0))

    if x2 <= x1 or y2 <= y1:
        return None

    # Roboflow class labels are deliberately omitted. Classification remains
    # the responsibility of the local EfficientNet/ResNet stage.
    return {
        "box": [x1, y1, x2, y2],
        "confidence": confidence,
    }


def detect_braille(
    image: Image.Image,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[BrailleDetection]:
    """Locate Braille cells with Roboflow and return clipped corner boxes.

    An empty Roboflow prediction list is a valid result and returns ``[]``.
    Network, authorization, or hosted-model failures raise
    :class:`RoboflowRequestError` without exposing the API key.
    """

    image_width, image_height = _image_size(image)
    threshold = _validate_confidence_threshold(confidence_threshold)
    model_id = _validate_model_id(ROBOFLOW_MODEL_ID)

    try:
        response = _client().infer(image, model_id=model_id)
    except Exception:
        raise RoboflowRequestError(
            "Roboflow inference failed. Check your network connection, API "
            f"key permissions, and model ID ({model_id})."
        ) from None

    if not isinstance(response, dict):
        raise RoboflowResponseError(
            "Roboflow returned an unexpected non-object response."
        )

    raw_predictions = response.get("predictions")
    if raw_predictions is None:
        raise RoboflowResponseError(
            "Roboflow response did not contain a predictions field."
        )
    if not isinstance(raw_predictions, list):
        raise RoboflowResponseError(
            "Roboflow predictions must be returned as a list."
        )

    detections: list[BrailleDetection] = []
    for raw_prediction in raw_predictions:
        if not isinstance(raw_prediction, dict):
            raise RoboflowResponseError(
                "Roboflow returned a malformed prediction."
            )

        confidence = _number(raw_prediction, "confidence")
        if confidence < threshold:
            continue

        detection = _normalise_prediction(
            raw_prediction,
            image_width=image_width,
            image_height=image_height,
        )
        if detection is not None:
            detections.append(detection)

    return detections


def draw_braille_detections(
    image: Image.Image,
    detections: list[BrailleDetection],
) -> Image.Image:
    """Draw detector-only boxes and confidence scores on a copy of an image."""

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        label = f"{confidence:.2f}"
        label_y = max(0, int(y1) - 14)
        label_box = draw.textbbox((int(x1), label_y), label, font=font)
        draw.rectangle(
            (int(x1), int(y1), int(x2), int(y2)),
            outline="#ff4b4b",
            width=2,
        )
        draw.rectangle(label_box, fill="white")
        draw.text((int(x1), label_y), label, fill="#ff4b4b", font=font)

    return annotated
