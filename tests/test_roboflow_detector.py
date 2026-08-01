import os
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from backend import roboflow_detector


class RoboflowDetectorTests(TestCase):
    def setUp(self) -> None:
        roboflow_detector._client.cache_clear()

    def test_converts_filters_and_clips_predictions(self) -> None:
        client = Mock()
        client.infer.return_value = {
            "predictions": [
                {
                    "x": 5,
                    "y": 10,
                    "width": 20,
                    "height": 30,
                    "confidence": 0.9,
                    "class": "ignored-by-design",
                },
                {
                    "x": 50,
                    "y": 50,
                    "width": 10,
                    "height": 10,
                    "confidence": 0.2,
                },
            ]
        }

        with patch.object(roboflow_detector, "_client", return_value=client):
            detections = roboflow_detector.detect_braille(
                Image.new("RGB", (100, 80)),
                confidence_threshold=0.5,
            )

        self.assertEqual(
            detections,
            [{"box": [0.0, 0.0, 15.0, 25.0], "confidence": 0.9}],
        )
        self.assertNotIn("class", detections[0])
        client.infer.assert_called_once()

    def test_empty_predictions_are_a_valid_empty_result(self) -> None:
        client = Mock()
        client.infer.return_value = {"predictions": []}

        with patch.object(roboflow_detector, "_client", return_value=client):
            detections = roboflow_detector.detect_braille(
                Image.new("RGB", (20, 20))
            )

        self.assertEqual(detections, [])

    def test_missing_api_key_has_a_clear_error(self) -> None:
        with (
            patch.object(roboflow_detector, "load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                roboflow_detector.RoboflowConfigurationError,
                "ROBOFLOW_API_KEY is not set",
            ):
                roboflow_detector._api_key()

    def test_invalid_model_id_is_rejected_before_request(self) -> None:
        client = Mock()
        with (
            patch.object(roboflow_detector, "ROBOFLOW_MODEL_ID", "invalid"),
            patch.object(roboflow_detector, "_client", return_value=client),
        ):
            with self.assertRaisesRegex(
                roboflow_detector.RoboflowConfigurationError,
                "Invalid Roboflow model ID",
            ):
                roboflow_detector.detect_braille(Image.new("RGB", (20, 20)))

        client.infer.assert_not_called()

    def test_request_error_does_not_expose_api_key(self) -> None:
        client = Mock()
        client.infer.side_effect = RuntimeError("secret-key-value")

        with patch.object(roboflow_detector, "_client", return_value=client):
            with self.assertRaises(roboflow_detector.RoboflowRequestError) as ctx:
                roboflow_detector.detect_braille(Image.new("RGB", (20, 20)))

        self.assertNotIn("secret-key-value", str(ctx.exception))

    def test_detector_annotation_returns_a_copy(self) -> None:
        source = Image.new("RGB", (40, 40), "white")
        annotated = roboflow_detector.draw_braille_detections(
            source,
            [{"box": [5.0, 5.0, 20.0, 20.0], "confidence": 0.75}],
        )

        self.assertIsNot(source, annotated)
        self.assertEqual(source.getpixel((5, 5)), (255, 255, 255))
        self.assertNotEqual(annotated.getpixel((5, 5)), (255, 255, 255))
