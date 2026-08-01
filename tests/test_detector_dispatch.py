from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from backend.inference import _detect_braille


class DetectorDispatchTests(TestCase):
    def test_dispatches_to_roboflow(self) -> None:
        image = Image.new("RGB", (20, 20))
        expected = [{"box": [1, 2, 3, 4], "confidence": 0.9}]

        with patch(
            "backend.roboflow_detector.detect_braille",
            return_value=expected,
        ) as detector:
            actual = _detect_braille(image, "roboflow", 0.4)

        self.assertEqual(actual, expected)
        detector.assert_called_once_with(image, confidence_threshold=0.4)

    def test_dispatches_to_local_yolo(self) -> None:
        image = Image.new("RGB", (20, 20))
        expected = [{"box": [1, 2, 3, 4], "confidence": 0.8}]

        with patch(
            "backend.local_detector.detect_braille",
            return_value=expected,
        ) as detector:
            actual = _detect_braille(image, "local", 0.3)

        self.assertEqual(actual, expected)
        detector.assert_called_once_with(image, confidence_threshold=0.3)

    def test_rejects_unknown_detector_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported detector backend"):
            _detect_braille(Image.new("RGB", (20, 20)), "unknown", 0.25)
