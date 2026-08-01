from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from backend import local_detector


class _CpuArray:
    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self):
        self.xyxy = _CpuArray([[-5, 10, 45, 60]])
        self.conf = _CpuArray([0.8])

    def __len__(self):
        return 1


class LocalDetectorTests(TestCase):
    def test_returns_clipped_boxes_without_yolo_class_predictions(self) -> None:
        detector = Mock()
        detector.predict.return_value = [Mock(boxes=_Boxes())]

        with patch.object(local_detector, "_detector", return_value=detector):
            detections = local_detector.detect_braille(
                Image.new("RGB", (40, 50)),
                confidence_threshold=0.25,
            )

        self.assertEqual(
            detections,
            [{"box": [0.0, 10.0, 40.0, 50.0], "confidence": 0.8}],
        )
        self.assertNotIn("class", detections[0])

    def test_empty_local_predictions_return_an_empty_list(self) -> None:
        detector = Mock()
        detector.predict.return_value = [Mock(boxes=None)]

        with patch.object(local_detector, "_detector", return_value=detector):
            detections = local_detector.detect_braille(
                Image.new("RGB", (40, 50))
            )

        self.assertEqual(detections, [])
