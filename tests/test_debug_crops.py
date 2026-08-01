import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.inference import _save_debug_crops


class DebugCropTests(unittest.TestCase):
    def test_saves_exact_crop_with_prediction_filename(self):
        crop = Image.new("RGB", (17, 23), color=(12, 34, 56))
        predictions = [
            {
                "letter": "G",
                "classifier_confidence": 0.994,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "debug_crops"
            result = _save_debug_crops([crop], predictions, output_dir)
            saved_path = output_dir / "01_G_0.99.png"

            self.assertTrue(saved_path.is_file())
            self.assertEqual(result[0]["path"], str(saved_path))
            self.assertEqual(result[0]["image"].size, crop.size)
            with Image.open(saved_path) as saved_crop:
                self.assertEqual(saved_crop.size, crop.size)
                self.assertEqual(saved_crop.getpixel((0, 0)), (12, 34, 56))


if __name__ == "__main__":
    unittest.main()
