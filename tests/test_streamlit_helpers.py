from unittest import TestCase

from backend.result_formatting import predictions_markdown


class StreamlitHelperTests(TestCase):
    def test_predictions_render_as_markdown_without_dataframe_conversion(self) -> None:
        markdown = predictions_markdown(
            [
                {
                    "reading_index": 1,
                    "line": 1,
                    "letter": "D",
                    "detector_confidence": 0.83721,
                    "classifier_confidence": 0.99801,
                    "box": [10, 20, 30, 40],
                }
            ]
        )

        self.assertIn("| 1 | 1 | D | 0.8372 | 0.9980 | `10, 20, 30, 40` |", markdown)
