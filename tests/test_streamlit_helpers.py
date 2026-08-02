from unittest import TestCase
from unittest.mock import MagicMock, patch

from backend.result_formatting import predictions_markdown
from backend.text_to_speech import synthesize_speech


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

        self.assertIn(
            "| 1 | 1 | D | 0.9980 | 0.8372 "
            "| `10, 20, 30, 40` |",
            markdown,
        )

    @patch("backend.text_to_speech.PiperVoice.load")
    @patch("backend.text_to_speech._ensure_voice_model")
    def test_synthesize_speech_returns_wav_audio(
        self,
        mock_ensure_voice_model: MagicMock,
        mock_piper_load: MagicMock,
    ) -> None:
        mock_ensure_voice_model.return_value = (
            "model.onnx",
            "model.onnx.json",
        )
        voice = MagicMock()
        voice.synthesize_wav.side_effect = (
            lambda text, wav_file, **_: (
                wav_file.setnchannels(1),
                wav_file.setsampwidth(2),
                wav_file.setframerate(22050),
                wav_file.writeframes(b"fake-audio"),
            )
        )
        mock_piper_load.return_value = voice

        result = synthesize_speech("hello world")

        self.assertIsNotNone(result)
        self.assertEqual(result.mime_type, "audio/wav")
        self.assertIn(b"RIFF", result.data)
        self.assertIn(b"WAVE", result.data)
        mock_ensure_voice_model.assert_called_once()
