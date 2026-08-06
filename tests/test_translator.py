from unittest import TestCase
from unittest.mock import MagicMock, patch

from backend.translator import braille_to_text, text_to_braille


class TranslatorCommandTests(TestCase):
    @patch("backend.translator._run_liblouis", return_value="⠁⠃⠉")
    def test_forward_translation_uses_legacy_compatible_table_list(
        self,
        run_liblouis: MagicMock,
    ) -> None:
        result = text_to_braille("abc", grade=1)

        self.assertEqual(result, "⠁⠃⠉")
        run_liblouis.assert_called_once_with(
            ["unicode.dis,en-us-g1.ctb"],
            "abc",
        )

    @patch("backend.translator._run_liblouis", return_value="the")
    def test_backward_translation_uses_legacy_compatible_table_list(
        self,
        run_liblouis: MagicMock,
    ) -> None:
        result = braille_to_text("⠮", grade=2)

        self.assertEqual(result, "the")
        run_liblouis.assert_called_once_with(
            ["-b", "unicode.dis,en-us-g2.ctb"],
            "⠮",
        )
