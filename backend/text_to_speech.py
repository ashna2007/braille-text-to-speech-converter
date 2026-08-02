import wave
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPER_CACHE_DIR = PROJECT_ROOT / ".cache" / "piper"
DEFAULT_VOICE = "en_US-lessac-low"


@dataclass(frozen=True)
class AudioResult:
    data: bytes
    mime_type: str


def _ensure_voice_model() -> tuple[Path, Path]:
    """Download the default local voice model if it has not been cached yet."""
    PIPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = PIPER_CACHE_DIR / f"{DEFAULT_VOICE}.onnx"
    config_path = PIPER_CACHE_DIR / f"{DEFAULT_VOICE}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        download_voice(DEFAULT_VOICE, PIPER_CACHE_DIR)

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Piper voice model failed to download. "
            "Check the network connection and try again."
        )

    return model_path, config_path


@lru_cache(maxsize=1)
def get_voice() -> PiperVoice:
    model_path, config_path = _ensure_voice_model()
    return PiperVoice.load(model_path, config_path=config_path)


def synthesize_speech(text: str) -> AudioResult | None:
    """Synthesize English audio for the selected text using a local Piper voice."""
    if text is None or not str(text).strip():
        return None

    voice = get_voice()
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(str(text), wav_file)

    return AudioResult(
        data=wav_buffer.getvalue(),
        mime_type="audio/wav",
    )
