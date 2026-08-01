from dataclasses import dataclass


@dataclass(frozen=True)
class AudioResult:
    data: bytes
    mime_type: str


def synthesize_speech(text: str) -> AudioResult | None:
    """Return audio after the team selects a text-to-speech provider."""

    del text
    return None
