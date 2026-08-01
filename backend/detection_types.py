"""Shared types for interchangeable Braille-cell detectors."""

from typing import TypedDict


class BrailleDetection(TypedDict):
    box: list[float]
    confidence: float
