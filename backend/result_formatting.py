"""Presentation-safe formatting for recognition results."""

from typing import Any


def predictions_markdown(predictions: list[dict[str, Any]]) -> str:
    """Render predictions without Streamlit's native PyArrow dataframe path."""

    rows = [
        "| Order | Line | Letter | Detector confidence | "
        "Classifier confidence | Box |",
        "|---:|---:|:---:|---:|---:|:---|",
    ]
    for prediction in predictions:
        box = ", ".join(str(value) for value in prediction["box"])
        rows.append(
            f"| {prediction['reading_index']} "
            f"| {prediction['line']} "
            f"| {prediction['letter']} "
            f"| {prediction['detector_confidence']:.4f} "
            f"| {prediction['classifier_confidence']:.4f} "
            f"| `{box}` |"
        )
    return "\n".join(rows)
