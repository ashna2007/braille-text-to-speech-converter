from statistics import median
from typing import Any


Prediction = dict[str, Any]


def _box(prediction: Prediction) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = prediction["box"]
    return float(x1), float(y1), float(x2), float(y2)


def _vertical_center(prediction: Prediction) -> float:
    _, y1, _, y2 = _box(prediction)
    return (y1 + y2) / 2


def group_predictions_into_rows(
    predictions: list[Prediction],
    row_tolerance: float = 0.6,
) -> list[list[Prediction]]:
    if not predictions:
        return []

    heights = [max(1.0, _box(item)[3] - _box(item)[1]) for item in predictions]
    center_tolerance = median(heights) * row_tolerance
    rows: list[list[Prediction]] = []

    for prediction in sorted(predictions, key=_vertical_center):
        prediction_center = _vertical_center(prediction)
        closest_row: list[Prediction] | None = None
        closest_distance = float("inf")

        for row in rows:
            row_center = sum(_vertical_center(item) for item in row) / len(row)
            distance = abs(prediction_center - row_center)
            if distance <= center_tolerance and distance < closest_distance:
                closest_row = row
                closest_distance = distance

        if closest_row is None:
            rows.append([prediction])
        else:
            closest_row.append(prediction)

    rows.sort(
        key=lambda row: sum(_vertical_center(item) for item in row) / len(row)
    )

    for row in rows:
        row.sort(key=lambda item: _box(item)[0])

    return rows


def order_predictions(
    predictions: list[Prediction],
    word_gap_factor: float = 1.0,
) -> tuple[list[Prediction], str]:
    rows = group_predictions_into_rows(predictions)
    if not rows:
        return [], ""

    widths = [max(1.0, _box(item)[2] - _box(item)[0]) for item in predictions]
    word_gap_threshold = median(widths) * word_gap_factor
    ordered_predictions: list[Prediction] = []
    text_rows: list[str] = []
    reading_index = 0

    for row_index, row in enumerate(rows):
        text_parts: list[str] = []
        previous_box: tuple[float, float, float, float] | None = None

        for prediction in row:
            current_box = _box(prediction)
            if previous_box is not None:
                horizontal_gap = current_box[0] - previous_box[2]
                if horizontal_gap > word_gap_threshold:
                    text_parts.append(" ")

            prediction["line"] = row_index + 1
            prediction["reading_index"] = reading_index + 1
            ordered_predictions.append(prediction)
            text_parts.append(str(prediction["letter"]))

            previous_box = current_box
            reading_index += 1

        text_rows.append("".join(text_parts))

    return ordered_predictions, "\n".join(text_rows)
