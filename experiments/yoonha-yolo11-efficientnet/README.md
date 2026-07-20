# YOLO11n + EfficientNet-B0

## Dataset

Roboflow Braille Detection Dataset

- 10,117 total images
- 26 Braille character classes (A-Z)
- 193,671 cropped character images

## Models

- Object Detection: YOLO11n, 50 epochs
- Classification: EfficientNet-B0, 50 epochs

## Pipeline

1. YOLO11n detects the location of each Braille character.
2. Each detected bounding box is cropped.
3. EfficientNet-B0 classifies each crop as a letter from A to Z.

## Results

### Object Detection

- mAP50: 96.15%
- mAP50-95: 73.96%
- Precision: 90.01%
- Recall: 95.91%

### Classification

- Accuracy: 99.41%
- Macro F1: 99.26%
- Macro Precision: 99.40%
- Macro Recall: 99.12%

### Combined Pipeline

- Character Recognition Accuracy: 97.32%
- End-to-End Precision: 79.90%
- End-to-End Recall: 97.32%
- End-to-End F1: 87.75%

The combined system correctly detected and classified 10,163 out of 10,443 ground-truth Braille characters. The high recall shows that most characters were recognized, while the lower precision was mainly caused by additional false-positive bounding boxes.
