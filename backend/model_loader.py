from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock
from typing import Callable

import torch
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
EFFICIENTNET_PATH = MODEL_DIR / "efficientnet_b0_best_model.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_to_idx.json"

CLASS_NAMES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


@dataclass
class ClassifierBundle:
    model: nn.Module
    transform: Callable


@dataclass
class ModelBundle:
    classifiers: dict[str, ClassifierBundle]
    idx_to_class: dict[int, str]
    torch_device: torch.device
    inference_lock: Lock = field(default_factory=Lock)


def _validate_model_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")

    if path.stat().st_size < 1024:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if content.startswith("version https://git-lfs"):
            raise RuntimeError(
                f"{path.name} is a Git LFS pointer. Run `git lfs pull`."
            )


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _evaluation_transform() -> transforms.Compose:
    """Match the EfficientNet notebook's evaluation preprocessing exactly."""

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )


def _load_class_mapping() -> dict[int, str]:
    class_to_idx = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    expected = {name: index for index, name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected:
        raise RuntimeError(
            "class_to_idx.json does not match the EfficientNet notebook's A-Z "
            "ImageFolder class order."
        )
    return {index: name for name, index in class_to_idx.items()}


def _load_efficientnet(device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(CLASS_NAMES))
    state_dict = torch.load(
        EFFICIENTNET_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_models() -> ModelBundle:
    _validate_model_file(EFFICIENTNET_PATH)
    idx_to_class = _load_class_mapping()
    torch_device = _select_device()

    return ModelBundle(
        classifiers={
            "efficientnet": ClassifierBundle(
                model=_load_efficientnet(torch_device),
                transform=_evaluation_transform(),
            ),
        },
        idx_to_class=idx_to_class,
        torch_device=torch_device,
    )
