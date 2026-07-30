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
CLASSIFIER_PATH = MODEL_DIR / "efficientnet_b0_best_model.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_to_idx.json"


@dataclass
class ModelBundle:
    classifier: nn.Module
    classifier_transform: Callable
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


def load_models() -> ModelBundle:
    _validate_model_file(CLASSIFIER_PATH)

    class_to_idx = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    idx_to_class = {
        int(class_index): class_name
        for class_name, class_index in class_to_idx.items()
    }

    torch_device = _select_device()

    classifier = models.efficientnet_b0(weights=None)
    input_features = classifier.classifier[1].in_features
    classifier.classifier[1] = nn.Linear(input_features, len(idx_to_class))

    state_dict = torch.load(
        CLASSIFIER_PATH,
        map_location=torch_device,
        weights_only=True,
    )
    classifier.load_state_dict(state_dict)
    classifier.to(torch_device)
    classifier.eval()

    classifier_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return ModelBundle(
        classifier=classifier,
        classifier_transform=classifier_transform,
        idx_to_class=idx_to_class,
        torch_device=torch_device,
    )
