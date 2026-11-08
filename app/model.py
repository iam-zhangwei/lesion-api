"""Model loading and inference. torch is imported lazily so tests and linting
can run without it installed."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

log = logging.getLogger("lesion-api")

# HAM10000 class order (alphabetical, as used by torchvision ImageFolder).
# If your training used a different order, save "classes" in the checkpoint.
DEFAULT_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
IMG_SIZE = 224


class LesionClassifier:
    """ResNet18 classifier. Falls back to an UNTRAINED demo model when no
    weights file is found, so the container always starts. /health says which."""

    def __init__(self, weights_path: str | None = None):
        import torch
        from torchvision import models, transforms

        self.torch = torch
        self.classes = list(DEFAULT_CLASSES)
        self.trained = False

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))

        if weights_path and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.classes = ckpt.get("classes", self.classes)
                ckpt = ckpt["state_dict"]
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
            model.load_state_dict(ckpt)
            self.trained = True
            log.info("Loaded weights from %s", weights_path)
        else:
            log.warning("No weights at %s - serving UNTRAINED demo model", weights_path)

        self.model = model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),  # lesions are centred in HAM10000
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image: Image.Image) -> dict:
        x = self.preprocess(image.convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        return {
            "label": self.classes[int(idx)],
            "confidence": float(conf),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
        }


def load_classifier() -> LesionClassifier:
    return LesionClassifier(os.getenv("MODEL_PATH", "models/model.pt"))
