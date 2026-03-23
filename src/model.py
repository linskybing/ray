"""Model loader for the inference pipeline.

Uses torchvision ResNet-152 to generate real GPU workload across all SMs.
The model is split into `features` (conv backbone) and `classifier` (fc head)
so that the pipeline can exercise different portions independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

INPUT_CHANNELS: Final[int] = 3
INPUT_SIZE: Final[int] = 224
NUM_CLASSES: Final[int] = 1000
DEFAULT_BATCH_SIZE: Final[int] = 128
FEATURE_DIM: Final[int] = 2048  # ResNet-152 outputs 2048-d features


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the inference model."""

    input_channels: int = INPUT_CHANNELS
    input_size: int = INPUT_SIZE
    num_classes: int = NUM_CLASSES
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "cuda"
    enable_torch_compile: bool = False


class InferenceModel(nn.Module):
    """ResNet-152 wrapper split into features + classifier for staged inference.

    ``features`` runs the heavy conv backbone (stages 2).
    ``classifier`` runs the final FC layer (stage 3).
    Both are kept as separate ``nn.Sequential`` blocks so the pipeline
    can call them independently.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        try:
            from torchvision import models
            from torchvision.models import ResNet152_Weights

            base = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        except ImportError:
            from torchvision import models

            base = models.resnet152(pretrained=False)

        # features: everything up to (and including) avgpool
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

        in_features = base.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)

        # Copy pretrained FC weights when available
        with torch.no_grad():
            self.classifier.weight.copy_(base.fc.weight)
            self.classifier.bias.copy_(base.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


def load_model(
    config: ModelConfig | None = None,
) -> tuple[InferenceModel, torch.device]:
    """Load the inference model onto the target device.

    Enables ``cudnn.benchmark`` for optimal kernel auto-tuning with
    fixed-size inputs (constant batch size across iterations).

    Returns:
        ``(model, device)`` tuple — model is in eval mode, ready for inference.
    """
    config = config or ModelConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = InferenceModel(num_classes=config.num_classes).to(device)
    model.eval()

    # torch.compile can improve steady-state throughput but may add heavy
    # shape-specialization overhead during startup.
    if device.type == "cuda" and config.enable_torch_compile:
        try:
            model.features = torch.compile(
                model.features, mode="reduce-overhead"
            )
            logger.info("torch.compile applied to model.features")
        except Exception:
            logger.warning(
                "torch.compile failed, falling back to eager mode",
                exc_info=True,
            )

    return model, device
