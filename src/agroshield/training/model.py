from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


CLASS_NAMES = [
    "cassava_bacterial_blight",
    "cassava_brown_streak_disease",
    "cassava_green_mottle",
    "cassava_mosaic_disease",
    "healthy",
]

NUM_CLASSES = len(CLASS_NAMES)


def create_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create the AgroShield MobileNetV3-Small classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to initialize with ImageNet pretrained weights.

    Returns:
        Configured PyTorch model.
    """

    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = None

    model = mobilenet_v3_small(weights=weights)

    # Replace the original ImageNet classifier.
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(parameter.numel() for parameter in model.parameters())

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def model_summary(model: nn.Module) -> dict[str, object]:
    """
    Return a small machine-readable summary of the model.
    """

    total_parameters, trainable_parameters = count_parameters(model)

    return {
        "architecture": "MobileNetV3-Small",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }