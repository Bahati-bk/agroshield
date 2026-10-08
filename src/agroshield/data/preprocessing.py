"""
AgroShield image preprocessing pipeline.

This module defines deterministic evaluation transforms and
training-only augmentation transforms for cassava disease
classification.

Expected model input:
    [batch_size, 3, 224, 224]

Color space:
    RGB

Normalization:
    ImageNet mean/std
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
EVALUATION_RESIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------

def _to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert an image to RGB.

    This ensures every image has exactly three color channels.
    """
    return image.convert("RGB")


# ---------------------------------------------------------------------
# Training transforms
# ---------------------------------------------------------------------

def get_train_transforms() -> transforms.Compose:
    """
    Return the preprocessing and augmentation pipeline used for
    training images.

    Random augmentation is intentionally enabled here.
    """
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),

            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.80, 1.00),
                ratio=(0.90, 1.10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),

            transforms.RandomHorizontalFlip(
                p=0.50
            ),

            transforms.RandomVerticalFlip(
                p=0.20
            ),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ---------------------------------------------------------------------
# Validation transforms
# ---------------------------------------------------------------------

def get_validation_transforms() -> transforms.Compose:
    """
    Return the deterministic preprocessing pipeline used for
    validation images.

    No random augmentation is applied.
    """
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),

            transforms.Resize(
                EVALUATION_RESIZE,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),

            transforms.CenterCrop(
                IMAGE_SIZE
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ---------------------------------------------------------------------
# Test transforms
# ---------------------------------------------------------------------

def get_test_transforms() -> transforms.Compose:
    """
    Return the deterministic preprocessing pipeline used for
    test images.

    Test preprocessing is deliberately identical to validation
    preprocessing.
    """
    return get_validation_transforms()


# ---------------------------------------------------------------------
# Single-image preprocessing helper
# ---------------------------------------------------------------------

def load_and_preprocess_image(
    image_path: str | Path,
    split: str = "validation",
) -> torch.Tensor:
    """
    Load one image from disk and convert it into a model-ready tensor.

    Parameters
    ----------
    image_path:
        Path to the image file.

    split:
        One of:
            "train"
            "validation"
            "test"

    Returns
    -------
    torch.Tensor
        Tensor with shape [3, 224, 224].
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if split == "train":
        transform = get_train_transforms()

    elif split == "validation":
        transform = get_validation_transforms()

    elif split == "test":
        transform = get_test_transforms()

    else:
        raise ValueError(
            "split must be one of: "
            "'train', 'validation', 'test'"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = cast(torch.Tensor, transform(image))

    return tensor