"""
AgroShield preprocessing smoke test.

Tests the preprocessing pipeline using a real cassava image.
"""

from pathlib import Path

import torch
from PIL import Image

from src.agroshield.data.preprocessing import (
    get_test_transforms,
    get_train_transforms,
    get_validation_transforms,
)


IMAGE_DIR = Path(
    "data/raw/cassava/_download/train_images"
)


def get_first_image() -> Path:
    """Return the first available JPEG image."""
    images = sorted(IMAGE_DIR.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            f"No JPEG images found in {IMAGE_DIR}"
        )

    return images[0]


def inspect_tensor(name: str, tensor: torch.Tensor) -> None:
    """Print basic tensor information."""
    print(f"\n{name}")
    print("-" * 60)
    print(f"Shape:       {tuple(tensor.shape)}")
    print(f"Dtype:       {tensor.dtype}")
    print(f"Min value:   {tensor.min().item():.4f}")
    print(f"Max value:   {tensor.max().item():.4f}")
    print(f"Mean:        {tensor.mean().item():.4f}")
    print(f"Std:         {tensor.std().item():.4f}")


def main() -> None:
    print("=" * 70)
    print("AGROSHIELD — PREPROCESSING SMOKE TEST")
    print("=" * 70)

    image_path = get_first_image()

    print("\n[1] Source image")
    print("-" * 60)
    print(f"Path: {image_path}")

    with Image.open(image_path) as image:
        print(f"Original size: {image.size}")
        print(f"Original mode: {image.mode}")

        train_transform = get_train_transforms()
        validation_transform = get_validation_transforms()
        test_transform = get_test_transforms()

        train_tensor = train_transform(image)
        validation_tensor = validation_transform(image)
        test_tensor = test_transform(image)

    inspect_tensor(
        "TRAIN TRANSFORM",
        train_tensor,
    )

    inspect_tensor(
        "VALIDATION TRANSFORM",
        validation_tensor,
    )

    inspect_tensor(
        "TEST TRANSFORM",
        test_tensor,
    )

    print("\n[2] Shape validation")
    print("-" * 60)

    expected_shape = (3, 224, 224)

    assert train_tensor.shape == expected_shape
    assert validation_tensor.shape == expected_shape
    assert test_tensor.shape == expected_shape

    print("Expected shape: (3, 224, 224)")
    print("Tensor shape validation: PASSED")

    print("\n[3] Type validation")
    print("-" * 60)

    assert isinstance(train_tensor, torch.Tensor)
    assert isinstance(validation_tensor, torch.Tensor)
    assert isinstance(test_tensor, torch.Tensor)

    print("Tensor type validation: PASSED")

    print("\n[4] Channel validation")
    print("-" * 60)

    assert train_tensor.shape[0] == 3
    assert validation_tensor.shape[0] == 3
    assert test_tensor.shape[0] == 3

    print("RGB channel validation: PASSED")

    print("\n" + "=" * 70)
    print("PREPROCESSING SMOKE TEST COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()