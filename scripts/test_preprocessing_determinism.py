"""
AgroShield preprocessing determinism test.

Verifies that:
    1. Training preprocessing is stochastic.
    2. Validation preprocessing is deterministic.
    3. Test preprocessing is deterministic.
    4. All pipelines produce the expected tensor shape.
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

EXPECTED_SHAPE = (3, 224, 224)


def get_test_image() -> Path:
    """Return the first available cassava image."""
    images = sorted(IMAGE_DIR.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            f"No JPEG images found in {IMAGE_DIR}"
        )

    return images[0]


def main() -> None:
    print("=" * 70)
    print("AGROSHIELD — PREPROCESSING DETERMINISM TEST")
    print("=" * 70)

    image_path = get_test_image()

    print("\n[1] Test image")
    print("-" * 60)
    print(f"Path: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        # -------------------------------------------------------------
        # Training transform
        # -------------------------------------------------------------

        train_transform = get_train_transforms()

        train_tensor_1 = train_transform(image)
        train_tensor_2 = train_transform(image)

        # -------------------------------------------------------------
        # Validation transform
        # -------------------------------------------------------------

        validation_transform = get_validation_transforms()

        validation_tensor_1 = validation_transform(image)
        validation_tensor_2 = validation_transform(image)

        # -------------------------------------------------------------
        # Test transform
        # -------------------------------------------------------------

        test_transform = get_test_transforms()

        test_tensor_1 = test_transform(image)
        test_tensor_2 = test_transform(image)

    # -----------------------------------------------------------------
    # Training stochasticity
    # -----------------------------------------------------------------

    print("\n[2] Training stochasticity")
    print("-" * 60)

    train_identical = torch.equal(
        train_tensor_1,
        train_tensor_2,
    )

    print(
        f"Training tensors identical: {train_identical}"
    )

    if train_identical:
        raise AssertionError(
            "Training preprocessing appears deterministic. "
            "Expected random augmentation to produce different "
            "outputs for at least this image."
        )

    print("Training stochasticity: PASSED")

    # -----------------------------------------------------------------
    # Validation determinism
    # -----------------------------------------------------------------

    print("\n[3] Validation determinism")
    print("-" * 60)

    validation_identical = torch.equal(
        validation_tensor_1,
        validation_tensor_2,
    )

    print(
        f"Validation tensors identical: "
        f"{validation_identical}"
    )

    if not validation_identical:
        raise AssertionError(
            "Validation preprocessing is not deterministic."
        )

    print("Validation determinism: PASSED")

    # -----------------------------------------------------------------
    # Test determinism
    # -----------------------------------------------------------------

    print("\n[4] Test determinism")
    print("-" * 60)

    test_identical = torch.equal(
        test_tensor_1,
        test_tensor_2,
    )

    print(
        f"Test tensors identical: {test_identical}"
    )

    if not test_identical:
        raise AssertionError(
            "Test preprocessing is not deterministic."
        )

    print("Test determinism: PASSED")

    # -----------------------------------------------------------------
    # Shape validation
    # -----------------------------------------------------------------

    print("\n[5] Tensor shape validation")
    print("-" * 60)

    tensors = {
        "train_1": train_tensor_1,
        "train_2": train_tensor_2,
        "validation_1": validation_tensor_1,
        "validation_2": validation_tensor_2,
        "test_1": test_tensor_1,
        "test_2": test_tensor_2,
    }

    for name, tensor in tensors.items():
        print(
            f"{name:<18} {tuple(tensor.shape)}"
        )

        if tuple(tensor.shape) != EXPECTED_SHAPE:
            raise AssertionError(
                f"{name} has shape {tuple(tensor.shape)}; "
                f"expected {EXPECTED_SHAPE}"
            )

    print("\nTensor shape validation: PASSED")

    # -----------------------------------------------------------------
    # Cross-pipeline sanity check
    # -----------------------------------------------------------------

    print("\n[6] Pipeline sanity check")
    print("-" * 60)

    assert train_tensor_1.shape == validation_tensor_1.shape
    assert validation_tensor_1.shape == test_tensor_1.shape

    print(
        "Train / validation / test input contract: PASSED"
    )

    # -----------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING DETERMINISM TEST COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()