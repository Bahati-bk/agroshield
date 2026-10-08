"""
AgroShield dataset batch-level integrity validation.

This script validates the actual tensors produced by the
Dataset/DataLoader pipeline.

It checks:
    - expected dataset sizes
    - valid class indices
    - class coverage
    - tensor shapes
    - tensor dtypes
    - finite tensor values
    - label/image alignment
"""

from collections import Counter

import torch

from src.agroshield.training.dataset import (
    CLASS_NAMES,
    create_cassava_dataloaders,
)


EXPECTED_SIZES = {
    "train": 14977,
    "validation": 3210,
    "test": 3210,
}

BATCH_SIZE = 32


def validate_loader(
    name: str,
    loader,
    expected_size: int,
) -> None:

    print(f"\n{name.upper()}")
    print("-" * 60)

    # -------------------------------------------------------------
    # Dataset size
    # -------------------------------------------------------------

    actual_size = len(loader.dataset)

    print(f"Dataset size: {actual_size:,}")
    print(f"Expected size: {expected_size:,}")

    assert actual_size == expected_size

    # -------------------------------------------------------------
    # Batch iteration
    # -------------------------------------------------------------

    total_samples = 0
    label_counts = Counter()

    for batch_index, (images, labels) in enumerate(loader):

        # ---------------------------------------------------------
        # Shape validation
        # ---------------------------------------------------------

        assert images.ndim == 4

        assert images.shape[1:] == (
            3,
            224,
            224,
        )

        assert labels.ndim == 1

        assert labels.shape[0] == images.shape[0]

        # ---------------------------------------------------------
        # Dtype validation
        # ---------------------------------------------------------

        assert images.dtype == torch.float32
        assert labels.dtype == torch.long

        # ---------------------------------------------------------
        # Finite-value validation
        # ---------------------------------------------------------

        assert torch.isfinite(images).all()

        assert torch.isfinite(
            labels.float()
        ).all()

        # ---------------------------------------------------------
        # Label-range validation
        # ---------------------------------------------------------

        assert torch.all(
            (labels >= 0)
            & (labels < len(CLASS_NAMES))
        )

        # ---------------------------------------------------------
        # Count samples
        # ---------------------------------------------------------

        total_samples += images.shape[0]

        for label in labels.tolist():
            label_counts[label] += 1

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        if batch_index == 0:
            print(
                f"First batch shape: "
                f"{tuple(images.shape)}"
            )

    # -------------------------------------------------------------
    # Final sample count
    # -------------------------------------------------------------

    print(f"Samples iterated: {total_samples:,}")

    assert total_samples == expected_size

    # -------------------------------------------------------------
    # Class coverage
    # -------------------------------------------------------------

    print("\nClass counts:")

    for class_index, class_name in enumerate(CLASS_NAMES):

        count = label_counts[class_index]

        print(
            f"  {class_index}: "
            f"{class_name:<35} "
            f"{count:,}"
        )

        assert count > 0, (
            f"Class {class_index} "
            f"({class_name}) has zero samples."
        )

    # -------------------------------------------------------------
    # Final class coverage
    # -------------------------------------------------------------

    assert sum(label_counts.values()) == expected_size

    print("\nBatch validation: PASSED")


def main() -> None:

    print("=" * 70)
    print("AGROSHIELD — DATASET BATCH INTEGRITY VALIDATION")
    print("=" * 70)

    print("\n[1] Creating DataLoaders")
    print("-" * 60)

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_cassava_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # -------------------------------------------------------------
    # Validate all splits
    # -------------------------------------------------------------

    print("\n[2] Validating training data")

    validate_loader(
        name="train",
        loader=train_loader,
        expected_size=EXPECTED_SIZES["train"],
    )

    print("\n[3] Validating validation data")

    validate_loader(
        name="validation",
        loader=validation_loader,
        expected_size=EXPECTED_SIZES["validation"],
    )

    print("\n[4] Validating test data")

    validate_loader(
        name="test",
        loader=test_loader,
        expected_size=EXPECTED_SIZES["test"],
    )

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET BATCH INTEGRITY VALIDATION COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()