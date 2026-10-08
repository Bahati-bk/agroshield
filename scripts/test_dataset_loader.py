"""
AgroShield Dataset/DataLoader integration test.

Verifies that:
    1. Train/validation/test datasets load correctly.
    2. Dataset sizes match the split CSVs.
    3. Individual samples have the correct shape and dtype.
    4. Labels are valid.
    5. DataLoaders produce correctly shaped batches.
"""

from pathlib import Path

import torch

from src.agroshield.training.dataset import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    create_cassava_datasets,
    create_cassava_dataloaders,
)


def main() -> None:

    print("=" * 70)
    print("AGROSHIELD — DATASET LOADER TEST")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Class vocabulary
    # -----------------------------------------------------------------

    print("\n[1] Class vocabulary")
    print("-" * 60)

    for index, class_name in enumerate(CLASS_NAMES):
        print(
            f"{index}: {class_name}"
        )

    assert len(CLASS_NAMES) == 5
    assert len(CLASS_TO_INDEX) == 5
    assert len(INDEX_TO_CLASS) == 5

    print("\nClass vocabulary: PASSED")

    # -----------------------------------------------------------------
    # Create datasets
    # -----------------------------------------------------------------

    print("\n[2] Creating datasets")
    print("-" * 60)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_cassava_datasets()

    print(
        f"Training samples:   {len(train_dataset):,}"
    )

    print(
        f"Validation samples: {len(validation_dataset):,}"
    )

    print(
        f"Test samples:       {len(test_dataset):,}"
    )

    assert len(train_dataset) == 14977
    assert len(validation_dataset) == 3210
    assert len(test_dataset) == 3210

    print("\nDataset sizes: PASSED")

    # -----------------------------------------------------------------
    # Individual samples
    # -----------------------------------------------------------------

    print("\n[3] Inspecting individual samples")
    print("-" * 60)

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    for name, dataset in datasets.items():

        image, label = dataset[0]

        print(f"\n{name.upper()}")

        print(
            f"Image shape: {tuple(image.shape)}"
        )

        print(
            f"Image dtype: {image.dtype}"
        )

        print(
            f"Label:       {label.item()}"
        )

        print(
            f"Label name:  "
            f"{INDEX_TO_CLASS[label.item()]}"
        )

        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)

        assert tuple(image.shape) == (
            3,
            224,
            224,
        )

        assert image.dtype == torch.float32
        assert label.dtype == torch.long

        assert 0 <= label.item() < len(CLASS_NAMES)

    print("\nIndividual sample validation: PASSED")

    # -----------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------

    print("\n[4] Creating DataLoaders")
    print("-" * 60)

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_cassava_dataloaders(
        batch_size=8,
        num_workers=0,
    )

    print(
        f"Training batches:   {len(train_loader):,}"
    )

    print(
        f"Validation batches: {len(validation_loader):,}"
    )

    print(
        f"Test batches:       {len(test_loader):,}"
    )

    # -----------------------------------------------------------------
    # Inspect batches
    # -----------------------------------------------------------------

    print("\n[5] Inspecting batches")
    print("-" * 60)

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }

    for name, loader in loaders.items():

        images, labels = next(iter(loader))

        print(f"\n{name.upper()}")

        print(
            f"Batch image shape: "
            f"{tuple(images.shape)}"
        )

        print(
            f"Batch label shape: "
            f"{tuple(labels.shape)}"
        )

        print(
            f"Image dtype: {images.dtype}"
        )

        print(
            f"Label dtype: {labels.dtype}"
        )

        assert images.shape[0] <= 8
        assert images.shape[1:] == (
            3,
            224,
            224,
        )

        assert labels.shape[0] == images.shape[0]

        assert images.dtype == torch.float32
        assert labels.dtype == torch.long

        assert torch.all(
            (labels >= 0)
            & (labels < len(CLASS_NAMES))
        )

    print("\nDataLoader batch validation: PASSED")

    # -----------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET LOADER TEST COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()