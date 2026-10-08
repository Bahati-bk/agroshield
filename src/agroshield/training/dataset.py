"""
AgroShield PyTorch dataset and DataLoader utilities.

This module connects the canonical split CSV files to the
cassava image preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.agroshield.data.preprocessing import (
    get_test_transforms,
    get_train_transforms,
    get_validation_transforms,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DEFAULT_IMAGE_DIR = Path(
    "data/raw/cassava/_download/train_images"
)

DEFAULT_SPLIT_DIR = Path(
    "data/splits"
)


# ---------------------------------------------------------------------
# Stable class vocabulary
# ---------------------------------------------------------------------

CLASS_NAMES = [
    "cassava_bacterial_blight",
    "cassava_brown_streak_disease",
    "cassava_green_mottle",
    "cassava_mosaic_disease",
    "healthy",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class CassavaDataset(Dataset):
    """
    PyTorch Dataset for the AgroShield cassava dataset.

    Parameters
    ----------
    csv_path:
        Path to train.csv, validation.csv, or test.csv.

    image_dir:
        Directory containing the original JPEG images.

    transform:
        Image preprocessing/augmentation pipeline.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path = DEFAULT_IMAGE_DIR,
        transform: Optional[object] = None,
    ) -> None:

        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.csv_path}"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        self.data = pd.read_csv(self.csv_path)

        required_columns = {
            "image_id",
            "canonical_label",
        }

        missing_columns = required_columns - set(
            self.data.columns
        )

        if missing_columns:
            raise ValueError(
                "Split CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # Validate canonical labels before constructing samples.
        unknown_labels = set(
            self.data["canonical_label"]
        ) - set(CLASS_TO_INDEX)

        if unknown_labels:
            raise ValueError(
                "Unknown canonical labels found: "
                f"{sorted(unknown_labels)}"
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess one sample.

        Returns
        -------
        image:
            Float tensor with shape [3, 224, 224].

        label:
            Long tensor containing the class index.
        """

        row = self.data.iloc[index]

        image_id = str(row["image_id"])
        canonical_label = str(
            row["canonical_label"]
        )

        image_path = self.image_dir / image_id

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by split does not exist: "
                f"{image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = CLASS_TO_INDEX[canonical_label]

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label_tensor


# ---------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------

def create_cassava_datasets(
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
) -> tuple[
    CassavaDataset,
    CassavaDataset,
    CassavaDataset,
]:
    """
    Create train, validation, and test datasets with the
    correct preprocessing pipeline for each split.
    """

    split_dir = Path(split_dir)

    train_dataset = CassavaDataset(
        csv_path=split_dir / "train.csv",
        image_dir=image_dir,
        transform=get_train_transforms(),
    )

    validation_dataset = CassavaDataset(
        csv_path=split_dir / "validation.csv",
        image_dir=image_dir,
        transform=get_validation_transforms(),
    )

    test_dataset = CassavaDataset(
        csv_path=split_dir / "test.csv",
        image_dir=image_dir,
        transform=get_test_transforms(),
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# ---------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------

def create_cassava_dataloaders(
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    batch_size:
        Number of images returned in each batch.

    num_workers:
        Number of worker processes used by the DataLoader.

        We default to 0 because this is the safest setting
        for Windows development.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_cassava_datasets(
        split_dir=split_dir,
        image_dir=image_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )