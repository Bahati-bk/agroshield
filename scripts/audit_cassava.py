from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/raw/cassava/_download")

CSV_PATH = DATA_DIR / "train.csv"

IMAGE_DIR_CANDIDATES = [
    DATA_DIR / "train_images",
    DATA_DIR / "images",
    DATA_DIR,
]


# ============================================================
# Helpers
# ============================================================

def find_image_directory() -> Path:
    """
    Find the directory containing the training images.
    """

    for directory in IMAGE_DIR_CANDIDATES:
        if not directory.exists():
            continue

        jpg_files = list(directory.glob("*.jpg"))

        if jpg_files:
            return directory

    raise FileNotFoundError(
        "Could not locate the directory containing .jpg images."
    )


def calculate_md5(file_path: Path) -> str:
    """
    Calculate an MD5 hash for duplicate detection.
    """

    md5 = hashlib.md5()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def load_label_mapping() -> dict:
    """
    Load Kaggle's disease label mapping if available.
    """

    mapping_path = DATA_DIR / "label_num_to_disease_map.json"

    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Main audit
# ============================================================

def main():

    print("=" * 70)
    print("AGROSHIELD — CASSAVA DATASET QUALITY AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------------

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find metadata file: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    print("\n[1] METADATA")
    print("-" * 70)

    print(f"Records in train.csv: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # --------------------------------------------------------
    # 2. Locate images
    # --------------------------------------------------------

    image_dir = find_image_directory()

    print("\n[2] IMAGE DIRECTORY")
    print("-" * 70)

    print(f"Image directory: {image_dir}")

    image_files = list(image_dir.glob("*.jpg"))

    print(f"JPEG files found: {len(image_files):,}")

    # --------------------------------------------------------
    # 3. Missing images
    # --------------------------------------------------------

    print("\n[3] MISSING IMAGE CHECK")
    print("-" * 70)

    csv_image_ids = set(df["image_id"].astype(str))
    disk_image_ids = {file.name for file in image_files}

    missing_images = csv_image_ids - disk_image_ids
    extra_images = disk_image_ids - csv_image_ids

    print(f"Images referenced by CSV: {len(csv_image_ids):,}")
    print(f"Images found on disk:     {len(disk_image_ids):,}")
    print(f"Missing images:            {len(missing_images):,}")
    print(f"Extra images:              {len(extra_images):,}")

    if missing_images:
        print("\nFirst missing images:")
        for image in list(sorted(missing_images))[:10]:
            print(f"  {image}")

    if extra_images:
        print("\nFirst extra images:")
        for image in list(sorted(extra_images))[:10]:
            print(f"  {image}")

    # --------------------------------------------------------
    # 4. Class distribution
    # --------------------------------------------------------

    print("\n[4] CLASS DISTRIBUTION")
    print("-" * 70)

    label_counts = df["label"].value_counts().sort_index()

    print(label_counts)

    print("\nPercentages:")

    label_percentages = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    for label, percentage in label_percentages.items():
        print(f"  Label {label}: {percentage:.2f}%")

    # --------------------------------------------------------
    # 5. Image integrity
    # --------------------------------------------------------

    print("\n[5] IMAGE INTEGRITY")
    print("-" * 70)

    corrupt_images = []
    dimensions = []
    formats = Counter()

    total_images = len(image_files)

    for index, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as image:

                # verify() checks whether the image can be decoded
                image.verify()

            # Reopen after verify because verify() invalidates the object
            with Image.open(image_path) as image:

                dimensions.append(image.size)
                formats[image.format] += 1

        except Exception as error:

            corrupt_images.append(
                {
                    "file": image_path.name,
                    "error": str(error),
                }
            )

        if index % 1000 == 0:
            print(
                f"Checked {index:,}/{total_images:,} images..."
            )

    print(f"\nCorrupt/unreadable images: {len(corrupt_images):,}")

    print("\nImage formats:")

    for image_format, count in formats.items():
        print(f"  {image_format}: {count:,}")

    # --------------------------------------------------------
    # 6. Image dimensions
    # --------------------------------------------------------

    print("\n[6] IMAGE DIMENSIONS")
    print("-" * 70)

    if dimensions:

        widths = np.array([width for width, _ in dimensions])
        heights = np.array([height for _, height in dimensions])

        print(
            f"Width  — min: {widths.min()}, "
            f"max: {widths.max()}, "
            f"mean: {widths.mean():.1f}"
        )

        print(
            f"Height — min: {heights.min()}, "
            f"max: {heights.max()}, "
            f"mean: {heights.mean():.1f}"
        )

        print(
            f"Unique dimension combinations: "
            f"{len(Counter(dimensions)):,}"
        )

        print("\nMost common dimensions:")

        for dimension, count in Counter(dimensions).most_common(10):
            print(f"  {dimension}: {count:,}")

    # --------------------------------------------------------
    # 7. Duplicate detection
    # --------------------------------------------------------

    print("\n[7] DUPLICATE IMAGE CHECK")
    print("-" * 70)

    hashes = {}
    duplicate_groups = []

    valid_image_paths = [
        image_dir / image_id
        for image_id in df["image_id"].astype(str)
        if (image_dir / image_id).exists()
    ]

    for index, image_path in enumerate(valid_image_paths, start=1):

        try:
            image_hash = calculate_md5(image_path)

            if image_hash in hashes:

                duplicate_groups.append(
                    (
                        image_hash,
                        hashes[image_hash],
                        image_path.name,
                    )
                )

            else:
                hashes[image_hash] = image_path.name

        except Exception:
            pass

        if index % 1000 == 0:
            print(
                f"Hashed {index:,}/{len(valid_image_paths):,} images..."
            )

    print(
        f"\nDuplicate image relationships found: "
        f"{len(duplicate_groups):,}"
    )

    if duplicate_groups:

        print("\nFirst duplicate pairs:")

        for image_hash, first, duplicate in duplicate_groups[:10]:
            print(f"  {first} <-> {duplicate}")

    # --------------------------------------------------------
    # 8. Label mapping
    # --------------------------------------------------------

    print("\n[8] LABEL MAPPING")
    print("-" * 70)

    mapping = load_label_mapping()

    if mapping:

        for label, disease in sorted(
            mapping.items(),
            key=lambda item: int(item[0])
        ):
            print(f"  {label} → {disease}")

    else:

        print(
            "label_num_to_disease_map.json was not found."
        )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"CSV records:       {len(df):,}")
    print(f"Images found:      {len(image_files):,}")
    print(f"Missing images:    {len(missing_images):,}")
    print(f"Extra images:      {len(extra_images):,}")
    print(f"Corrupt images:    {len(corrupt_images):,}")
    print(f"Duplicate pairs:   {len(duplicate_groups):,}")
    print(f"Unique dimensions:  {len(Counter(dimensions)):,}")

    print("\nAudit complete.")


if __name__ == "__main__":
    main()