from pathlib import Path
import json

import pandas as pd
from PIL import Image


# ============================================================
# AGROSHIELD — CASSAVA CANONICAL METADATA BUILDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cassava"
    / "_download"
    / "train.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cassava"
    / "_download"
    / "train_images"
)

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cassava"
    "_download"
    / "label_num_to_disease_map.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cassava_metadata"
)

OUTPUT_PATH = OUTPUT_DIR / "cassava_metadata.csv"


# ------------------------------------------------------------
# Canonical label mapping
# ------------------------------------------------------------

CANONICAL_LABELS = {
    0: "cassava_bacterial_blight",
    1: "cassava_brown_streak_disease",
    2: "cassava_green_mottle",
    3: "cassava_mosaic_disease",
    4: "healthy",
}


def load_source_data():
    """Load the original dataset metadata."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {CSV_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    return dataframe


def load_original_label_mapping():
    """Load the original dataset label mapping if available."""

    if not LABEL_MAP_PATH.exists():
        print(
            "WARNING: Original label mapping file was not found."
        )
        return {}

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_image_metadata(image_id):
    """Extract basic metadata from an image."""

    image_path = IMAGE_DIR / image_id

    if not image_path.exists():
        return {
            "width": None,
            "height": None,
            "format": None,
            "verified": False,
        }

    try:
        with Image.open(image_path) as image:
            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "verified": True,
            }

    except Exception:
        return {
            "width": None,
            "height": None,
            "format": None,
            "verified": False,
        }


def build_metadata(dataframe):
    """Convert source metadata into AgroShield's canonical schema."""

    records = []

    for _, row in dataframe.iterrows():

        image_id = row["image_id"]
        source_label = int(row["label"])

        if source_label not in CANONICAL_LABELS:
            raise ValueError(
                f"Unknown source label encountered: {source_label}"
            )

        image_metadata = get_image_metadata(image_id)

        record = {
            "image_id": image_id,

            # Provenance
            "source_dataset": (
                "kaggle_cassava_leaf_disease_classification"
            ),

            # Preserve original label
            "source_label": source_label,

            # Standardized AgroShield label
            "canonical_label": CANONICAL_LABELS[source_label],

            # Crop
            "crop": "cassava",

            # Geographic metadata
            "country": "Uganda",
            "region": None,

            # Collection environment
            "collection_type": "field",

            # Image metadata
            "width": image_metadata["width"],
            "height": image_metadata["height"],
            "format": image_metadata["format"],

            # Integrity verification
            "verified": image_metadata["verified"],
        }

        records.append(record)

    return pd.DataFrame(records)


def validate_schema(dataframe):
    """Validate the canonical metadata schema."""

    required_columns = [
        "image_id",
        "source_dataset",
        "source_label",
        "canonical_label",
        "crop",
        "country",
        "region",
        "collection_type",
        "width",
        "height",
        "format",
        "verified",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if dataframe["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs detected."
        )

    if dataframe["canonical_label"].isna().any():
        raise ValueError(
            "Missing canonical labels detected."
        )

    if dataframe["crop"].isna().any():
        raise ValueError(
            "Missing crop values detected."
        )

    if not dataframe["verified"].all():
        raise ValueError(
            "One or more images failed verification."
        )


def main():

    print("=" * 70)
    print("AGROSHIELD — CANONICAL CASSAVA METADATA BUILDER")
    print("=" * 70)

    print("\n[1] Loading source dataset...")

    dataframe = load_source_data()

    print(f"Source records: {len(dataframe):,}")

    print("\n[2] Loading original label mapping...")

    original_mapping = load_original_label_mapping()

    if original_mapping:
        print("Original label mapping found:")
        for label, disease in original_mapping.items():
            print(f"  {label} → {disease}")

    print("\n[3] Building canonical metadata...")

    metadata = build_metadata(dataframe)

    print(f"Canonical records created: {len(metadata):,}")

    print("\n[4] Validating schema...")

    validate_schema(metadata)

    print("Schema validation: PASSED")

    print("\n[5] Creating output directory...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[6] Saving canonical metadata...")

    metadata.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to: {OUTPUT_PATH}")

    print("\n[7] Canonical class distribution:")

    distribution = (
        metadata["canonical_label"]
        .value_counts()
        .sort_index()
    )

    for label, count in distribution.items():
        percentage = (
            count / len(metadata)
        ) * 100

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print("\n[8] Schema preview:")
    print(metadata.head().to_string(index=False))

    print("\n" + "=" * 70)
    print("METADATA BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()