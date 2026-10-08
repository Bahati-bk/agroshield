from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# AGROSHIELD — CASSAVA PROVENANCE & LABEL VALIDATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cassava_metadata"
    / "cassava_metadata.csv"
)

DATASET_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dataset_registry.yaml"
)

LABEL_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "label_registry.yaml"
)


EXPECTED_CANONICAL_LABELS = {
    "cassava_bacterial_blight",
    "cassava_brown_streak_disease",
    "cassava_green_mottle",
    "cassava_mosaic_disease",
    "healthy",
}


def load_yaml(path):
    """Load a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_dataset_registry(registry):
    """Validate dataset provenance configuration."""

    datasets = registry.get("datasets", {})

    dataset_key = "cassava_makerere_nacrri_2020"

    if dataset_key not in datasets:
        raise ValueError(
            f"Dataset registry missing: {dataset_key}"
        )

    dataset = datasets[dataset_key]

    required_fields = [
        "source_dataset",
        "source_host",
        "source_url",
        "original_owner",
        "country",
        "annotation_source",
        "source_license",
        "license_status",
        "provenance_status",
    ]

    missing = [
        field
        for field in required_fields
        if field not in dataset
    ]

    if missing:
        raise ValueError(
            f"Missing provenance fields: {missing}"
        )

    return dataset


def validate_label_registry(registry):
    """Validate the canonical cassava label vocabulary."""

    cassava = registry.get("crops", {}).get("cassava")

    if cassava is None:
        raise ValueError(
            "Cassava label registry entry not found."
        )

    diseases = cassava.get("diseases", {})

    registered_labels = set(diseases.keys())

    if registered_labels != EXPECTED_CANONICAL_LABELS:
        raise ValueError(
            "Canonical label registry does not match "
            "expected cassava labels.\n"
            f"Expected: {EXPECTED_CANONICAL_LABELS}\n"
            f"Found: {registered_labels}"
        )

    return registered_labels


def validate_metadata(
    metadata,
    dataset_registry,
    registered_labels,
):
    """Validate metadata against provenance and label registries."""

    print("\n[1] RECORD COUNT")
    print("-" * 70)

    print(f"Metadata records: {len(metadata):,}")

    if len(metadata) != 21397:
        raise ValueError(
            f"Unexpected record count: {len(metadata):,}"
        )

    print("Record count: PASSED")

    print("\n[2] DATASET PROVENANCE")
    print("-" * 70)

    expected_source = (
        dataset_registry["source_dataset"]
    )

    actual_sources = (
        set(metadata["source_dataset"].unique())
    )

    print(f"Expected source: {expected_source}")
    print(f"Detected sources: {actual_sources}")

    if actual_sources != {expected_source}:
        raise ValueError(
            "Unexpected dataset source detected."
        )

    print("Dataset provenance: PASSED")

    print("\n[3] CROP VALIDATION")
    print("-" * 70)

    crops = set(metadata["crop"].unique())

    print(f"Detected crops: {crops}")

    if crops != {"cassava"}:
        raise ValueError(
            "Unexpected crop values detected."
        )

    print("Crop validation: PASSED")

    print("\n[4] COUNTRY VALIDATION")
    print("-" * 70)

    countries = set(
        metadata["country"].dropna().unique()
    )

    print(f"Detected countries: {countries}")

    if countries != {"Uganda"}:
        raise ValueError(
            "Unexpected country values detected."
        )

    print("Country validation: PASSED")

    print("\n[5] LABEL VALIDATION")
    print("-" * 70)

    labels = set(
        metadata["canonical_label"].unique()
    )

    print(f"Registered labels: {registered_labels}")
    print(f"Detected labels:   {labels}")

    if labels != registered_labels:
        raise ValueError(
            "Metadata contains labels that are not "
            "registered in the canonical label vocabulary."
        )

    print("Canonical labels: PASSED")

    print("\n[6] SOURCE LABEL VALIDATION")
    print("-" * 70)

    source_labels = set(
        metadata["source_label"].unique()
    )

    expected_source_labels = {0, 1, 2, 3, 4}

    print(f"Expected source labels: {expected_source_labels}")
    print(f"Detected source labels: {source_labels}")

    if source_labels != expected_source_labels:
        raise ValueError(
            "Unexpected source labels detected."
        )

    print("Source labels: PASSED")

    print("\n[7] IMAGE VERIFICATION")
    print("-" * 70)

    verified_count = metadata["verified"].sum()

    print(
        f"Verified images: "
        f"{verified_count:,}/{len(metadata):,}"
    )

    if verified_count != len(metadata):
        raise ValueError(
            "Not all images are marked as verified."
        )

    print("Image verification: PASSED")


def main():

    print("=" * 70)
    print("AGROSHIELD — PROVENANCE & LABEL VALIDATION")
    print("=" * 70)

    print("\nLoading canonical metadata...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    print("Loading dataset registry...")

    dataset_registry = load_yaml(
        DATASET_REGISTRY_PATH
    )

    print("Loading label registry...")

    label_registry = load_yaml(
        LABEL_REGISTRY_PATH
    )

    print("\nValidating dataset registry...")

    dataset_info = validate_dataset_registry(
        dataset_registry
    )

    print("Dataset registry: PASSED")

    print("\nValidating label registry...")

    registered_labels = validate_label_registry(
        label_registry
    )

    print("Label registry: PASSED")

    validate_metadata(
        metadata,
        dataset_info,
        registered_labels,
    )

    print("\n" + "=" * 70)
    print("PROVENANCE & LABEL VALIDATION COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()