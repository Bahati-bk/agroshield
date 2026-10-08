from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# AGROSHIELD — CASSAVA DATASET SPLIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cassava_metadata"
    / "cassava_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"


# ------------------------------------------------------------
# Split configuration
# ------------------------------------------------------------

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def validate_configuration():

    total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 1e-9:

        raise ValueError(
            "Split ratios must sum to 1.0."
        )


def load_metadata():

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    return pd.read_csv(
        METADATA_PATH
    )


def create_splits(metadata):

    # --------------------------------------------------------
    # First split:
    #
    # 70% train
    # 30% temporary
    # --------------------------------------------------------

    train, temporary = train_test_split(
        metadata,
        test_size=(
            VALIDATION_RATIO
            + TEST_RATIO
        ),
        stratify=metadata["canonical_label"],
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Second split:
    #
    # Temporary 30%
    #
    # becomes:
    # 15% validation
    # 15% test
    # --------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        / (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    validation, test = train_test_split(
        temporary,
        test_size=relative_test_ratio,
        stratify=temporary["canonical_label"],
        random_state=RANDOM_STATE,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def validate_split_integrity(
    train,
    validation,
    test,
):

    # --------------------------------------------------------
    # Check record counts
    # --------------------------------------------------------

    total_records = (
        len(train)
        + len(validation)
        + len(test)
    )

    print("\n[1] SPLIT COUNTS")
    print("-" * 70)

    print(
        f"Training:   {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    print(
        f"Total:      {total_records:,}"
    )

    # --------------------------------------------------------
    # Check duplicate image IDs across splits
    # --------------------------------------------------------

    print("\n[2] CROSS-SPLIT LEAKAGE CHECK")
    print("-" * 70)

    train_ids = set(train["image_id"])
    validation_ids = set(validation["image_id"])
    test_ids = set(test["image_id"])

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    print(
        "Train ↔ Validation overlap:",
        len(train_validation_overlap),
    )

    print(
        "Train ↔ Test overlap:",
        len(train_test_overlap),
    )

    print(
        "Validation ↔ Test overlap:",
        len(validation_test_overlap),
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):

        raise ValueError(
            "Image ID overlap detected across splits."
        )

    print(
        "Cross-split image leakage: PASSED"
    )

    # --------------------------------------------------------
    # Check class distributions
    # --------------------------------------------------------

    print("\n[3] CLASS DISTRIBUTIONS")
    print("-" * 70)

    overall_distribution = (
        train["canonical_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    train_distribution = (
        train["canonical_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    validation_distribution = (
        validation["canonical_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    test_distribution = (
        test["canonical_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    labels = sorted(
        set(train["canonical_label"])
        | set(validation["canonical_label"])
        | set(test["canonical_label"])
    )

    for label in labels:

        print(
            f"\n{label}"
        )

        print(
            f"  Train:      "
            f"{train_distribution.get(label, 0) * 100:.2f}%"
        )

        print(
            f"  Validation: "
            f"{validation_distribution.get(label, 0) * 100:.2f}%"
        )

        print(
            f"  Test:       "
            f"{test_distribution.get(label, 0) * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Verify every record appears exactly once
    # --------------------------------------------------------

    all_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    if len(all_ids) != total_records:

        raise ValueError(
            "Some records may be duplicated or missing."
        )

    print(
        "\nRecord uniqueness: PASSED"
    )


def save_splits(
    train,
    validation,
    test,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test.to_csv(
        TEST_PATH,
        index=False
    )

    print("\n[4] FILES SAVED")
    print("-" * 70)

    print(
        f"Training:   {TRAIN_PATH}"
    )

    print(
        f"Validation: {VALIDATION_PATH}"
    )

    print(
        f"Test:       {TEST_PATH}"
    )


def main():

    print("=" * 70)
    print("AGROSHIELD — CASSAVA DATASET SPLITTING")
    print("=" * 70)

    validate_configuration()

    print("\n[1] Loading canonical metadata...")

    metadata = load_metadata()

    print(
        f"Records loaded: {len(metadata):,}"
    )

    print("\n[2] Creating stratified splits...")

    train, validation, test = create_splits(
        metadata
    )

    validate_split_integrity(
        train,
        validation,
        test,
    )

    save_splits(
        train,
        validation,
        test,
    )

    print("\n" + "=" * 70)
    print("DATASET SPLITTING COMPLETE")
    print("STATUS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()