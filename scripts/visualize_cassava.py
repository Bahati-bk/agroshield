from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/raw/cassava/_download")
IMAGE_DIR = DATA_DIR / "train_images"
CSV_PATH = DATA_DIR / "train.csv"

OUTPUT_DIR = Path("data/interim/cassava_audit")

SAMPLES_PER_CLASS = 8

LABEL_NAMES = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    print("=" * 70)
    print("AGROSHIELD — VISUAL CASSAVA DATASET INSPECTION")
    print("=" * 70)

    print(f"\nSamples per class: {SAMPLES_PER_CLASS}")

    # --------------------------------------------------------
    # Create one balanced contact sheet
    # --------------------------------------------------------

    rows = SAMPLES_PER_CLASS
    cols = len(LABEL_NAMES)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 20),
    )

    for column, label in enumerate(sorted(LABEL_NAMES)):

        class_df = df[df["label"] == label]

        # Fixed random state makes the inspection reproducible.
        samples = class_df.sample(
            n=min(SAMPLES_PER_CLASS, len(class_df)),
            random_state=42,
        ).reset_index(drop=True)

        for row in range(rows):

            ax = axes[row, column]

            ax.axis("off")

            if row >= len(samples):
                continue

            image_id = samples.loc[row, "image_id"]

            image_path = IMAGE_DIR / image_id

            image = Image.open(image_path)

            ax.imshow(image)

            if row == 0:
                ax.set_title(
                    f"{LABEL_NAMES[label]}\n"
                    f"Label {label}",
                    fontsize=10,
                )

            ax.set_xlabel(
                image_id,
                fontsize=6,
            )

    fig.suptitle(
        "AgroShield — Cassava Dataset Visual Audit",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "cassava_class_samples.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved contact sheet:")
    print(output_path)

    # --------------------------------------------------------
    # Create individual class sheets
    # --------------------------------------------------------

    for label, label_name in LABEL_NAMES.items():

        class_df = df[df["label"] == label]

        samples = class_df.sample(
            n=min(20, len(class_df)),
            random_state=42,
        ).reset_index(drop=True)

        columns = 5
        rows = math.ceil(len(samples) / columns)

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(15, 3 * rows),
        )

        axes = axes.flatten()

        for index, ax in enumerate(axes):

            ax.axis("off")

            if index >= len(samples):
                continue

            image_id = samples.loc[index, "image_id"]

            image_path = IMAGE_DIR / image_id

            image = Image.open(image_path)

            ax.imshow(image)

            ax.set_title(
                image_id,
                fontsize=7,
            )

        fig.suptitle(
            f"{label_name} — Label {label}",
            fontsize=14,
        )

        plt.tight_layout()

        filename = f"class_{label}_{label_name.split()[0].lower()}.png"

        output_path = OUTPUT_DIR / filename

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    print("\nIndividual class sheets created.")

    print("\nVisual inspection complete.")


if __name__ == "__main__":
    main()