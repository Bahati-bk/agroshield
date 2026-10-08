from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/cassava/_download")


def main():
    csv_path = DATA_DIR / "train.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)

    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== LABEL COUNTS ===")
    print(df["label"].value_counts().sort_index())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()