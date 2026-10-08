from pathlib import Path
from collections import defaultdict

import pandas as pd
from PIL import Image
import imagehash


# ============================================================
# AGROSHIELD — NEAR-DUPLICATE IMAGE AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cassava_metadata"
    / "cassava_metadata.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cassava"
    / "_download"
    / "train_images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cassava_audit"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "near_duplicate_candidates.csv"
)


# Maximum perceptual-hash Hamming distance
# considered a near-duplicate candidate.
HASH_DISTANCE_THRESHOLD = 5


def compute_hash(image_path):
    """Compute a 64-bit perceptual hash."""

    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image)

    except Exception as error:
        print(
            f"WARNING: Could not hash "
            f"{image_path.name}: {error}"
        )

        return None


def hash_to_int(image_hash):
    """
    Convert ImageHash into a 64-bit integer.
    """

    return int(str(image_hash), 16)


def get_bit_chunks(hash_value, number_of_chunks=8):
    """
    Split a 64-bit hash into 8-bit chunks.

    Returns:
        tuple of 8 integer chunks.
    """

    chunks = []

    for index in range(number_of_chunks):

        shift = (
            (number_of_chunks - 1 - index)
            * 8
        )

        chunk = (
            hash_value >> shift
        ) & 0xFF

        chunks.append(chunk)

    return tuple(chunks)


def hamming_distance(hash_a, hash_b):
    """
    Calculate Hamming distance between
    two 64-bit integers.
    """

    return (
        hash_a ^ hash_b
    ).bit_count()


def main():

    print("=" * 70)
    print("AGROSHIELD — NEAR-DUPLICATE IMAGE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\n[1] Loading canonical metadata...")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    print(
        f"Images to inspect: {len(metadata):,}"
    )

    # --------------------------------------------------------
    # Compute hashes
    # --------------------------------------------------------

    print("\n[2] Computing perceptual hashes...")

    hashes = {}

    for index, image_id in enumerate(
        metadata["image_id"],
        start=1
    ):

        image_path = IMAGE_DIR / image_id

        image_hash = compute_hash(image_path)

        if image_hash is not None:

            hashes[image_id] = (
                hash_to_int(image_hash)
            )

        if index % 1000 == 0:
            print(
                f"  Processed "
                f"{index:,}/{len(metadata):,}"
            )

    print(
        f"\nSuccessfully hashed: "
        f"{len(hashes):,}"
    )

    # --------------------------------------------------------
    # Exact hash collisions
    # --------------------------------------------------------

    print(
        "\n[3] Checking exact perceptual "
        "hash collisions..."
    )

    exact_groups = defaultdict(list)

    for image_id, hash_value in hashes.items():

        exact_groups[hash_value].append(
            image_id
        )

    exact_hash_groups = {
        hash_value: image_ids
        for hash_value, image_ids
        in exact_groups.items()
        if len(image_ids) > 1
    }

    print(
        f"Exact perceptual-hash groups: "
        f"{len(exact_hash_groups):,}"
    )

    # --------------------------------------------------------
    # Multi-band candidate generation
    # --------------------------------------------------------

    print(
        "\n[4] Generating near-duplicate candidates..."
    )

    print(
        "Using 8 independent 8-bit hash bands."
    )

    buckets = [
        defaultdict(set)
        for _ in range(8)
    ]

    for image_id, hash_value in hashes.items():

        chunks = get_bit_chunks(
            hash_value
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):

            buckets[chunk_index][chunk].add(
                image_id
            )

    # --------------------------------------------------------
    # Generate candidate pairs
    # --------------------------------------------------------

    candidate_pairs = set()

    for bucket_index, bucket in enumerate(
        buckets
    ):

        print(
            f"  Processing hash band "
            f"{bucket_index + 1}/8..."
        )

        for image_ids in bucket.values():

            if len(image_ids) < 2:
                continue

            image_ids = sorted(image_ids)

            for i in range(len(image_ids)):

                for j in range(
                    i + 1,
                    len(image_ids)
                ):

                    pair = (
                        image_ids[i],
                        image_ids[j],
                    )

                    candidate_pairs.add(pair)

    print(
        f"\nUnique candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    # --------------------------------------------------------
    # Calculate actual distances
    # --------------------------------------------------------

    print(
        "\n[5] Calculating exact Hamming distances..."
    )

    candidates = []

    for image_a, image_b in candidate_pairs:

        distance = hamming_distance(
            hashes[image_a],
            hashes[image_b],
        )

        if distance <= HASH_DISTANCE_THRESHOLD:

            candidates.append(
                {
                    "image_id_a": image_a,
                    "image_id_b": image_b,
                    "hash_distance": distance,
                }
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame(
        candidates,
        columns=[
            "image_id_a",
            "image_id_b",
            "hash_distance",
        ],
    )

    if not results.empty:

        results = results.sort_values(
            "hash_distance"
        )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NEAR-DUPLICATE AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"Images hashed:              "
        f"{len(hashes):,}"
    )

    print(
        f"Exact hash groups:          "
        f"{len(exact_hash_groups):,}"
    )

    print(
        f"Candidate pairs examined:   "
        f"{len(candidate_pairs):,}"
    )

    print(
        f"Near-duplicate candidates:  "
        f"{len(candidates):,}"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Near-duplicate candidates are NOT automatically duplicates."
    )

    print(
        "They require visual inspection before removal or grouping."
    )


if __name__ == "__main__":
    main()