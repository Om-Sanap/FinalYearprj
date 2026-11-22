
"""
WAYANAD NISAR - REFERENCE SAMPLE GENERATION

Generates reproducible, stratified random sample locations from the
current K-Means classification for independent reference labeling.

IMPORTANT:
- Predicted cluster IDs are not validated land-cover labels.
- Reference labels remain blank until independently assigned.
- Sampling locations alone do not constitute an accuracy assessment.
"""

from pathlib import Path

import csv
import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "NISAR_results"

CLASSIFICATION_FILE = (
    RESULTS_DIR / "wayanad_kmeans_classification.tif"
)

OUTPUT_CSV = RESULTS_DIR / "wayanad_reference_samples.csv"


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42
WINDOW_SIZE = 512

# Class IDs are detected from the raster.
# Class 0 and negative values are treated as background/invalid.


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WAYANAD NISAR REFERENCE SAMPLE GENERATION")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

if not CLASSIFICATION_FILE.is_file():
    raise FileNotFoundError(
        f"\nClassification raster not found:\n"
        f"{CLASSIFICATION_FILE}\n\n"
        "Expected the current classification in NISAR_results."
    )

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# OPEN CLASSIFICATION AND DETECT CLASSES
# ============================================================

with rasterio.open(CLASSIFICATION_FILE) as src:

    if src.crs is None:
        raise ValueError("Classification raster has no CRS.")

    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height

    print("\nClassification:", CLASSIFICATION_FILE)
    print("Image dimensions:", width, "columns x", height, "rows")
    print("CRS:", crs)

    # --------------------------------------------------------
    # PASS 1: detect class IDs and count available pixels
    # --------------------------------------------------------

    print("\nScanning for actual class IDs...")

    class_counts = {}

    for row_off in range(0, height, WINDOW_SIZE):
        for col_off in range(0, width, WINDOW_SIZE):

            window = Window(
                col_off,
                row_off,
                min(WINDOW_SIZE, width - col_off),
                min(WINDOW_SIZE, height - row_off),
            )

            band = src.read(1, window=window, masked=True)
            values = np.asarray(band.data)

            valid = ~np.ma.getmaskarray(band)
            valid &= np.isfinite(values)
            valid &= values > 0

            if not np.any(valid):
                continue

            class_ids, counts = np.unique(
                values[valid].astype(np.int64),
                return_counts=True,
            )

            for class_id, count in zip(class_ids, counts):
                class_id = int(class_id)
                class_counts[class_id] = (
                    class_counts.get(class_id, 0) + int(count)
                )

    class_ids = sorted(class_counts)

    if not class_ids:
        raise ValueError(
            "No valid class IDs greater than zero were found."
        )

    print("\nDetected classes:", class_ids)

    for class_id in class_ids:
        print(
            f"Class {class_id}: "
            f"{class_counts[class_id]:,} available pixels"
        )

    # --------------------------------------------------------
    # Determine sample counts
    # --------------------------------------------------------

    sample_targets = {}

    for class_id in class_ids:
        available = class_counts[class_id]
        sample_targets[class_id] = min(
            SAMPLES_PER_CLASS,
            available,
        )

        if available < SAMPLES_PER_CLASS:
            print(
                f"Warning: class {class_id} has only {available} "
                f"pixels; sampling all available pixels."
            )

    # --------------------------------------------------------
    # PASS 2: choose reproducible random pixels.
    #
    # Random priorities are assigned to valid pixels. Retaining
    # the lowest priorities produces a uniform random sample
    # without replacement, while keeping memory use bounded.
    # --------------------------------------------------------

    print("\nSelecting random sample locations...")

    # For each class, keep (priority, row, column) arrays.
    reservoirs = {
        class_id: (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
        for class_id in class_ids
    }

    for row_off in range(0, height, WINDOW_SIZE):
        for col_off in range(0, width, WINDOW_SIZE):

            window = Window(
                col_off,
                row_off,
                min(WINDOW_SIZE, width - col_off),
                min(WINDOW_SIZE, height - row_off),
            )

            band = src.read(1, window=window, masked=True)
            values = np.asarray(band.data)

            valid = ~np.ma.getmaskarray(band)
            valid &= np.isfinite(values)
            valid &= values > 0

            if not np.any(valid):
                continue

            window_class_ids = np.unique(
                values[valid].astype(np.int64)
            )

            for class_value in window_class_ids:
                class_id = int(class_value)
                target = sample_targets[class_id]

                if target == 0:
                    continue

                local_rows, local_cols = np.where(
                    valid & (values == class_value)
                )

                if local_rows.size == 0:
                    continue

                candidate_priorities = rng.random(local_rows.size)

                candidate_rows = local_rows.astype(np.int64) + row_off
                candidate_cols = local_cols.astype(np.int64) + col_off

                old_priorities, old_rows, old_cols = reservoirs[class_id]

                combined_priorities = np.concatenate(
                    [old_priorities, candidate_priorities]
                )
                combined_rows = np.concatenate(
                    [old_rows, candidate_rows]
                )
                combined_cols = np.concatenate(
                    [old_cols, candidate_cols]
                )

                keep_count = min(target, combined_priorities.size)

                if combined_priorities.size > keep_count:
                    keep = np.argpartition(
                        combined_priorities,
                        keep_count - 1,
                    )[:keep_count]
                else:
                    keep = np.arange(combined_priorities.size)

                reservoirs[class_id] = (
                    combined_priorities[keep],
                    combined_rows[keep],
                    combined_cols[keep],
                )

    # --------------------------------------------------------
    # BUILD SAMPLE RECORDS
    # --------------------------------------------------------

    all_samples = []

    for class_id in class_ids:
        _, rows, cols = reservoirs[class_id]

        if len(rows) != sample_targets[class_id]:
            raise RuntimeError(
                f"Expected {sample_targets[class_id]} samples for "
                f"class {class_id}, but selected {len(rows)}."
            )

        # Shuffle the class-specific locations before creating records.
        order = rng.permutation(len(rows))

        rows = rows[order]
        cols = cols[order]

        for row, col in zip(rows, cols):
            x, y = rasterio.transform.xy(
                transform,
                int(row),
                int(col),
                offset="center",
            )

            all_samples.append({
                "sample_id": None,
                "row": int(row),
                "column": int(col),
                "x": float(x),
                "y": float(y),
                "crs": str(crs),
                "predicted_class": int(class_id),
                "predicted_class_name": f"K-Means cluster {class_id}",
                "reference_class": "",
                "reference_class_name": "",
                "confidence": "",
                "notes": "",
            })

    # Shuffle the complete sample list and assign unique IDs.
    rng.shuffle(all_samples)

    for sample_id, sample in enumerate(all_samples, start=1):
        sample["sample_id"] = sample_id


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "sample_id",
    "row",
    "column",
    "x",
    "y",
    "crs",
    "predicted_class",
    "predicted_class_name",
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes",
]

print("\nSaving reference samples...")

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_samples)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE SAMPLE GENERATION COMPLETE")
print("=" * 70)

print("Total samples:", len(all_samples))

for class_id in class_ids:
    actual_count = sum(
        sample["predicted_class"] == class_id
        for sample in all_samples
    )

    print(
        f"Class {class_id}: {actual_count} samples "
        f"from {class_counts[class_id]:,} available pixels"
    )

print("\nCreated:")
print(OUTPUT_CSV)

print("\nIMPORTANT NEXT STEP")
print("-" * 70)
print(
    "Independently label each sample using suitable reference imagery, "
    "field observations, or other reliable geographic data."
)
print(
    "Do not copy predicted_class into reference_class. Leave the "
    "reference fields blank until the actual/reference label is assigned."
)
print(
    "Record the source and date of the reference evidence in the notes "
    "field where possible."
)
print("=" * 70)
