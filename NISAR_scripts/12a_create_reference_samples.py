import os
import csv
import numpy as np
import rasterio


# ============================================================
# WAYANAD NISAR REFERENCE SAMPLE GENERATION
# ============================================================

print("=" * 70)
print("WAYANAD NISAR REFERENCE SAMPLE GENERATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\FinalYearPrj\prj"

CLASSIFICATION_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_kmeans_classes.tif"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "NISAR_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "wayanad_reference_samples.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of validation samples per K-Means class
SAMPLES_PER_CLASS = 200

# Random seed for reproducibility
RANDOM_SEED = 42

CLASSES = [1, 2, 3, 4, 5]

CLASS_NAMES = {
    1: "Mixed vegetation / agricultural",
    2: "Built-up / exposed structured",
    3: "Strong structural / double-bounce-like",
    4: "Low-backscatter smooth surface",
    5: "Forest / dense vegetation"
}


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(CLASSIFICATION_FILE):

    raise FileNotFoundError(
        f"\nClassification raster not found:\n"
        f"{CLASSIFICATION_FILE}"
    )


# ============================================================
# LOAD CLASSIFICATION
# ============================================================

print("\nLoading K-Means classification...")

with rasterio.open(CLASSIFICATION_FILE) as src:

    classification = src.read(1)

    transform = src.transform
    crs = src.crs

    height = src.height
    width = src.width

print("Image shape :", classification.shape)
print("CRS         :", crs)
print("Width       :", width)
print("Height      :", height)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# CREATE STRATIFIED SAMPLES
# ============================================================

print("\nCreating stratified validation samples...")

all_samples = []


for class_id in CLASSES:

    print(f"\nClass {class_id}")
    print("-" * 40)

    rows, cols = np.where(
        classification == class_id
    )

    total_pixels = len(rows)

    print("Available pixels :", total_pixels)

    if total_pixels < SAMPLES_PER_CLASS:

        raise ValueError(
            f"Not enough pixels in Class {class_id}"
        )

    # Randomly select pixels
    selected_indices = rng.choice(
        total_pixels,
        size=SAMPLES_PER_CLASS,
        replace=False
    )

    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]


    for row, col in zip(
        selected_rows,
        selected_cols
    ):

        # Convert pixel coordinates to map coordinates
        x, y = rasterio.transform.xy(
            transform,
            int(row),
            int(col),
            offset="center"
        )

        all_samples.append({

            "sample_id": len(all_samples) + 1,

            "row": int(row),

            "column": int(col),

            "x": float(x),

            "y": float(y),

            "predicted_class": int(class_id),

            "predicted_class_name":
                CLASS_NAMES[class_id],

            # This is intentionally blank.
            # YOU will fill this after visual/reference
            # interpretation.
            "reference_class": "",

            "reference_class_name": "",

            "confidence": "",

            "notes": ""

        })


    print(
        "Samples created   :",
        SAMPLES_PER_CLASS
    )


# ============================================================
# SHUFFLE SAMPLES
# ============================================================

rng.shuffle(all_samples)


# Reassign sample IDs after shuffling

for i, sample in enumerate(all_samples):

    sample["sample_id"] = i + 1


# ============================================================
# SAVE CSV
# ============================================================

print("\nSaving reference sample file...")

fieldnames = [
    "sample_id",
    "row",
    "column",
    "x",
    "y",
    "predicted_class",
    "predicted_class_name",
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        all_samples
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("REFERENCE SAMPLE GENERATION COMPLETE")
print("=" * 70)

print("\nTotal samples :", len(all_samples))

for class_id in CLASSES:

    count = sum(
        s["predicted_class"] == class_id
        for s in all_samples
    )

    print(
        f"Class {class_id} : {count} samples"
    )


print("\nCreated:")
print(OUTPUT_CSV)

print("\n")
print("=" * 70)
print("IMPORTANT NEXT STEP")
print("=" * 70)

print("""
The reference_class column is intentionally EMPTY.

Do NOT copy predicted_class into reference_class.

You must determine the actual/reference class for each
sample using independent geographic/reference information.

For example:

reference_class = 1
reference_class = 2
reference_class = 3
reference_class = 4
reference_class = 5

Then record the appropriate reference_class_name,
confidence and notes.

The reference labels should be based on independent
reference information, not the K-Means result itself.
""")

print("=" * 70)