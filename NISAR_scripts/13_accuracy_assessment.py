import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    classification_report
)


# ============================================================
# WAYANAD NISAR ACCURACY ASSESSMENT
# ============================================================

print("=" * 70)
print("WAYANAD NISAR CLASSIFICATION ACCURACY ASSESSMENT")
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

REFERENCE_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_reference_classes.tif"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "NISAR_results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


CSV_FILE = os.path.join(
    RESULTS_DIR,
    "wayanad_accuracy_statistics.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULTS_DIR,
    "wayanad_confusion_matrix.png"
)

REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "wayanad_accuracy_report.txt"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CLASSIFICATION_FILE):
    raise FileNotFoundError(
        f"\nClassification file not found:\n{CLASSIFICATION_FILE}"
    )

if not os.path.exists(REFERENCE_FILE):
    raise FileNotFoundError(
        f"""
Reference raster not found:

{REFERENCE_FILE}

Create/import an independent reference raster first.
"""
    )


# ============================================================
# LOAD CLASSIFICATION
# ============================================================

print("\nLoading K-Means classification...")

with rasterio.open(CLASSIFICATION_FILE) as src:

    classification = src.read(1)

    class_transform = src.transform
    class_crs = src.crs
    class_width = src.width
    class_height = src.height
    class_profile = src.profile

print("Classification shape:", classification.shape)
print("Classification CRS  :", class_crs)
print("Classification size  :", class_width, "x", class_height)


# ============================================================
# LOAD REFERENCE
# ============================================================

print("\nLoading reference data...")

with rasterio.open(REFERENCE_FILE) as src:

    reference = src.read(1)

    ref_transform = src.transform
    ref_crs = src.crs
    ref_width = src.width
    ref_height = src.height

print("Reference shape:", reference.shape)
print("Reference CRS  :", ref_crs)
print("Reference size :", ref_width, "x", ref_height)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

print("\nChecking raster alignment...")

if classification.shape != reference.shape:
    raise ValueError(
        """
Classification and reference rasters have different dimensions.

Both rasters must be aligned to the same grid.

Use QGIS/GDAL to align the reference raster to:
wayanad_kmeans_classes.tif
"""
    )

if class_crs != ref_crs:
    raise ValueError(
        """
Classification and reference rasters have different CRS.

Reproject the reference raster to the classification CRS.
"""
    )

if class_transform != ref_transform:
    raise ValueError(
        """
Classification and reference rasters have different geotransforms.

The reference raster must be spatially aligned with
the classification raster.
"""
    )

print("Raster alignment: OK")


# ============================================================
# DEFINE VALID CLASSES
# ============================================================

CLASSES = [1, 2, 3, 4, 5]

CLASS_NAMES = {
    1: "Mixed vegetation / agricultural",
    2: "Built-up / exposed structured",
    3: "Strong structural / double-bounce-like",
    4: "Low-backscatter smooth surface",
    5: "Forest / dense vegetation"
}


# ============================================================
# CREATE VALID MASK
# ============================================================

print("\nCreating validation mask...")

valid_mask = (
    np.isin(classification, CLASSES)
    &
    np.isin(reference, CLASSES)
)

y_pred = classification[valid_mask]
y_true = reference[valid_mask]

print("Total validation pixels:", len(y_true))


if len(y_true) == 0:
    raise ValueError(
        "No valid validation pixels found."
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCalculating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASSES
)

print("\nConfusion Matrix")
print("-" * 50)

print("Reference rows")
print("Classification columns\n")

print(cm)


# ============================================================
# ACCURACY METRICS
# ============================================================

overall_accuracy = accuracy_score(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average=None,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average=None,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average=None,
    zero_division=0
)

iou = jaccard_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average=None,
    zero_division=0
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OVERALL ACCURACY RESULTS")
print("=" * 70)

print(f"\nOverall Accuracy : {overall_accuracy * 100:.2f}%")
print(f"Cohen's Kappa    : {kappa:.4f}")


# ============================================================
# CLASS STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CLASS-WISE ACCURACY")
print("=" * 70)

print(
    f"{'Class':<8}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1':<15}"
    f"{'IoU':<15}"
)

print("-" * 70)

for i, class_id in enumerate(CLASSES):

    print(
        f"{class_id:<8}"
        f"{precision[i] * 100:<15.2f}"
        f"{recall[i] * 100:<15.2f}"
        f"{f1[i] * 100:<15.2f}"
        f"{iou[i] * 100:<15.2f}"
    )


# ============================================================
# AREA CALCULATION
# ============================================================

print("\n")
print("=" * 70)
print("REFERENCE VS CLASSIFIED AREA")
print("=" * 70)

pixel_width = abs(class_transform.a)
pixel_height = abs(class_transform.e)

pixel_area = pixel_width * pixel_height

print(f"\nPixel width  : {pixel_width:.2f} m")
print(f"Pixel height : {pixel_height:.2f} m")
print(f"Pixel area   : {pixel_area:.2f} m²")


# ============================================================
# CSV OUTPUT
# ============================================================

print("\nSaving accuracy statistics...")

with open(CSV_FILE, "w", encoding="utf-8") as f:

    f.write(
        "Class,Class_Name,Reference_Pixels,"
        "Classified_Pixels,Reference_Area_ha,"
        "Classified_Area_ha,Precision,Recall,F1,IoU\n"
    )

    for i, class_id in enumerate(CLASSES):

        reference_pixels = np.sum(y_true == class_id)
        classified_pixels = np.sum(y_pred == class_id)

        reference_area_ha = (
            reference_pixels * pixel_area / 10000
        )

        classified_area_ha = (
            classified_pixels * pixel_area / 10000
        )

        f.write(
            f"{class_id},"
            f"{CLASS_NAMES[class_id]},"
            f"{reference_pixels},"
            f"{classified_pixels},"
            f"{reference_area_ha:.4f},"
            f"{classified_area_ha:.4f},"
            f"{precision[i]:.6f},"
            f"{recall[i]:.6f},"
            f"{f1[i]:.6f},"
            f"{iou[i]:.6f}\n"
        )

print("Saved:")
print(CSV_FILE)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

print("\nSaving accuracy report...")

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("WAYANAD NISAR CLASSIFICATION ACCURACY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("INPUT DATA\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Classification:\n{CLASSIFICATION_FILE}\n\n"
    )

    f.write(
        f"Reference:\n{REFERENCE_FILE}\n\n"
    )

    f.write(
        f"Validation pixels: {len(y_true)}\n"
    )

    f.write(
        f"Pixel area: {pixel_area:.2f} m²\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("OVERALL ACCURACY\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Overall Accuracy : "
        f"{overall_accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Cohen's Kappa    : "
        f"{kappa:.6f}\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        "Rows = Reference\n"
        "Columns = Classification\n\n"
    )

    f.write(str(cm))
    f.write("\n\n")

    f.write("=" * 70 + "\n")
    f.write("CLASS-WISE ACCURACY\n")
    f.write("=" * 70 + "\n\n")

    for i, class_id in enumerate(CLASSES):

        f.write(
            f"CLASS {class_id}\n"
        )

        f.write(
            f"Name      : {CLASS_NAMES[class_id]}\n"
        )

        f.write(
            f"Precision : {precision[i] * 100:.2f}%\n"
        )

        f.write(
            f"Recall    : {recall[i] * 100:.2f}%\n"
        )

        f.write(
            f"F1-score  : {f1[i] * 100:.2f}%\n"
        )

        f.write(
            f"IoU       : {iou[i] * 100:.2f}%\n\n"
        )

    f.write("=" * 70 + "\n")
    f.write("CLASS INTERPRETATIONS\n")
    f.write("=" * 70 + "\n\n")

    for class_id in CLASSES:

        f.write(
            f"Class {class_id}: "
            f"{CLASS_NAMES[class_id]}\n"
        )


print("Saved:")
print(REPORT_FILE)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\nCreating confusion matrix map...")

fig, ax = plt.subplots(
    figsize=(10, 8)
)

im = ax.imshow(
    cm,
    interpolation="nearest"
)

ax.set_title(
    "Wayanad NISAR Classification\nConfusion Matrix"
)

ax.set_xlabel(
    "Predicted / K-Means Class"
)

ax.set_ylabel(
    "Reference Class"
)

ax.set_xticks(
    np.arange(len(CLASSES))
)

ax.set_yticks(
    np.arange(len(CLASSES))
)

ax.set_xticklabels(CLASSES)
ax.set_yticklabels(CLASSES)


# Add numbers inside cells

for i in range(len(CLASSES)):

    for j in range(len(CLASSES)):

        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )


fig.colorbar(
    im,
    ax=ax,
    label="Number of Pixels"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:")
print(CONFUSION_MATRIX_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ACCURACY ASSESSMENT COMPLETE")
print("=" * 70)

print("\nCreated:")

print(CSV_FILE)
print(CONFUSION_MATRIX_FILE)
print(REPORT_FILE)

print("\n")
print(f"Overall Accuracy : {overall_accuracy * 100:.2f}%")
print(f"Cohen's Kappa    : {kappa:.4f}")

print("\nNext step:")
print("Review the confusion matrix and class-wise accuracy.")
print("Then prepare the final land-cover/disaster interpretation.")