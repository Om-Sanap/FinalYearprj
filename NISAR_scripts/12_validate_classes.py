"""
WAYANAD NISAR - CLASS VALIDATION AND INTERPRETATION
====================================================

Purpose:
    Validate and interpret K-Means classes using:
    - HH dB
    - HV dB
    - HH/HV ratio
    - Normalized Difference
    - Spatial distribution

IMPORTANT:
    K-Means class numbers are arbitrary.
    The land-cover interpretations produced here are PROVISIONAL
    and should be validated against reference/ground-truth data.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\FinalYearPrj\prj"

INPUT_DIR = os.path.join(BASE_DIR, "NISAR_output")
RESULTS_DIR = os.path.join(BASE_DIR, "NISAR_results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

CLASS_FILE = os.path.join(
    INPUT_DIR,
    "wayanad_kmeans_classes.tif"
)

HH_FILE = os.path.join(
    INPUT_DIR,
    "wayanad_HH_dB.tif"
)

HV_FILE = os.path.join(
    INPUT_DIR,
    "wayanad_HV_dB.tif"
)

RATIO_FILE = os.path.join(
    INPUT_DIR,
    "wayanad_HH_HV_ratio.tif"
)

ND_FILE = os.path.join(
    INPUT_DIR,
    "wayanad_normalized_difference.tif"
)


# ============================================================
# OUTPUT FILES
# ============================================================

VALIDATION_CSV = os.path.join(
    RESULTS_DIR,
    "wayanad_class_validation.csv"
)

VALIDATION_MAP = os.path.join(
    RESULTS_DIR,
    "wayanad_class_validation_map.png"
)

INTERPRETATION_TXT = os.path.join(
    RESULTS_DIR,
    "wayanad_class_interpretation.txt"
)


# ============================================================
# DISPLAY HEADER
# ============================================================

print("=" * 70)
print("WAYANAD NISAR CLASS VALIDATION AND INTERPRETATION")
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

files = [
    CLASS_FILE,
    HH_FILE,
    HV_FILE,
    RATIO_FILE,
    ND_FILE
]

for file in files:

    if not os.path.exists(file):

        print("\nERROR:")
        print("File not found:")
        print(file)

        raise FileNotFoundError(file)


# ============================================================
# LOAD RASTERS
# ============================================================

print("\nLoading classification...")
with rasterio.open(CLASS_FILE) as src:

    classes = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    nodata = src.nodata


print("Loading HH dB...")
with rasterio.open(HH_FILE) as src:
    hh = src.read(1)


print("Loading HV dB...")
with rasterio.open(HV_FILE) as src:
    hv = src.read(1)


print("Loading HH/HV ratio...")
with rasterio.open(RATIO_FILE) as src:
    ratio = src.read(1)


print("Loading normalized difference...")
with rasterio.open(ND_FILE) as src:
    nd = src.read(1)


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nImage shape:", classes.shape)

if not (
    classes.shape == hh.shape
    == hv.shape
    == ratio.shape
    == nd.shape
):

    raise ValueError(
        "Input rasters do not have identical dimensions."
    )


# ============================================================
# VALID PIXEL MASK
# ============================================================

valid = (
    np.isfinite(classes)
    & np.isfinite(hh)
    & np.isfinite(hv)
    & np.isfinite(ratio)
    & np.isfinite(nd)
    & (classes > 0)
)

valid_pixels = np.sum(valid)

print("Valid pixels:", valid_pixels)


# ============================================================
# DETERMINE PIXEL AREA
# ============================================================

pixel_width = abs(transform.a)
pixel_height = abs(transform.e)

pixel_area_m2 = pixel_width * pixel_height

print("\nPixel dimensions:")
print("Width  :", pixel_width, "m")
print("Height :", pixel_height, "m")
print("Area   :", pixel_area_m2, "m²")


# ============================================================
# CLASS INTERPRETATION
# ============================================================

interpretations = {

    1: {
        "label": "Mixed vegetation / agricultural surface",
        "description":
            "Moderate HH and HV response with intermediate "
            "HH/HV ratio; may represent mixed vegetation, "
            "agricultural land or moderately structured surfaces."
    },

    2: {
        "label": "Built-up / exposed structured surface",
        "description":
            "Strong HH dominance and relatively high HH/HV "
            "ratio; potentially associated with built-up, "
            "exposed or strongly structured surfaces."
    },

    3: {
        "label": "Strong structural / double-bounce-like scattering",
        "description":
            "Very strong HH dominance, very low HV and high "
            "HH/HV ratio; potentially associated with "
            "strongly structured or anthropogenic surfaces."
    },

    4: {
        "label": "Low-backscatter smooth surface",
        "description":
            "Relatively low HH and HV backscatter; may include "
            "water, radar-shadow, smooth terrain or other "
            "low-scattering surfaces."
    },

    5: {
        "label": "Forest / dense vegetation",
        "description":
            "Relatively strong HV response, low HH/HV ratio "
            "and low normalized difference; consistent with "
            "vegetation-related volume scattering."
    }
}


# ============================================================
# CLASS STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CLASS VALIDATION STATISTICS")
print("=" * 70)


class_numbers = np.unique(
    classes[valid].astype(np.int32)
)

records = []


for class_id in class_numbers:

    class_mask = valid & (
        classes == class_id
    )

    count = np.sum(class_mask)

    percentage = (
        count / valid_pixels
    ) * 100.0

    area_m2 = (
        count * pixel_area_m2
    )

    area_ha = (
        area_m2 / 10000.0
    )

    area_km2 = (
        area_m2 / 1_000_000.0
    )

    hh_values = hh[class_mask]
    hv_values = hv[class_mask]
    ratio_values = ratio[class_mask]
    nd_values = nd[class_mask]

    hh_mean = np.mean(hh_values)
    hh_std = np.std(hh_values)

    hv_mean = np.mean(hv_values)
    hv_std = np.std(hv_values)

    ratio_mean = np.mean(ratio_values)
    ratio_std = np.std(ratio_values)

    nd_mean = np.mean(nd_values)
    nd_std = np.std(nd_values)

    if class_id in interpretations:

        label = interpretations[class_id]["label"]

        description = interpretations[class_id][
            "description"
        ]

    else:

        label = "Unclassified"
        description = "No predefined interpretation."

    print("\n")
    print("CLASS", class_id)
    print("-" * 50)

    print("Interpretation :", label)

    print("Pixels         :", count)
    print("Percentage     :", f"{percentage:.2f}%")

    print("Area (m²)      :", f"{area_m2:.2f}")
    print("Area (ha)      :", f"{area_ha:.2f}")
    print("Area (km²)     :", f"{area_km2:.2f}")

    print("\nHH dB")
    print("  Mean :", f"{hh_mean:.4f}")
    print("  Std  :", f"{hh_std:.4f}")

    print("\nHV dB")
    print("  Mean :", f"{hv_mean:.4f}")
    print("  Std  :", f"{hv_std:.4f}")

    print("\nHH/HV Ratio")
    print("  Mean :", f"{ratio_mean:.4f}")
    print("  Std  :", f"{ratio_std:.4f}")

    print("\nNormalized Difference")
    print("  Mean :", f"{nd_mean:.4f}")
    print("  Std  :", f"{nd_std:.4f}")

    records.append({

        "Class": int(class_id),

        "Interpretation": label,

        "Pixels": int(count),

        "Percentage": percentage,

        "Area_m2": area_m2,

        "Area_hectares": area_ha,

        "Area_km2": area_km2,

        "HH_mean_dB": hh_mean,
        "HH_std_dB": hh_std,

        "HV_mean_dB": hv_mean,
        "HV_std_dB": hv_std,

        "HH_HV_ratio_mean": ratio_mean,
        "HH_HV_ratio_std": ratio_std,

        "Normalized_difference_mean": nd_mean,
        "Normalized_difference_std": nd_std,

        "Description": description
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE CSV
# ============================================================

print("\n")
print("Saving validation statistics...")

df.to_csv(
    VALIDATION_CSV,
    index=False
)

print("Saved:")
print(VALIDATION_CSV)


# ============================================================
# CREATE INTERPRETATION TEXT FILE
# ============================================================

print("\nCreating interpretation report...")

with open(
    INTERPRETATION_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "WAYANAD NISAR K-MEANS CLASS INTERPRETATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        "IMPORTANT:\n"
        "K-Means class numbers are arbitrary. The interpretations "
        "below are provisional and must be validated using "
        "reference data or known geographic features.\n\n"
    )

    for _, row in df.iterrows():

        f.write(
            f"CLASS {int(row['Class'])}\n"
        )

        f.write("-" * 50 + "\n")

        f.write(
            f"Interpretation: "
            f"{row['Interpretation']}\n"
        )

        f.write(
            f"Percentage: "
            f"{row['Percentage']:.2f}%\n"
        )

        f.write(
            f"Area: "
            f"{row['Area_km2']:.2f} km²\n"
        )

        f.write(
            f"HH mean: "
            f"{row['HH_mean_dB']:.4f} dB\n"
        )

        f.write(
            f"HV mean: "
            f"{row['HV_mean_dB']:.4f} dB\n"
        )

        f.write(
            f"HH/HV ratio: "
            f"{row['HH_HV_ratio_mean']:.4f}\n"
        )

        f.write(
            f"Normalized difference: "
            f"{row['Normalized_difference_mean']:.4f}\n"
        )

        f.write(
            f"\nDescription:\n"
            f"{row['Description']}\n\n"
        )


print("Saved:")
print(INTERPRETATION_TXT)


# ============================================================
# CREATE CLEAN CLASS MAP
# ============================================================

print("\nCreating class interpretation map...")


# Create masked array
display_classes = classes.astype(float)

display_classes[~valid] = np.nan


# ------------------------------------------------------------
# Custom colors for five classes
# ------------------------------------------------------------

class_colors = [
    "#7FBF7B",   # Class 1
    "#E69F00",   # Class 2
    "#D73027",   # Class 3
    "#4C78A8",   # Class 4
    "#2C7FB8"    # Class 5
]

cmap = ListedColormap(class_colors)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)


im = ax.imshow(
    display_classes,
    cmap=cmap,
    vmin=1,
    vmax=5,
    interpolation="nearest"
)


ax.set_title(
    "Wayanad NISAR - Validated K-Means Class Interpretation",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Pixel Column"
)

ax.set_ylabel(
    "Pixel Row"
)


# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    im,
    ax=ax,
    ticks=[1, 2, 3, 4, 5],
    fraction=0.046,
    pad=0.04
)

cbar.set_label(
    "Provisional Land-Cover / Scattering Class"
)


# ============================================================
# CLASS LABELS
# ============================================================

class_labels = []

for class_id in range(1, 6):

    if class_id in interpretations:

        class_labels.append(
            f"{class_id}: "
            f"{interpretations[class_id]['label']}"
        )

    else:

        class_labels.append(
            f"{class_id}: Unknown"
        )


# Add legend
from matplotlib.patches import Patch

legend_elements = []

for i in range(5):

    legend_elements.append(
        Patch(
            facecolor=class_colors[i],
            label=class_labels[i]
        )
    )


ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    fontsize=9,
    frameon=True
)


plt.tight_layout()


# ============================================================
# SAVE MAP
# ============================================================

plt.savefig(
    VALIDATION_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:")
print(VALIDATION_MAP)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CLASS VALIDATION COMPLETE")
print("=" * 70)

print("\nCreated:")

print(VALIDATION_CSV)
print(VALIDATION_MAP)
print(INTERPRETATION_TXT)

print("\n")
print("PROVISIONAL CLASS INTERPRETATION")
print("=" * 70)

for _, row in df.iterrows():

    print(
        f"Class {int(row['Class'])}: "
        f"{row['Interpretation']} "
        f"({row['Percentage']:.2f}%)"
    )

print("\n")
print("IMPORTANT:")
print(
    "These are provisional interpretations based on "
    "NISAR scattering characteristics."
)

print(
    "Final land-cover labels require validation "
    "against reference data."
)

print("\nNext step:")
print(
    "Validate the classes using geographic/reference data "
    "and calculate classification accuracy."
)

print("=" * 70)