
from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_FILE = RESULTS_DIR / "wayanad_kmeans_classification.tif"

FEATURE_FILES = {
    "HH_dB": INPUT_DIR / "wayanad_HH_dB.tif",
    "HV_dB": INPUT_DIR / "wayanad_HV_dB.tif",
    "HH_HV_ratio": INPUT_DIR / "wayanad_HH_HV_ratio.tif",
    "Normalized_Difference": (
        INPUT_DIR / "wayanad_normalized_difference.tif"
    ),
}

OUTPUT_CSV = RESULTS_DIR / "wayanad_class_statistics.csv"
OUTPUT_MAP = RESULTS_DIR / "wayanad_class_analysis.png"

WINDOW_SIZE = 512
PREVIEW_MAX_DIMENSION = 1800


# ============================================================
# 2. HELPERS
# ============================================================

def generate_windows(width, height, size):
    """Yield non-overlapping windows covering the raster."""
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(
                col_off=col,
                row_off=row,
                width=min(size, width - col),
                height=min(size, height - row),
            )


def read_valid_array(dataset, window):
    """
    Read one raster window and return its values plus a validity mask.
    Handles non-finite values and declared raster NoData values.
    """
    masked = dataset.read(1, window=window, masked=True)
    values = np.asarray(masked.data)
    valid = ~np.ma.getmaskarray(masked)
    valid &= np.isfinite(values)
    return values, valid


# ============================================================
# 3. CHECK FILES AND GRID ALIGNMENT
# ============================================================

print("=" * 70)
print("WAYANAD NISAR - CLASS ANALYSIS")
print("=" * 70)

if not CLASS_FILE.exists():
    raise FileNotFoundError(
        f"Classification raster not found:\n{CLASS_FILE}\n"
        "Run the revised classification script first."
    )

for name, path in FEATURE_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Feature raster not found for {name}:\n{path}"
        )

with rasterio.open(CLASS_FILE) as src:
    width = src.width
    height = src.height
    reference_crs = src.crs
    reference_transform = src.transform
    class_profile = src.profile.copy()
    class_nodata = src.nodata

if reference_crs is None:
    raise ValueError("Classification raster has no CRS.")

if not reference_crs.is_projected:
    raise ValueError(
        "Classification CRS is not projected. "
        "Check the CRS before calculating area."
    )

# Wayanad raster is expected to use UTM Zone 43N (EPSG:32643),
# whose coordinate units are metres.
if reference_crs.to_epsg() != 32643:
    raise ValueError(
        f"Expected WGS 84 / UTM zone 43N (EPSG:32643), "
        f"but found {reference_crs}. "
        "Verify the CRS and pixel-area units before proceeding."
    )

for name, path in FEATURE_FILES.items():
    with rasterio.open(path) as src:
        if (src.width, src.height) != (width, height):
            raise ValueError(
                f"Grid dimensions mismatch for {name}:\n{path}"
            )

        if src.crs != reference_crs:
            raise ValueError(
                f"CRS mismatch for {name}:\n{path}"
            )

        if src.transform != reference_transform:
            raise ValueError(
                f"Transform mismatch for {name}:\n{path}"
            )

print(f"\nClassification: {CLASS_FILE}")
print(f"Raster dimensions: {width} columns x {height} rows")
print(f"CRS: {reference_crs}")
print("All feature rasters match the classification grid.")


# ============================================================
# 4. ACCUMULATE STATISTICS WINDOW BY WINDOW
# ============================================================

feature_names = list(FEATURE_FILES.keys())
all_paths = [CLASS_FILE] + [
    FEATURE_FILES[name] for name in feature_names
]

# Each cluster gets exact counts and running sums from all valid
# pixels, without keeping the entire image in memory.
stats = {}

valid_pixel_count = 0
total_classified_pixel_count = 0
class_ids_seen = set()

pixel_area_m2 = abs(
    reference_transform.a * reference_transform.e
    - reference_transform.b * reference_transform.d
)
pixel_area_ha = pixel_area_m2 / 10000.0

print("\nCalculating cluster statistics...")

with (
    rasterio.open(CLASS_FILE) as class_src,
    rasterio.open(FEATURE_FILES["HH_dB"]) as hh_src,
    rasterio.open(FEATURE_FILES["HV_dB"]) as hv_src,
    rasterio.open(FEATURE_FILES["HH_HV_ratio"]) as ratio_src,
    rasterio.open(
        FEATURE_FILES["Normalized_Difference"]
    ) as nd_src,
):
    datasets = [hh_src, hv_src, ratio_src, nd_src]

    for window in generate_windows(width, height, WINDOW_SIZE):
        class_values, class_valid = read_valid_array(
            class_src, window
        )

        # Class 0 is the NoData/background value from the
        # classification workflow.
        classified = class_valid & (class_values > 0)
        total_classified_pixel_count += int(
            np.count_nonzero(classified)
        )

        feature_arrays = []
        common_valid = classified.copy()

        for dataset in datasets:
            values, feature_valid = read_valid_array(
                dataset, window
            )
            feature_arrays.append(values.astype(np.float64))
            common_valid &= feature_valid

        valid_pixel_count += int(np.count_nonzero(common_valid))

        if not np.any(common_valid):
            continue

        window_classes = class_values[common_valid].astype(np.int64)
        class_ids_seen.update(np.unique(window_classes).tolist())

        window_features = np.column_stack([
            values[common_valid] for values in feature_arrays
        ])

        for class_id in np.unique(window_classes):
            select = window_classes == class_id
            data = window_features[select]
            count = int(np.count_nonzero(select))

            if class_id not in stats:
                stats[class_id] = {
                    "count": 0,
                    "sum": np.zeros(len(feature_names), dtype=np.float64),
                    "sum_sq": np.zeros(len(feature_names), dtype=np.float64),
                }

            stats[class_id]["count"] += count
            stats[class_id]["sum"] += np.sum(data, axis=0)
            stats[class_id]["sum_sq"] += np.sum(
                data * data, axis=0
            )

if not stats or valid_pixel_count == 0:
    raise RuntimeError(
        "No valid classified pixels overlap all feature rasters. "
        "Check NoData masks and feature alignment."
    )

class_ids = sorted(stats.keys())

print(f"Detected cluster IDs: {class_ids}")
print(f"Classified pixels (class > 0): {total_classified_pixel_count:,}")
print(f"Valid pixels used in statistics: {valid_pixel_count:,}")

excluded = total_classified_pixel_count - valid_pixel_count
print(f"Classified pixels excluded due to invalid features: {excluded:,}")


# ============================================================
# 5. CALCULATE FINAL STATISTICS AND SAVE CSV
# ============================================================

csv_rows = []

print("\n" + "=" * 70)
print("CLASS STATISTICS")
print("=" * 70)

for class_id in class_ids:
    item = stats[class_id]
    count = item["count"]

    means = item["sum"] / count
    variances = item["sum_sq"] / count - means ** 2
    stds = np.sqrt(np.maximum(variances, 0.0))

    percentage = 100.0 * count / valid_pixel_count
    area_m2 = count * pixel_area_m2
    area_ha = count * pixel_area_ha

    row = {
        "Class_ID": int(class_id),
        "Pixels": int(count),
        "Percentage_of_valid_pixels": percentage,
        "Area_m2": area_m2,
        "Area_hectares": area_ha,
    }

    for i, name in enumerate(feature_names):
        row[f"Mean_{name}"] = means[i]
        row[f"Std_{name}"] = stds[i]

    csv_rows.append(row)

    print(f"\nCLUSTER {class_id}")
    print("-" * 45)
    print(f"Pixels          : {count:,}")
    print(f"Percentage      : {percentage:.2f}%")
    print(f"Area (m²)       : {area_m2:,.2f}")
    print(f"Area (hectares) : {area_ha:,.2f}")

    for i, name in enumerate(feature_names):
        print(
            f"{name}: mean={means[i]:.5f}, "
            f"std={stds[i]:.5f}"
        )

fieldnames = [
    "Class_ID",
    "Pixels",
    "Percentage_of_valid_pixels",
    "Area_m2",
    "Area_hectares",
]

for name in feature_names:
    fieldnames.extend([f"Mean_{name}", f"Std_{name}"])

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nStatistics CSV saved:\n{OUTPUT_CSV}")


# ============================================================
# 6. CREATE CLASSIFICATION PREVIEW
# ============================================================

print("\nCreating classification preview...")

preview_width = min(width, PREVIEW_MAX_DIMENSION)
preview_height = min(height, PREVIEW_MAX_DIMENSION)

with rasterio.open(CLASS_FILE) as src:
    preview = src.read(
        1,
        out_shape=(preview_height, preview_width),
        resampling=Resampling.nearest,
    )

# Mask zero/NoData; cluster IDs are discovered dynamically.
masked_preview = np.ma.masked_where(preview == 0, preview)

fig, ax = plt.subplots(figsize=(10, 12))

image = ax.imshow(
    masked_preview,
    cmap="tab20",
    interpolation="nearest",
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    ticks=class_ids,
)
colorbar.set_label("K-Means cluster ID")

ax.set_title("Wayanad NISAR - K-Means Class Analysis")
ax.set_xlabel("Pixel column")
ax.set_ylabel("Pixel row")

fig.tight_layout()
fig.savefig(OUTPUT_MAP, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Classification preview saved:\n{OUTPUT_MAP}")


# ============================================================
# 7. INTERPRETATION NOTES
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION NOTES")
print("=" * 70)

print("""
- Cluster IDs are arbitrary identifiers, not land-cover labels.
- HH and HV dB describe radar backscatter in decibels.
- HH/HV ratio and normalized difference are derived indicators.
- Similar feature statistics do not prove identical land cover.
- Use independent reference data before naming clusters as
  forest, water, built-up land, or disaster-damaged areas.
- Area estimates describe valid classified pixels only.
""")

print("=" * 70)
print("CLASS ANALYSIS COMPLETE")
print("=" * 70)
print(f"CSV: {OUTPUT_CSV}")
print(f"Map: {OUTPUT_MAP}")
