
"""
WAYANAD NISAR - CLASS VALIDATION AND INTERPRETATION

Purpose:
- Analyze the current K-Means classification.
- Detect actual class IDs dynamically.
- Validate raster alignment and handle NoData values.
- Calculate per-class pixel counts, percentages, areas, and feature statistics.
- Save a CSV, text report, and dynamically colored class map with editable cluster names.

Important:
K-Means classes represent clusters of radar features, not confirmed
land-cover categories. Independent reference data is required for
a true accuracy assessment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from rasterio.enums import Resampling


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the newer classification produced by the K-Means script.
CLASS_FILE = RESULTS_DIR / "wayanad_kmeans_classification.tif"

FEATURE_FILES = {
    "HH_dB": INPUT_DIR / "wayanad_HH_dB.tif",
    "HV_dB": INPUT_DIR / "wayanad_HV_dB.tif",
    "HH_HV_ratio": INPUT_DIR / "wayanad_HH_HV_ratio.tif",
    "Normalized_difference": INPUT_DIR / "wayanad_normalized_difference.tif",
}

VALIDATION_CSV = RESULTS_DIR / "wayanad_class_validation.csv"
VALIDATION_MAP = RESULTS_DIR / "wayanad_class_validation_map.png"
INTERPRETATION_TXT = RESULTS_DIR / "wayanad_class_interpretation.txt"

WINDOW_SIZE = 512

# ============================================================
# EDITABLE NAMES FOR THE SIX K-MEANS CLUSTERS
# ============================================================
# These are provisional display names, not automatic ground-truth labels.
# K-Means cluster IDs are arbitrary. Check the reference samples and
# per-cluster feature statistics, then change the names assigned to IDs.
# If a cluster does not fit a category, use a neutral name such as
# "Mixed / uncertain surface" rather than forcing a land-cover label.

CLASS_NAMES = {
    1: "Dense vegetation / forest",
    2: "Mixed vegetation",
    3: "Built-up",
    4: "Water / smooth surface",
    5: "Bare / exposed surface",
    6: "Wetland / waterlogged vegetation",
}

def get_class_name(class_id):
    """Return an editable descriptive name for a cluster ID."""
    return CLASS_NAMES.get(int(class_id), f"Unassigned radar cluster {int(class_id)}")


# ============================================================
# HELPERS
# ============================================================

def check_raster_alignment(reference, candidate, candidate_path):
    """Require matching dimensions, CRS, and affine transform."""

    if candidate.width != reference.width or candidate.height != reference.height:
        raise ValueError(
            f"Raster dimensions do not match:\n"
            f"Reference: {CLASS_FILE}\n"
            f"Other raster: {candidate_path}"
        )

    if candidate.crs != reference.crs:
        raise ValueError(
            f"CRS mismatch for {candidate_path}\n"
            f"Expected: {reference.crs}\n"
            f"Found: {candidate.crs}"
        )

    if not np.allclose(
        tuple(candidate.transform),
        tuple(reference.transform),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            f"Affine transform mismatch for {candidate_path}.\n"
            "The rasters must occupy the same pixel grid."
        )


def get_valid_values(dataset, window):
    """Read a raster window and return values plus a validity mask."""

    band = dataset.read(1, window=window, masked=True)
    values = np.asarray(band.data, dtype=np.float64)

    valid = ~np.ma.getmaskarray(band)
    valid &= np.isfinite(values)

    return values, valid


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("WAYANAD NISAR CLASS VALIDATION AND INTERPRETATION")
print("=" * 70)

required_files = [CLASS_FILE, *FEATURE_FILES.values()]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"\nRequired raster not found:\n{path}\n\n"
            "Check that the classification is saved in NISAR_results "
            "and the feature rasters are saved in NISAR_output."
        )

print("\nInput classification:")
print(CLASS_FILE)


# ============================================================
# OPEN RASTERS AND VALIDATE GRID
# ============================================================

with rasterio.open(CLASS_FILE) as class_src:

    if class_src.crs is None:
        raise ValueError("The classification raster has no CRS.")

    if not class_src.crs.is_projected:
        raise ValueError(
            f"Expected a projected CRS for area calculations; "
            f"found {class_src.crs}."
        )

    # Check that projected coordinate units are metres when possible.
    # Wayanad extraction is expected to use EPSG:32643.
    epsg = class_src.crs.to_epsg()
    if epsg != 32643:
        raise ValueError(
            f"Expected WGS 84 / UTM zone 43N (EPSG:32643), "
            f"but found {class_src.crs}. "
            "Verify the CRS and area units before proceeding."
        )

    profile = class_src.profile.copy()
    transform = class_src.transform
    crs = class_src.crs

    width = class_src.width
    height = class_src.height

    # Pixel area from the affine transform determinant.
    # This also accounts for rotated/sheared grids.
    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )

    if pixel_area_m2 <= 0:
        raise ValueError("Invalid pixel area derived from the raster transform.")

    feature_sources = {}

    try:
        for feature_name, feature_path in FEATURE_FILES.items():
            source = rasterio.open(feature_path)
            check_raster_alignment(class_src, source, feature_path)
            feature_sources[feature_name] = source

        print(f"\nRaster dimensions: {width} columns x {height} rows")
        print(f"CRS: {crs}")
        print(f"Pixel area: {pixel_area_m2:.2f} m²")

        # ----------------------------------------------------
        # FIRST PASS: discover actual class IDs
        # ----------------------------------------------------

        print("\nScanning classification for actual class IDs...")

        detected_classes = set()

        for row_off in range(0, height, WINDOW_SIZE):
            for col_off in range(0, width, WINDOW_SIZE):

                window = rasterio.windows.Window(
                    col_off,
                    row_off,
                    min(WINDOW_SIZE, width - col_off),
                    min(WINDOW_SIZE, height - row_off),
                )

                class_values, class_valid = get_valid_values(class_src, window)

                # Class 0 is reserved as background/nodata by the classifier.
                # Negative values are not treated as valid cluster IDs.
                valid_class = class_valid & (class_values > 0)

                if np.any(valid_class):
                    detected_classes.update(
                        np.unique(class_values[valid_class].astype(np.int64)).tolist()
                    )

        class_ids = sorted(detected_classes)

        if not class_ids:
            raise ValueError(
                "No valid class IDs greater than zero were found "
                "in the classification raster."
            )

        print("Detected class IDs:", class_ids)

        # ----------------------------------------------------
        # INITIALIZE STATISTICS
        # ----------------------------------------------------

        stats = {}

        for class_id in class_ids:
            stats[class_id] = {
                "count": 0,
                "features": {
                    name: {
                        "count": 0,
                        "sum": 0.0,
                        "sum_sq": 0.0,
                    }
                    for name in FEATURE_FILES
                },
            }

        total_common_valid = 0

        # ----------------------------------------------------
        # SECOND PASS: accumulate feature statistics
        # ----------------------------------------------------

        print("\nCalculating statistics in raster windows...")

        for row_off in range(0, height, WINDOW_SIZE):
            for col_off in range(0, width, WINDOW_SIZE):

                window = rasterio.windows.Window(
                    col_off,
                    row_off,
                    min(WINDOW_SIZE, width - col_off),
                    min(WINDOW_SIZE, height - row_off),
                )

                class_values, class_valid = get_valid_values(class_src, window)

                feature_arrays = {}
                common_valid = class_valid & (class_values > 0)

                for feature_name, source in feature_sources.items():
                    values, feature_valid = get_valid_values(source, window)
                    feature_arrays[feature_name] = values
                    common_valid &= feature_valid

                total_common_valid += int(np.count_nonzero(common_valid))

                if not np.any(common_valid):
                    continue

                window_classes = class_values[common_valid].astype(np.int64)

                for class_id in class_ids:
                    selected = window_classes == class_id
                    count = int(np.count_nonzero(selected))

                    if count == 0:
                        continue

                    stats[class_id]["count"] += count

                    for feature_name in FEATURE_FILES:
                        values = feature_arrays[feature_name][common_valid][selected]

                        feature_stats = stats[class_id]["features"][feature_name]
                        feature_stats["count"] += values.size
                        feature_stats["sum"] += float(np.sum(values, dtype=np.float64))
                        feature_stats["sum_sq"] += float(
                            np.sum(values * values, dtype=np.float64)
                        )

        if total_common_valid == 0:
            raise ValueError(
                "No pixels were simultaneously valid in the classification "
                "and all four feature rasters."
            )

        # ----------------------------------------------------
        # BUILD TABLE
        # ----------------------------------------------------

        records = []

        for class_id in class_ids:
            class_stats = stats[class_id]
            count = class_stats["count"]

            # If a detected class has no common valid feature pixels,
            # do not silently calculate statistics from a different mask.
            if count == 0:
                print(
                    f"Warning: class {class_id} has no pixels valid "
                    "across all feature rasters."
                )
                continue

            area_m2 = count * pixel_area_m2

            record = {
                "Class": int(class_id),
                "Class_name": get_class_name(class_id),
                "Interpretation": (
                    f"{get_class_name(class_id)} "
                    f"(provisional interpretation of radar cluster {class_id})"
                ),
                "Pixels": int(count),
                "Percentage_of_common_valid_pixels": (
                    count / total_common_valid * 100.0
                ),
                "Area_m2": area_m2,
                "Area_hectares": area_m2 / 10_000.0,
                "Area_km2": area_m2 / 1_000_000.0,
            }

            for feature_name in FEATURE_FILES:
                feature_stats = class_stats["features"][feature_name]
                n = feature_stats["count"]

                if n == 0:
                    mean = np.nan
                    std = np.nan
                else:
                    mean = feature_stats["sum"] / n
                    variance = feature_stats["sum_sq"] / n - mean**2
                    # Guard against tiny negative floating-point rounding errors.
                    std = np.sqrt(max(variance, 0.0))

                record[f"{feature_name}_mean"] = mean
                record[f"{feature_name}_std"] = std

            records.append(record)

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError("No classes had valid feature statistics.")

        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        df.to_csv(VALIDATION_CSV, index=False)

        print("\nSaved statistics:")
        print(VALIDATION_CSV)

        # ----------------------------------------------------
        # WRITE INTERPRETATION REPORT
        # ----------------------------------------------------

        with open(INTERPRETATION_TXT, "w", encoding="utf-8") as report:

            report.write("WAYANAD NISAR K-MEANS CLASS VALIDATION\n")
            report.write("=" * 65 + "\n\n")

            report.write(f"Classification raster: {CLASS_FILE}\n")
            report.write(f"CRS: {crs}\n")
            report.write(f"Raster dimensions: {width} x {height}\n")
            report.write(f"Pixel area: {pixel_area_m2:.2f} m²\n")
            report.write(f"Detected class IDs: {class_ids}\n")
            report.write(
                f"Common valid pixels used for statistics: "
                f"{total_common_valid:,}\n\n"
            )

            report.write("IMPORTANT LIMITATIONS\n")
            report.write("-" * 65 + "\n")
            report.write(
                "K-Means class numbers are arbitrary cluster identifiers. "
                "The statistics describe radar-feature distributions and "
                "do not independently establish land-cover identity.\n\n"
            )
            report.write(
                "The class percentages are calculated relative to pixels "
                "valid in the classification and all four feature rasters. "
                "They may differ from percentages based on the entire "
                "district or the classification raster alone.\n\n"
            )
            report.write(
                "These statistics are descriptive, not an independent "
                "accuracy assessment. Accuracy assessment requires "
                "independent reference labels and an appropriate sampling "
                "and validation procedure.\n\n"
            )

            report.write("CLASS STATISTICS\n")
            report.write("=" * 65 + "\n\n")

            for _, row in df.iterrows():
                report.write(f"CLASS {int(row['Class'])}: {row['Class_name']}\n")
                report.write("-" * 45 + "\n")
                report.write(
                    f"Interpretation: {row['Interpretation']}\n"
                )
                report.write(f"Pixels: {int(row['Pixels']):,}\n")
                report.write(
                    "Percentage of common valid pixels: "
                    f"{row['Percentage_of_common_valid_pixels']:.2f}%\n"
                )
                report.write(f"Area: {row['Area_km2']:.3f} km²\n")

                for feature_name in FEATURE_FILES:
                    report.write(
                        f"{feature_name} mean: "
                        f"{row[f'{feature_name}_mean']:.6f}\n"
                    )
                    report.write(
                        f"{feature_name} standard deviation: "
                        f"{row[f'{feature_name}_std']:.6f}\n"
                    )

                report.write("\n")

        print("Saved interpretation report:")
        print(INTERPRETATION_TXT)

        # ----------------------------------------------------
        # CREATE A DYNAMICALLY SIZED CLASS MAP
        # ----------------------------------------------------

        print("\nCreating class map...")

        # Read a reduced-resolution preview to avoid loading the
        # full-resolution classification into memory for plotting.
        max_preview_dimension = 1600
        scale = min(
            1.0,
            max_preview_dimension / max(width, height),
        )

        preview_width = max(1, int(width * scale))
        preview_height = max(1, int(height * scale))

        preview = class_src.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.nearest,
            masked=True,
        )

        preview_values = np.asarray(preview.data, dtype=np.float64)
        preview_mask = np.ma.getmaskarray(preview)
        preview_mask |= ~np.isfinite(preview_values)
        preview_mask |= preview_values <= 0

        display_classes = np.ma.array(
            preview_values,
            mask=preview_mask,
        )

        # Color and legend entries adapt to the IDs actually detected.
        cmap = ListedColormap(
            plt.get_cmap("tab20")(
                np.linspace(0, 1, max(len(class_ids), 1))
            )
        )
        cmap.set_bad(color="white")

        # The class IDs may not be consecutive, so use explicit boundaries.
        boundaries = np.arange(len(class_ids) + 1) + 0.5
        class_to_index = {
            class_id: index + 1
            for index, class_id in enumerate(class_ids)
        }

        # Remap class IDs to consecutive display indices for reliable
        # discrete colors while preserving original IDs in the legend.
        indexed_preview = np.zeros(display_classes.shape, dtype=np.float32)
        indexed_mask = np.ma.getmaskarray(display_classes)

        for class_id, display_index in class_to_index.items():
            indexed_preview[
                (~indexed_mask) & (display_classes.data == class_id)
            ] = display_index

        indexed_preview = np.ma.array(
            indexed_preview,
            mask=indexed_mask,
        )

        norm = BoundaryNorm(boundaries, cmap.N)

        fig, ax = plt.subplots(figsize=(12, 10))

        ax.imshow(
            indexed_preview,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(
            "Wayanad NISAR - K-Means Radar-Scattering Clusters",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Preview pixel column")
        ax.set_ylabel("Preview pixel row")

        legend_elements = [
            Patch(
                facecolor=cmap(class_to_index[class_id] - 1),
                label=f"Class {class_id}: {get_class_name(class_id)}",
            )
            for class_id in class_ids
        ]

        ax.legend(
            handles=legend_elements,
            title="Cluster ID and provisional name",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=9,
        )

        fig.text(
            0.5,
            0.015,
            "Names are provisional; verify each cluster against independent reference labels. Cluster IDs are arbitrary.",
            ha="center",
            fontsize=9,
        )

        plt.tight_layout(rect=(0, 0.04, 0.82, 1))
        plt.savefig(
            VALIDATION_MAP,
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Saved class map:")
        print(VALIDATION_MAP)

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("CLASS VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Detected class IDs: {class_ids}")
        print(f"Common valid pixels: {total_common_valid:,}")
        print(f"Pixel area: {pixel_area_m2:.2f} m²")
        print("\nClass areas and percentages:")
        print(
            df[
                [
                    "Class",
                    "Pixels",
                    "Percentage_of_common_valid_pixels",
                    "Area_km2",
                ]
            ].to_string(index=False)
        )

        print("\nCreated:")
        print(VALIDATION_CSV)
        print(VALIDATION_MAP)
        print(INTERPRETATION_TXT)

        print(
            "\nNote: this is descriptive cluster analysis, not "
            "independent validation of land-cover accuracy."
        )

    finally:
        for source in feature_sources.values():
            source.close()

print("\nDone.")
