
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "HH_dB": INPUT_DIR / "wayanad_HH_dB.tif",
    "HV_dB": INPUT_DIR / "wayanad_HV_dB.tif",
    "Normalized_Difference": (
        INPUT_DIR / "wayanad_normalized_difference.tif"
    ),
}

OUTPUT_TIF = RESULTS_DIR / "wayanad_kmeans_classification.tif"
OUTPUT_PNG = RESULTS_DIR / "wayanad_kmeans_classification.png"

N_CLUSTERS = 6
SAMPLES_PER_WINDOW = 1000
WINDOW_SIZE = 512
RANDOM_SEED = 42



# ============================================================
# 2. CHECK INPUT FILES AND RASTER ALIGNMENT
# ============================================================

for name, path in FEATURE_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature raster: {path}\n"
            f"Check the output filename from script 8."
        )

feature_names = list(FEATURE_FILES.keys())
feature_paths = [FEATURE_FILES[name] for name in feature_names]

with rasterio.open(feature_paths[0]) as reference:
    reference_profile = reference.profile.copy()
    width = reference.width
    height = reference.height
    reference_crs = reference.crs
    reference_transform = reference.transform

for path in feature_paths[1:]:
    with rasterio.open(path) as src:
        if src.width != width or src.height != height:
            raise ValueError(
                f"Raster dimensions do not match: {path.name}"
            )

        if src.crs != reference_crs:
            raise ValueError(
                f"CRS does not match: {path.name}"
            )

        if src.transform != reference_transform:
            raise ValueError(
                f"Raster transform does not match: {path.name}"
            )

print("Input rasters aligned successfully.")
print(f"Raster dimensions: {width} x {height}")
print(f"CRS: {reference_crs}")
print(f"Features used: {', '.join(feature_names)}")


# ============================================================
# 3. WINDOW GENERATOR
# ============================================================

def generate_windows(raster_width, raster_height, window_size):
    """Yield windows that cover the raster without overlap."""
    for row_off in range(0, raster_height, window_size):
        for col_off in range(0, raster_width, window_size):
            win_width = min(window_size, raster_width - col_off)
            win_height = min(window_size, raster_height - row_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=win_width,
                height=win_height,
            )


# ============================================================
# 4. READ FEATURES FOR ONE WINDOW
# ============================================================

def read_feature_window(datasets, window):
    """
    Return a feature matrix and a validity mask for one window.
    Invalid pixels are excluded from clustering.
    """
    arrays = [
        ds.read(1, window=window).astype(np.float32)
        for ds in datasets
    ]

    valid = np.ones(arrays[0].shape, dtype=bool)

    for array in arrays:
        valid &= np.isfinite(array)

    # Flatten valid pixel values into (pixels, features).
    feature_matrix = np.column_stack(
        [array[valid] for array in arrays]
    )

    return feature_matrix, valid


# ============================================================
# 5. COLLECT A REPRESENTATIVE TRAINING SAMPLE
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)
sample_parts = []

print("\nCollecting training samples from raster windows...")

with (
    rasterio.open(FEATURE_FILES["HH_dB"]) as hh_src,
    rasterio.open(FEATURE_FILES["HV_dB"]) as hv_src,
    rasterio.open(FEATURE_FILES["Normalized_Difference"]) as nd_src,
):
    datasets = [hh_src, hv_src, nd_src]

    for window in generate_windows(width, height, WINDOW_SIZE):
        features, valid = read_feature_window(datasets, window)

        if features.shape[0] == 0:
            continue

        n_to_take = min(SAMPLES_PER_WINDOW, features.shape[0])

        if features.shape[0] > n_to_take:
            chosen = rng.choice(
                features.shape[0],
                size=n_to_take,
                replace=False,
            )
            features = features[chosen]

        sample_parts.append(features)

if not sample_parts:
    raise RuntimeError(
        "No valid pixels found. Check the feature rasters "
        "and their NoData values."
    )

training_sample = np.concatenate(sample_parts, axis=0)

if training_sample.shape[0] < N_CLUSTERS:
    raise RuntimeError(
        f"Only {training_sample.shape[0]} valid samples found; "
        f"cannot fit {N_CLUSTERS} clusters."
    )

print(f"Training samples collected: {len(training_sample):,}")


# ============================================================
# 6. SCALE FEATURES AND TRAIN K-MEANS
# ============================================================

print("\nScaling training features...")

scaler = StandardScaler()
training_scaled = scaler.fit_transform(training_sample)

print("Training K-Means model...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    n_init=10,
)

kmeans.fit(training_scaled)

print("K-Means training complete.")
print(f"Number of clusters: {N_CLUSTERS}")


# ============================================================
# 7. CLASSIFY THE FULL RASTER IN WINDOWS
# ============================================================

output_profile = reference_profile.copy()
output_profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw",
    BIGTIFF="IF_SAFER",
)

class_counts = np.zeros(N_CLUSTERS + 1, dtype=np.int64)
valid_pixel_count = 0

print("\nClassifying the full raster in windows...")

with (
    rasterio.open(FEATURE_FILES["HH_dB"]) as hh_src,
    rasterio.open(FEATURE_FILES["HV_dB"]) as hv_src,
    rasterio.open(FEATURE_FILES["Normalized_Difference"]) as nd_src,
    rasterio.open(OUTPUT_TIF, "w", **output_profile) as out_dst,
):
    datasets = [hh_src, hv_src, nd_src]

    for window in generate_windows(width, height, WINDOW_SIZE):
        features, valid = read_feature_window(datasets, window)

        # Zero represents NoData. Valid clusters are numbered 1 to 4.
        output_array = np.zeros(valid.shape, dtype=np.uint8)

        if features.shape[0] > 0:
            scaled_features = scaler.transform(features)
            predictions = kmeans.predict(scaled_features)

            output_array[valid] = (predictions + 1).astype(np.uint8)

            counts = np.bincount(
                predictions,
                minlength=N_CLUSTERS,
            )
            class_counts[1:] += counts
            valid_pixel_count += len(predictions)

        out_dst.write(output_array, 1, window=window)

print(f"\nClassification raster saved:\n{OUTPUT_TIF}")
print(f"Valid classified pixels: {valid_pixel_count:,}")

for cluster_id in range(1, N_CLUSTERS + 1):
    count = int(class_counts[cluster_id])
    percentage = (
        100.0 * count / valid_pixel_count
        if valid_pixel_count
        else 0.0
    )
    print(
        f"Cluster {cluster_id}: {count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 8. CREATE A LOWER-RESOLUTION VISUALIZATION
# ============================================================

print("\nCreating classification preview...")

preview_max_dimension = 1500
scale_factor = min(
    1.0,
    preview_max_dimension / max(width, height),
)

preview_width = max(1, int(width * scale_factor))
preview_height = max(1, int(height * scale_factor))

with rasterio.open(OUTPUT_TIF) as src:
    preview = src.read(
        1,
        out_shape=(preview_height, preview_width),
        resampling=rasterio.enums.Resampling.nearest,
    )

masked_preview = np.ma.masked_equal(preview, 0)

fig, ax = plt.subplots(figsize=(10, 8))

image = ax.imshow(
    masked_preview,
    cmap="tab10",
    vmin=1,
    vmax=N_CLUSTERS,
    interpolation="nearest",
)

colorbar = fig.colorbar(
    image,
    ax=ax,
    ticks=list(range(1, N_CLUSTERS + 1)),
)
colorbar.set_label("K-Means cluster ID")

ax.set_title("Wayanad NISAR GCOV — K-Means Clustering")
ax.set_xlabel("Raster column")
ax.set_ylabel("Raster row")

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Classification preview saved:\n{OUTPUT_PNG}")
print("\nDone.")
print(
    "Note: cluster IDs indicate radar-scattering groups; "
    "they are not validated land-cover labels."
)
