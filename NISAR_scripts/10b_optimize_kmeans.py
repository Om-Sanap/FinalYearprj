
from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.windows import Window

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use the same three features as the revised classifier.
FEATURE_FILES = {
    "HH_dB": INPUT_DIR / "wayanad_HH_dB.tif",
    "HV_dB": INPUT_DIR / "wayanad_HV_dB.tif",
    "Normalized_Difference": (
        INPUT_DIR / "wayanad_normalized_difference.tif"
    ),
}

CSV_PATH = RESULTS_DIR / "wayanad_kmeans_optimization.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

K_VALUES = range(3, 9)

MAX_TRAINING_SAMPLES = 100_000
SILHOUETTE_SAMPLE_SIZE = 5_000

WINDOW_SIZE = 512
RANDOM_STATE = 42


# ============================================================
# 3. WINDOW AND FEATURE HELPERS
# ============================================================

def generate_windows(width, height, size):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(
                col_off=col,
                row_off=row,
                width=min(size, width - col),
                height=min(size, height - row),
            )


def read_features(datasets, window):
    arrays = [
        ds.read(1, window=window).astype(np.float32)
        for ds in datasets
    ]

    valid = np.ones(arrays[0].shape, dtype=bool)

    for array in arrays:
        valid &= np.isfinite(array)

    features = np.column_stack(
        [array[valid] for array in arrays]
    )

    return features


# ============================================================
# 4. CHECK INPUTS AND GRID ALIGNMENT
# ============================================================

for name, path in FEATURE_FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found for {name}:\n{path}"
        )

feature_paths = list(FEATURE_FILES.values())

with rasterio.open(feature_paths[0]) as ref:
    width = ref.width
    height = ref.height
    ref_crs = ref.crs
    ref_transform = ref.transform

for path in feature_paths[1:]:
    with rasterio.open(path) as src:
        if (src.width, src.height) != (width, height):
            raise ValueError(
                f"Raster dimensions do not match: {path.name}"
            )
        if src.crs != ref_crs:
            raise ValueError(f"CRS mismatch: {path.name}")
        if src.transform != ref_transform:
            raise ValueError(
                f"Raster transform mismatch: {path.name}"
            )

print("=" * 70)
print("WAYANAD NISAR K-MEANS OPTIMIZATION")
print("=" * 70)
print(f"Raster size: {width} x {height}")
print(f"Features: {', '.join(FEATURE_FILES.keys())}")
print(f"K values: {list(K_VALUES)}")
print("Input rasters aligned successfully.")


# ============================================================
# 5. FIRST PASS: COUNT VALID PIXELS PER WINDOW
# ============================================================

print("\nCounting valid pixels by window...")

window_info = []
total_valid = 0


# Open all feature rasters together.
from contextlib import ExitStack

with ExitStack() as stack:
    datasets = [
        stack.enter_context(rasterio.open(path))
        for path in feature_paths
    ]

    for window in generate_windows(width, height, WINDOW_SIZE):
        features = read_features(datasets, window)
        count = len(features)

        window_info.append((window, count))
        total_valid += count

if total_valid == 0:
    raise RuntimeError("No valid feature pixels were found.")

print(f"Total valid pixels: {total_valid:,}")


# ============================================================
# 6. SECOND PASS: DRAW A SPATIALLY REPRESENTATIVE SAMPLE
# ============================================================

target_sample = min(MAX_TRAINING_SAMPLES, total_valid)

# Allocate the sample approximately in proportion to each
# window's valid pixel count.
quotas = [
    int(target_sample * count / total_valid)
    for _, count in window_info
]

# Distribute any remaining sample slots.
remaining = target_sample - sum(quotas)

fractions = [
    (target_sample * count / total_valid) - quota
    for (_, count), quota in zip(window_info, quotas)
]

order = np.argsort(fractions)[::-1]

for idx in order[:remaining]:
    quotas[idx] += 1

rng = np.random.default_rng(RANDOM_STATE)
sample_parts = []

print(f"\nCollecting up to {target_sample:,} sample pixels...")

with ExitStack() as stack:
    datasets = [
        stack.enter_context(rasterio.open(path))
        for path in feature_paths
    ]

    for (window, count), quota in zip(window_info, quotas):
        if quota == 0 or count == 0:
            continue

        features = read_features(datasets, window)

        if len(features) > quota:
            chosen = rng.choice(
                len(features),
                size=quota,
                replace=False,
            )
            features = features[chosen]

        sample_parts.append(features)

X_raw = np.concatenate(sample_parts, axis=0)

if len(X_raw) < max(K_VALUES):
    raise RuntimeError(
        f"Only {len(X_raw)} sample pixels available; "
        "not enough for the requested K values."
    )

print(f"Sample collected: {len(X_raw):,} pixels")


# ============================================================
# 7. STANDARDIZE FEATURES
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Remove any unexpected numerical issues.
finite_rows = np.all(np.isfinite(X), axis=1)
X = X[finite_rows]

if len(X) < max(K_VALUES):
    raise RuntimeError("Too few finite standardized samples.")

print("Standardization complete.")


# ============================================================
# 8. TEST DIFFERENT K VALUES
# ============================================================

results = []

silhouette_n = min(SILHOUETTE_SAMPLE_SIZE, len(X))
silhouette_indices = rng.choice(
    len(X),
    size=silhouette_n,
    replace=False,
)

X_silhouette = X[silhouette_indices]

print("\n" + "=" * 70)
print("TESTING CLUSTER COUNTS")
print("=" * 70)

for k in K_VALUES:
    print(f"\nRunning K = {k}...")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )

    labels = model.fit_predict(X)

    # Silhouette is calculated on a smaller sample for speed.
    silhouette_labels = labels[silhouette_indices]

    silhouette = silhouette_score(
        X_silhouette,
        silhouette_labels,
        metric="euclidean",
    )

    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    result = {
        "K": k,
        "Silhouette": float(silhouette),
        "Davies_Bouldin": float(davies_bouldin),
        "Calinski_Harabasz": float(calinski_harabasz),
        "Inertia": float(model.inertia_),
    }

    results.append(result)

    print(f"  Silhouette Score      : {silhouette:.6f}")
    print(f"  Davies-Bouldin Index  : {davies_bouldin:.6f}")
    print(f"  Calinski-Harabasz     : {calinski_harabasz:.2f}")
    print(f"  Inertia               : {model.inertia_:.2f}")


# ============================================================
# 9. SAVE CSV
# ============================================================

fieldnames = [
    "K",
    "Silhouette",
    "Davies_Bouldin",
    "Calinski_Harabasz",
    "Inertia",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


# ============================================================
# 10. REPORT RESULTS
# ============================================================

best_silhouette = max(results, key=lambda item: item["Silhouette"])
best_davies = min(results, key=lambda item: item["Davies_Bouldin"])
best_calinski = max(
    results, key=lambda item: item["Calinski_Harabasz"]
)

print("\n" + "=" * 70)
print("OPTIMIZATION SUMMARY")
print("=" * 70)

print(
    f"Highest Silhouette: K={best_silhouette['K']} "
    f"({best_silhouette['Silhouette']:.6f})"
)
print(
    f"Lowest Davies-Bouldin: K={best_davies['K']} "
    f"({best_davies['Davies_Bouldin']:.6f})"
)
print(
    f"Highest Calinski-Harabasz: K={best_calinski['K']} "
    f"({best_calinski['Calinski_Harabasz']:.2f})"
)

print("\nComplete comparison:")
print(
    f"{'K':>3} | {'Silhouette':>12} | {'Davies-Bouldin':>15} | "
    f"{'Calinski-Harabasz':>18} | {'Inertia':>14}"
)
print("-" * 75)

for result in results:
    print(
        f"{result['K']:>3} | "
        f"{result['Silhouette']:>12.6f} | "
        f"{result['Davies_Bouldin']:>15.6f} | "
        f"{result['Calinski_Harabasz']:>18.2f} | "
        f"{result['Inertia']:>14.2f}"
    )

print("\nOptimization complete.")
print(f"Results saved to: {CSV_PATH}")
print(
    "\nThese metrics are internal clustering diagnostics. "
    "They do not establish land-cover accuracy."
)
print(
    "Do not change the existing classification raster until "
    "the metrics and reference data have been reviewed."
)
