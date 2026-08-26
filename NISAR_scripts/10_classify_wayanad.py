import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# WAYANAD NISAR - K-MEANS CLASSIFICATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "NISAR_output")
RESULT_DIR = os.path.join(BASE_DIR, "NISAR_results")

os.makedirs(RESULT_DIR, exist_ok=True)

# Input feature files
FEATURE_FILES = {
    "HH_dB": os.path.join(INPUT_DIR, "wayanad_HH_dB.tif"),
    "HV_dB": os.path.join(INPUT_DIR, "wayanad_HV_dB.tif"),
    "HH_HV_ratio": os.path.join(INPUT_DIR, "wayanad_HH_HV_ratio.tif"),
    "Normalized_Difference": os.path.join(
        INPUT_DIR, "wayanad_normalized_difference.tif"
    ),
}

OUTPUT_TIF = os.path.join(
    INPUT_DIR, "wayanad_kmeans_classes.tif"
)

OUTPUT_PNG = os.path.join(
    RESULT_DIR, "wayanad_kmeans_map.png"
)


print("=" * 70)
print("WAYANAD NISAR K-MEANS CLASSIFICATION")
print("=" * 70)


# ============================================================
# 1. READ FEATURES
# ============================================================

arrays = []
profile = None

for name, path in FEATURE_FILES.items():

    print(f"Loading: {name}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    with rasterio.open(path) as src:

        data = src.read(1).astype(np.float32)

        if profile is None:
            profile = src.profile.copy()

        arrays.append(data)


hh_db = arrays[0]
hv_db = arrays[1]
ratio = arrays[2]
normalized_difference = arrays[3]


# ============================================================
# 2. CREATE VALID PIXEL MASK
# ============================================================

valid_mask = (
    np.isfinite(hh_db)
    & np.isfinite(hv_db)
    & np.isfinite(ratio)
    & np.isfinite(normalized_difference)
)

print()
print("Input shape:", hh_db.shape)
print("Valid pixels:", np.sum(valid_mask))
print("Invalid pixels:", np.sum(~valid_mask))


# ============================================================
# 3. CREATE FEATURE MATRIX
# ============================================================

features = np.column_stack([
    hh_db[valid_mask],
    hv_db[valid_mask],
    ratio[valid_mask],
    normalized_difference[valid_mask]
])

print("Feature matrix shape:", features.shape)


# ============================================================
# 4. REMOVE EXTREME VALUES
# ============================================================
# Ratio can contain unusually large values.
# Limit extreme values using percentile clipping.

ratio_low = np.percentile(features[:, 2], 1)
ratio_high = np.percentile(features[:, 2], 99)

features[:, 2] = np.clip(
    features[:, 2],
    ratio_low,
    ratio_high
)

print()
print("Ratio clipping:")
print("1st percentile :", ratio_low)
print("99th percentile:", ratio_high)


# ============================================================
# 5. STANDARDIZE FEATURES
# ============================================================

print()
print("Standardizing features...")

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)


# ============================================================
# 6. K-MEANS CLASSIFICATION
# ============================================================

# Number of classes
N_CLASSES = 5

print()
print("Running K-Means...")
print("Number of classes:", N_CLASSES)

kmeans = KMeans(
    n_clusters=N_CLASSES,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(features_scaled)

# Make class numbers start from 1
labels = labels + 1


# ============================================================
# 7. CREATE CLASSIFICATION RASTER
# ============================================================

classification = np.zeros(
    hh_db.shape,
    dtype=np.uint8
)

classification[valid_mask] = labels


# ============================================================
# 8. SAVE CLASSIFICATION RASTER
# ============================================================

profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="lzw"
)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    **profile
) as dst:

    dst.write(classification, 1)


print()
print("Classification raster saved:")
print(OUTPUT_TIF)


# ============================================================
# 9. PRINT CLASS STATISTICS
# ============================================================

print()
print("=" * 70)
print("CLASS STATISTICS")
print("=" * 70)

for class_id in range(1, N_CLASSES + 1):

    mask = labels == class_id

    count = np.sum(mask)
    percentage = (count / len(labels)) * 100

    class_features = features[mask]

    print()
    print(f"Class {class_id}")
    print("-" * 30)
    print("Pixels     :", count)
    print("Percentage :", f"{percentage:.2f}%")
    print("HH dB mean :", np.mean(class_features[:, 0]))
    print("HV dB mean :", np.mean(class_features[:, 1]))
    print("Ratio mean :", np.mean(class_features[:, 2]))
    print(
        "ND mean    :",
        np.mean(class_features[:, 3])
    )


# ============================================================
# 10. VISUALIZE CLASSIFICATION
# ============================================================

print()
print("Creating classification map...")

plt.figure(figsize=(10, 12))

masked_classification = np.ma.masked_where(
    classification == 0,
    classification
)

plt.imshow(
    masked_classification,
    cmap="tab10",
    interpolation="nearest"
)

plt.title(
    "NISAR Wayanad - K-Means Unsupervised Classification"
)

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

cbar = plt.colorbar()
cbar.set_label("K-Means Class")

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("K-MEANS CLASSIFICATION COMPLETE")
print("=" * 70)

print()
print("Created:")
print(OUTPUT_TIF)
print(OUTPUT_PNG)

print()
print("Number of classes:", N_CLASSES)
print("Valid pixels:", np.sum(valid_mask))

print("=" * 70)