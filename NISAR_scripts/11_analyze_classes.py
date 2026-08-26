import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# WAYANAD NISAR - CLASS ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_DIR = os.path.join(BASE_DIR, "NISAR_output")
RESULT_DIR = os.path.join(BASE_DIR, "NISAR_results")

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# FILES
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

OUTPUT_MAP = os.path.join(
    RESULT_DIR,
    "wayanad_class_analysis.png"
)

OUTPUT_CSV = os.path.join(
    RESULT_DIR,
    "wayanad_class_statistics.csv"
)


print("=" * 70)
print("WAYANAD NISAR CLASS ANALYSIS")
print("=" * 70)


# ============================================================
# 1. READ DATA
# ============================================================

print()
print("Loading classification...")

with rasterio.open(CLASS_FILE) as src:
    classes = src.read(1)
    profile = src.profile.copy()
    transform = src.transform


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
# 2. VALID MASK
# ============================================================

valid = (
    (classes > 0)
    & np.isfinite(hh)
    & np.isfinite(hv)
    & np.isfinite(ratio)
    & np.isfinite(nd)
)


valid_classes = classes[valid]

print()
print("Image shape:", classes.shape)
print("Valid pixels:", np.sum(valid))


# ============================================================
# 3. CLASS STATISTICS
# ============================================================

print()
print("=" * 70)
print("CLASS STATISTICS")
print("=" * 70)

results = []

pixel_area_m2 = abs(transform.a * transform.e)
pixel_area_ha = pixel_area_m2 / 10000.0


for class_id in range(1, 6):

    mask = valid & (classes == class_id)

    count = np.sum(mask)

    if count == 0:
        continue

    class_hh = hh[mask]
    class_hv = hv[mask]
    class_ratio = ratio[mask]
    class_nd = nd[mask]

    percentage = (
        count / np.sum(valid)
    ) * 100

    area_m2 = count * pixel_area_m2
    area_ha = count * pixel_area_ha

    print()
    print(f"CLASS {class_id}")
    print("-" * 40)

    print("Pixels          :", count)
    print("Percentage      :", f"{percentage:.2f}%")
    print("Area (m²)       :", f"{area_m2:.2f}")
    print("Area (hectares) :", f"{area_ha:.2f}")

    print()
    print("HH dB")
    print("  Mean :", f"{np.mean(class_hh):.4f}")
    print("  Std  :", f"{np.std(class_hh):.4f}")

    print()
    print("HV dB")
    print("  Mean :", f"{np.mean(class_hv):.4f}")
    print("  Std  :", f"{np.std(class_hv):.4f}")

    print()
    print("HH/HV Ratio")
    print("  Mean :", f"{np.mean(class_ratio):.4f}")
    print("  Std  :", f"{np.std(class_ratio):.4f}")

    print()
    print("Normalized Difference")
    print("  Mean :", f"{np.mean(class_nd):.4f}")
    print("  Std  :", f"{np.std(class_nd):.4f}")

    results.append([
        class_id,
        count,
        percentage,
        area_m2,
        area_ha,
        np.mean(class_hh),
        np.mean(class_hv),
        np.mean(class_ratio),
        np.mean(class_nd)
    ])


# ============================================================
# 4. SAVE CSV
# ============================================================

print()
print("Saving class statistics...")

header = (
    "Class,"
    "Pixels,"
    "Percentage,"
    "Area_m2,"
    "Area_hectares,"
    "Mean_HH_dB,"
    "Mean_HV_dB,"
    "Mean_HH_HV_Ratio,"
    "Mean_Normalized_Difference"
)

np.savetxt(
    OUTPUT_CSV,
    np.array(results),
    delimiter=",",
    header=header,
    comments="",
    fmt="%.6f"
)

print("Saved:")
print(OUTPUT_CSV)


# ============================================================
# 5. CREATE CLASS MAP
# ============================================================

print()
print("Creating class analysis map...")


masked_classes = np.ma.masked_where(
    classes == 0,
    classes
)


plt.figure(figsize=(10, 12))

plt.imshow(
    masked_classes,
    cmap="tab10",
    vmin=1,
    vmax=5,
    interpolation="nearest"
)

plt.title(
    "Wayanad NISAR - K-Means Class Analysis"
)

plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

cbar = plt.colorbar(
    ticks=[1, 2, 3, 4, 5]
)

cbar.set_label("K-Means Class")

plt.tight_layout()

plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. PRINT INTERPRETATION GUIDE
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)

print("""
Use the following characteristics when interpreting the classes:

Higher Normalized Difference
    -> stronger HH dominance
    -> often associated with vegetation/structured surfaces

Lower Normalized Difference
    -> HH and HV responses are more similar
    -> may indicate water, built-up areas, bare surfaces,
       or other low-vegetation surfaces

Higher HH/HV ratio
    -> stronger co-polarized response relative to cross-polarized
    -> useful for distinguishing surface scattering behaviour

Lower HH/HV ratio
    -> relatively stronger HV response
    -> may indicate different scattering mechanisms

IMPORTANT:
K-Means class numbers are arbitrary.
Class 1 is NOT automatically "Class 1 land cover".

The class labels should be assigned only after examining:
1. Spatial distribution
2. HH dB
3. HV dB
4. HH/HV ratio
5. Normalized Difference
6. Known geographic features
""")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("CLASS ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Created:")
print(OUTPUT_CSV)
print(OUTPUT_MAP)

print()
print("Next step:")
print("Interpret the five K-Means classes.")
print("=" * 70)