from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "NISAR_output"
RESULTS_DIR = PROJECT_ROOT / "NISAR_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HH_FILE = INPUT_DIR / "wayanad_HH_clean.tif"
HV_FILE = INPUT_DIR / "wayanad_HV_clean.tif"


# ============================================================
# READ DATA
# ============================================================

print("=" * 70)
print("WAYANAD NISAR VISUALIZATION")
print("=" * 70)

with rasterio.open(HH_FILE) as src:
    hh = src.read(1)
    transform = src.transform
    bounds = src.bounds

with rasterio.open(HV_FILE) as src:
    hv = src.read(1)


# ============================================================
# VALID DATA
# ============================================================

valid = (
    np.isfinite(hh) &
    np.isfinite(hv) &
    (hh > 0) &
    (hv > 0)
)

print("\nValid pixels:", np.sum(valid))

# ------------------------------------------------------------
# Use percentile limits to avoid extreme-value domination
# ------------------------------------------------------------

hh_valid = hh[valid]
hv_valid = hv[valid]

hh_min, hh_max = np.percentile(hh_valid, [2, 98])
hv_min, hv_max = np.percentile(hv_valid, [2, 98])

print("\nHH 2-98 percentile:", hh_min, hh_max)
print("HV 2-98 percentile:", hv_min, hv_max)


# ============================================================
# HH MAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    hh,
    cmap="gray",
    vmin=hh_min,
    vmax=hh_max
)

plt.colorbar(label="HH GCOV")
plt.title("NISAR Wayanad - HH Backscatter")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.tight_layout()

hh_png = RESULTS_DIR / "wayanad_HH_map.png"

plt.savefig(hh_png, dpi=300)
plt.close()


# ============================================================
# HV MAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    hv,
    cmap="gray",
    vmin=hv_min,
    vmax=hv_max
)

plt.colorbar(label="HV GCOV")
plt.title("NISAR Wayanad - HV Backscatter")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.tight_layout()

hv_png = RESULTS_DIR / "wayanad_HV_map.png"

plt.savefig(hv_png, dpi=300)
plt.close()


# ============================================================
# HH / HV RATIO
# ============================================================

ratio = np.full(hh.shape, np.nan, dtype=np.float32)

ratio[valid] = hh[valid] / hv[valid]

ratio_valid = ratio[np.isfinite(ratio)]

ratio_min, ratio_max = np.percentile(
    ratio_valid,
    [2, 98]
)

print("\nHH/HV ratio:")
print("Min:", np.min(ratio_valid))
print("Max:", np.max(ratio_valid))
print("Mean:", np.mean(ratio_valid))
print("2-98 percentile:", ratio_min, ratio_max)


# ============================================================
# RATIO MAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    ratio,
    cmap="gray",
    vmin=ratio_min,
    vmax=ratio_max
)

plt.colorbar(label="HH / HV")
plt.title("NISAR Wayanad - HH/HV Ratio")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.tight_layout()

ratio_png = RESULTS_DIR / "wayanad_HH_HV_ratio.png"

plt.savefig(ratio_png, dpi=300)
plt.close()


# ============================================================
# HISTOGRAMS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    hh_valid,
    bins=200
)

plt.xlabel("HH GCOV")
plt.ylabel("Pixel Count")
plt.title("Wayanad HH Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "wayanad_HH_histogram.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10, 6))

plt.hist(
    hv_valid,
    bins=200
)

plt.xlabel("HV GCOV")
plt.ylabel("Pixel Count")
plt.title("Wayanad HV Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "wayanad_HV_histogram.png",
    dpi=300
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nResults saved in:")
print(RESULTS_DIR)

print("\nGenerated files:")
print("  wayanad_HH_map.png")
print("  wayanad_HV_map.png")
print("  wayanad_HH_HV_ratio.png")
print("  wayanad_HH_histogram.png")
print("  wayanad_HV_histogram.png")