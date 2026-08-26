from pathlib import Path
import rasterio
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "NISAR_output"
OUTPUT_DIR = PROJECT_ROOT / "NISAR_output"

HH_INPUT = INPUT_DIR / "wayanad_HH.tif"
HV_INPUT = INPUT_DIR / "wayanad_HV.tif"
MASK_INPUT = INPUT_DIR / "wayanad_mask.tif"

HH_OUTPUT = OUTPUT_DIR / "wayanad_HH_clean.tif"
HV_OUTPUT = OUTPUT_DIR / "wayanad_HV_clean.tif"


# ============================================================
# READ DATA
# ============================================================

print("=" * 70)
print("WAYANAD PREPROCESSING")
print("=" * 70)

with rasterio.open(HH_INPUT) as src:
    hh = src.read(1)
    profile = src.profile.copy()

with rasterio.open(HV_INPUT) as src:
    hv = src.read(1)

with rasterio.open(MASK_INPUT) as src:
    mask = src.read(1)


# ============================================================
# VALID DATA MASK
# ============================================================

# NISAR mask:
# 0   = invalid / partially focused
# 1+  = valid radar data
# 255 = outside acquisition extent

valid_mask = (
    (mask != 255) &
    (mask != 0) &
    np.isfinite(hh) &
    np.isfinite(hv)
)

print("\nInput dimensions:")
print("Rows:", hh.shape[0])
print("Cols:", hh.shape[1])

print("\nValid pixels:", np.sum(valid_mask))
print("Invalid pixels:", np.sum(~valid_mask))


# ============================================================
# REMOVE INVALID DATA
# ============================================================

hh_clean = hh.astype(np.float32)
hv_clean = hv.astype(np.float32)

hh_clean[~valid_mask] = np.nan
hv_clean[~valid_mask] = np.nan


# ============================================================
# STATISTICS
# ============================================================

hh_valid = hh_clean[np.isfinite(hh_clean)]
hv_valid = hv_clean[np.isfinite(hv_clean)]

print("\nHH statistics:")
print("Min :", np.min(hh_valid))
print("Max :", np.max(hh_valid))
print("Mean:", np.mean(hh_valid))
print("Std :", np.std(hh_valid))

print("\nHV statistics:")
print("Min :", np.min(hv_valid))
print("Max :", np.max(hv_valid))
print("Mean:", np.mean(hv_valid))
print("Std :", np.std(hv_valid))


# ============================================================
# WRITE CLEAN HH
# ============================================================

profile.update(
    dtype="float32",
    count=1,
    nodata=np.nan,
    compress="deflate"
)

with rasterio.open(HH_OUTPUT, "w", **profile) as dst:
    dst.write(hh_clean, 1)


# ============================================================
# WRITE CLEAN HV
# ============================================================

with rasterio.open(HV_OUTPUT, "w", **profile) as dst:
    dst.write(hv_clean, 1)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nCreated:")
print(HH_OUTPUT)
print(HV_OUTPUT)