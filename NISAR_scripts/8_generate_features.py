import os
import numpy as np
import rasterio

BASE_DIR = r"D:\FinalYearPrj\prj"
INPUT_DIR = os.path.join(BASE_DIR, "NISAR_output")

HH_INPUT = os.path.join(INPUT_DIR, "wayanad_HH_clean.tif")
HV_INPUT = os.path.join(INPUT_DIR, "wayanad_HV_clean.tif")

HH_DB_OUTPUT = os.path.join(INPUT_DIR, "wayanad_HH_dB.tif")
HV_DB_OUTPUT = os.path.join(INPUT_DIR, "wayanad_HV_dB.tif")
RATIO_OUTPUT = os.path.join(INPUT_DIR, "wayanad_HH_HV_ratio.tif")
NORM_DIFF_OUTPUT = os.path.join(INPUT_DIR, "wayanad_normalized_difference.tif")


def save_raster(path, data, profile):
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate"
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


print("=" * 70)
print("WAYANAD NISAR FEATURE GENERATION")
print("=" * 70)

with rasterio.open(HH_INPUT) as hh_src, rasterio.open(HV_INPUT) as hv_src:

    hh = hh_src.read(1).astype(np.float32)
    hv = hv_src.read(1).astype(np.float32)

    profile = hh_src.profile

print("Input shape:", hh.shape)

# ---------------------------------------------------------
# VALID DATA
# ---------------------------------------------------------

valid = (
    np.isfinite(hh) &
    np.isfinite(hv) &
    (hh > 0) &
    (hv > 0)
)

print("Valid pixels:", np.sum(valid))

# ---------------------------------------------------------
# 1. HH / HV IN dB
# ---------------------------------------------------------

hh_db = np.full(hh.shape, np.nan, dtype=np.float32)
hv_db = np.full(hv.shape, np.nan, dtype=np.float32)

hh_db[valid] = 10.0 * np.log10(hh[valid])
hv_db[valid] = 10.0 * np.log10(hv[valid])

# ---------------------------------------------------------
# 2. HH / HV RATIO
# ---------------------------------------------------------

ratio = np.full(hh.shape, np.nan, dtype=np.float32)

ratio[valid] = hh[valid] / hv[valid]

# Remove unrealistic extreme ratios
ratio[ratio > 100] = np.nan

# ---------------------------------------------------------
# 3. NORMALIZED DIFFERENCE
# ---------------------------------------------------------

normalized_difference = np.full(
    hh.shape,
    np.nan,
    dtype=np.float32
)

denominator = hh + hv

nd_valid = valid & (denominator > 0)

normalized_difference[nd_valid] = (
    (hh[nd_valid] - hv[nd_valid]) /
    denominator[nd_valid]
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

save_raster(HH_DB_OUTPUT, hh_db, profile)
save_raster(HV_DB_OUTPUT, hv_db, profile)
save_raster(RATIO_OUTPUT, ratio, profile)
save_raster(NORM_DIFF_OUTPUT, normalized_difference, profile)

# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

print("\nHH dB")
print("Min :", np.nanmin(hh_db))
print("Max :", np.nanmax(hh_db))
print("Mean:", np.nanmean(hh_db))

print("\nHV dB")
print("Min :", np.nanmin(hv_db))
print("Max :", np.nanmax(hv_db))
print("Mean:", np.nanmean(hv_db))

print("\nHH/HV Ratio")
print("Min :", np.nanmin(ratio))
print("Max :", np.nanmax(ratio))
print("Mean:", np.nanmean(ratio))

print("\nNormalized Difference")
print("Min :", np.nanmin(normalized_difference))
print("Max :", np.nanmax(normalized_difference))
print("Mean:", np.nanmean(normalized_difference))

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print("\nCreated:")
print(HH_DB_OUTPUT)
print(HV_DB_OUTPUT)
print(RATIO_OUTPUT)
print(NORM_DIFF_OUTPUT)