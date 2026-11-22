
from pathlib import Path

import numpy as np
import rasterio

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "NISAR_output"

HH_INPUT = INPUT_DIR / "wayanad_HH_clean.tif"
HV_INPUT = INPUT_DIR / "wayanad_HV_clean.tif"

HH_DB_OUTPUT = INPUT_DIR / "wayanad_HH_dB.tif"
HV_DB_OUTPUT = INPUT_DIR / "wayanad_HV_dB.tif"
RATIO_OUTPUT = INPUT_DIR / "wayanad_HH_HV_ratio.tif"
NORM_DIFF_OUTPUT = INPUT_DIR / "wayanad_normalized_difference.tif"


def save_raster(path, data, profile):
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
    )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32), 1)


print("=" * 70)
print("WAYANAD NISAR FEATURE GENERATION")
print("=" * 70)

with rasterio.open(HH_INPUT) as hh_src, rasterio.open(HV_INPUT) as hv_src:
    if (
        hh_src.shape != hv_src.shape
        or hh_src.crs != hv_src.crs
        or hh_src.transform != hv_src.transform
    ):
        raise ValueError(
            "HH and HV rasters do not share the same grid."
        )

    hh = hh_src.read(1).astype(np.float32)
    hv = hv_src.read(1).astype(np.float32)
    profile = hh_src.profile.copy()

print("Input shape:", hh.shape)

# Valid pixels: both measurements finite and positive.
valid = (
    np.isfinite(hh)
    & np.isfinite(hv)
    & (hh > 0)
    & (hv > 0)
)

if not np.any(valid):
    raise ValueError("No valid positive HH/HV pixels found.")

print("Valid pixels:", int(np.sum(valid)))

# 1. Convert linear power to dB.
hh_db = np.full(hh.shape, np.nan, dtype=np.float32)
hv_db = np.full(hv.shape, np.nan, dtype=np.float32)

hh_db[valid] = 10.0 * np.log10(hh[valid])
hv_db[valid] = 10.0 * np.log10(hv[valid])

# 2. HH/HV ratio.
ratio = np.full(hh.shape, np.nan, dtype=np.float32)
ratio[valid] = hh[valid] / hv[valid]

# 3. Normalized difference.
normalized_difference = np.full(
    hh.shape, np.nan, dtype=np.float32
)

denominator = hh + hv
nd_valid = valid & np.isfinite(denominator) & (denominator > 0)

normalized_difference[nd_valid] = (
    (hh[nd_valid] - hv[nd_valid])
    / denominator[nd_valid]
)

# Save features.
save_raster(HH_DB_OUTPUT, hh_db, profile)
save_raster(HV_DB_OUTPUT, hv_db, profile)
save_raster(RATIO_OUTPUT, ratio, profile)
save_raster(NORM_DIFF_OUTPUT, normalized_difference, profile)

# Print statistics using finite values only.
for name, data in [
    ("HH dB", hh_db),
    ("HV dB", hv_db),
    ("HH/HV Ratio", ratio),
    ("Normalized Difference", normalized_difference),
]:
    values = data[np.isfinite(data)]
    print(f"\n{name}")
    print("Valid pixels:", len(values))
    print("Min :", float(np.min(values)))
    print("Max :", float(np.max(values)))
    print("Mean:", float(np.mean(values)))

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print("\nCreated:")
print(HH_DB_OUTPUT)
print(HV_DB_OUTPUT)
print(RATIO_OUTPUT)
print(NORM_DIFF_OUTPUT)
