import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

INPUT = "D:\\FinalYearPrj\\prj\\NISAR_original\\NISAR_L2_PR_GCOV_028_077_D_083_4005_DHDH_A_20260818T131953_20260818T132027_P05023_N_F_J_001.h5"

from pathlib import Path

# Project root = one level above NISAR_scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = next(
    (PROJECT_ROOT / "NISAR_original").glob("*.h5")
)

OUTPUT_DIR = PROJECT_ROOT / "NISAR_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HH_OUTPUT = OUTPUT_DIR / "wayanad_HH.tif"
HV_OUTPUT = OUTPUT_DIR / "wayanad_HV.tif"
MASK_OUTPUT = OUTPUT_DIR / "wayanad_mask.tif"

print("Input:", INPUT)
print("Output:", OUTPUT_DIR)

# Wayanad pixel window obtained previously
ROW_MIN = 2772
ROW_MAX = 10553

COL_MIN = 4325
COL_MAX = 10100

# NISAR georeferencing
X_FIRST = 573125.0
Y_FIRST = 1365835.0

PIXEL_SIZE = 10.0

with h5py.File(INPUT, "r") as f:

    HH = f["science/LSAR/GCOV/grids/frequencyA/HHHH"][
        ROW_MIN:ROW_MAX,
        COL_MIN:COL_MAX
    ]

    HV = f["science/LSAR/GCOV/grids/frequencyA/HVHV"][
        ROW_MIN:ROW_MAX,
        COL_MIN:COL_MAX
    ]

    MASK = f["science/LSAR/GCOV/grids/frequencyA/mask"][
        ROW_MIN:ROW_MAX,
        COL_MIN:COL_MAX
    ]

# Replace invalid values
HH = np.asarray(HH, dtype=np.float32)
HV = np.asarray(HV, dtype=np.float32)
MASK = np.asarray(MASK, dtype=np.uint8)

HH[MASK == 255] = np.nan
HV[MASK == 255] = np.nan

# Calculate geographic origin of extracted window
x_origin = X_FIRST + COL_MIN * PIXEL_SIZE
y_origin = Y_FIRST - ROW_MIN * PIXEL_SIZE

transform = from_origin(
    x_origin,
    y_origin,
    PIXEL_SIZE,
    PIXEL_SIZE
)

height, width = HH.shape

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:32643",
    "transform": transform,
    "nodata": np.nan,
    "compress": "deflate"
}

with rasterio.open(HH_OUTPUT, "w", **profile) as dst:
    dst.write(HH, 1)

with rasterio.open(HV_OUTPUT, "w", **profile) as dst:
    dst.write(HV, 1)

mask_profile = profile.copy()
mask_profile["dtype"] = "uint8"
mask_profile["nodata"] = 255

with rasterio.open(MASK_OUTPUT, "w", **mask_profile) as dst:
    dst.write(MASK, 1)

print("=" * 70)
print("WAYANAD EXTRACTION COMPLETE")
print("=" * 70)

print("Shape:", HH.shape)
print("HH:", HH_OUTPUT)
print("HV:", HV_OUTPUT)
print("MASK:", MASK_OUTPUT)