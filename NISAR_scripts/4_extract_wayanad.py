
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT_ROOT / "NISAR_original"
    / "NISAR_L2_PR_GCOV_028_005_D_082_4005_DHDH_A_20260813T132738_20260813T132813_P05023_N_F_J_001.h5"
)

OUTPUT_DIR = PROJECT_ROOT / "NISAR_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HH_OUTPUT = OUTPUT_DIR / "wayanad_HH.tif"
HV_OUTPUT = OUTPUT_DIR / "wayanad_HV.tif"
MASK_OUTPUT = OUTPUT_DIR / "wayanad_mask.tif"

BASE = "science/LSAR/GCOV/grids/frequencyA"

# Approximate Wayanad-area bounds in EPSG:32643.
# These are a provisional rectangle, NOT the district boundary.
X_MIN, X_MAX = 584125.76, 656496.39
Y_MIN, Y_MAX = 1265174.46, 1323548.90

with h5py.File(INPUT, "r") as f:
    grid = f[BASE]

    x = grid["xCoordinates"][:]
    y = grid["yCoordinates"][:]

    pixel_size = float(grid["xCoordinateSpacing"][()])

    # Find coordinate indices within the requested bounds.
    cols = np.where((x >= X_MIN) & (x <= X_MAX))[0]
    rows = np.where((y >= Y_MIN) & (y <= Y_MAX))[0]

    if len(cols) == 0 or len(rows) == 0:
        raise ValueError(
            "Requested bounds do not overlap the scene coordinates. "
            "Check X/Y bounds and the scene coverage."
        )

    row_min, row_max = int(rows.min()), int(rows.max()) + 1
    col_min, col_max = int(cols.min()), int(cols.max()) + 1

    print("Input:", INPUT.name)
    print("CRS: EPSG:32643")
    print("Crop rows:", row_min, "to", row_max - 1)
    print("Crop cols:", col_min, "to", col_max - 1)
    print("Crop size:", row_max - row_min, "x", col_max - col_min)
    print(
        "Crop coordinate extent:",
        float(x[col_min]), float(x[col_max - 1]),
        float(y[row_max - 1]), float(y[row_min])
    )

    HH = grid["HHHH"][row_min:row_max, col_min:col_max]
    HV = grid["HVHV"][row_min:row_max, col_min:col_max]
    MASK = grid["mask"][row_min:row_max, col_min:col_max]

    HH = np.asarray(HH, dtype=np.float32)
    HV = np.asarray(HV, dtype=np.float32)
    MASK = np.asarray(MASK, dtype=np.uint8)


# Mask outside-acquisition and invalid/partially focused pixels.
invalid = (MASK == 255) | (MASK == 0)
HH[invalid] = np.nan
HV[invalid] = np.nan

# pixel_size was already read inside the HDF5 file block.
# The product metadata reports x spacing +10 m and y spacing -10 m.
transform = from_origin(
    float(x[col_min]) - pixel_size / 2,
    float(y[row_min]) + pixel_size / 2,
    pixel_size,
    pixel_size
)

profile = {
    "driver": "GTiff",
    "height": HH.shape[0],
    "width": HH.shape[1],
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:32643",
    "transform": transform,
    "nodata": np.nan,
    "compress": "deflate",
    "tiled": True
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

print("\nExtraction complete.")
print("HH:", HH_OUTPUT)
print("HV:", HV_OUTPUT)
print("Mask:", MASK_OUTPUT)
print("Valid HH pixels:", int(np.isfinite(HH).sum()))
print("Valid HV pixels:", int(np.isfinite(HV).sum()))
