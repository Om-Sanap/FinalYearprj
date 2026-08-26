# import h5py
# import numpy as np
# from pyproj import Transformer

# FILE = r"D:\FinalYearPrj\prj\NISAR_original\NISAR_L2_PR_GCOV_028_077_D_083_4005_DHDH_A_20260818T131953_20260818T132027_P05023_N_F_J_001.h5"

# BASE = "science/LSAR/GCOV/grids/frequencyA"

# # ---------------------------------------------------------
# # WAYANAD TARGET REGION
# # ---------------------------------------------------------

# WEST = 75.70
# EAST = 76.60
# SOUTH = 11.40
# NORTH = 12.10

# # Geographic coordinates -> UTM Zone 43N
# transformer = Transformer.from_crs(
#     "EPSG:4326",
#     "EPSG:32643",
#     always_xy=True
# )

# xmin, ymin = transformer.transform(WEST, SOUTH)
# xmax, ymax = transformer.transform(EAST, NORTH)

# print("=" * 80)
# print("WAYANAD NISAR TARGET REGION")
# print("=" * 80)

# print("\nGeographic bounding box:")
# print(f"  West  = {WEST}")
# print(f"  East  = {EAST}")
# print(f"  South = {SOUTH}")
# print(f"  North = {NORTH}")

# print("\nUTM EPSG:32643 bounding box:")
# print(f"  xmin = {xmin:.2f}")
# print(f"  xmax = {xmax:.2f}")
# print(f"  ymin = {ymin:.2f}")
# print(f"  ymax = {ymax:.2f}")


# with h5py.File(FILE, "r") as f:

#     x = f[f"{BASE}/xCoordinates"][:]
#     y = f[f"{BASE}/yCoordinates"][:]

#     print("\nNISAR raster:")
#     print(f"  X: {x[0]} -> {x[-1]}")
#     print(f"  Y: {y[0]} -> {y[-1]}")

#     # -----------------------------------------------------
#     # FIND PIXEL INDICES
#     # -----------------------------------------------------

#     # X increases
#     x_indices = np.where(
#         (x >= xmin) &
#         (x <= xmax)
#     )[0]

#     # Y decreases
#     y_indices = np.where(
#         (y <= ymax) &
#         (y >= ymin)
#     )[0]

#     if len(x_indices) == 0:
#         raise RuntimeError("No X coordinates found for Wayanad.")

#     if len(y_indices) == 0:
#         raise RuntimeError("No Y coordinates found for Wayanad.")

#     col_start = x_indices[0]
#     col_end = x_indices[-1]

#     row_start = y_indices[0]
#     row_end = y_indices[-1]

#     print("\nWayanad pixel window:")
#     print(f"  rows = {row_start} -> {row_end}")
#     print(f"  cols = {col_start} -> {col_end}")

#     rows = row_end - row_start + 1
#     cols = col_end - col_start + 1

#     print("\nWayanad raster dimensions:")
#     print(f"  rows = {rows}")
#     print(f"  cols = {cols}")
#     print(f"  pixels = {rows * cols:,}")

#     # Float32 size
#     size_gb = rows * cols * 4 / (1024 ** 3)

#     print(f"\nApprox uncompressed float32 size:")
#     print(f"  {size_gb:.2f} GB")

#     # -----------------------------------------------------
#     # READ ONLY WAYANAD
#     # -----------------------------------------------------

#     hh = f[f"{BASE}/HHHH"]
#     hv = f[f"{BASE}/HVHV"]
#     mask = f[f"{BASE}/mask"]

#     print("\nReading Wayanad HH/HV sample...")

#     # Take a smaller sample from the Wayanad region
#     sample_size = 1000

#     r0 = row_start
#     c0 = col_start

#     r1 = min(r0 + sample_size, row_end + 1)
#     c1 = min(c0 + sample_size, col_end + 1)

#     hh_sample = hh[r0:r1, c0:c1]
#     hv_sample = hv[r0:r1, c0:c1]
#     mask_sample = mask[r0:r1, c0:c1]

#     print("\nSample window:")
#     print(f"  rows = {r0}:{r1}")
#     print(f"  cols = {c0}:{c1}")

#     print("\nMask values:")

#     values, counts = np.unique(
#         mask_sample,
#         return_counts=True
#     )

#     for value, count in zip(values, counts):
#         print(f"  {value}: {count:,}")

#     print("\nHHHH statistics:")

#     finite = hh_sample[np.isfinite(hh_sample)]

#     if len(finite) > 0:
#         print(f"  finite pixels = {len(finite):,}")
#         print(f"  min           = {np.min(finite)}")
#         print(f"  max           = {np.max(finite)}")
#         print(f"  mean          = {np.mean(finite)}")
#         print(f"  median        = {np.median(finite)}")
#     else:
#         print("  NO FINITE VALUES")

#     print("\nHVHV statistics:")

#     finite = hv_sample[np.isfinite(hv_sample)]

#     if len(finite) > 0:
#         print(f"  finite pixels = {len(finite):,}")
#         print(f"  min           = {np.min(finite)}")
#         print(f"  max           = {np.max(finite)}")
#         print(f"  mean          = {np.mean(finite)}")
#         print(f"  median        = {np.median(finite)}")
#     else:
#         print("  NO FINITE VALUES")

#     print("\n" + "=" * 80)
#     print("WAYANAD DIAGNOSTIC COMPLETE")
#     print("=" * 80) 


import h5py
import numpy as np

FILE = r"D:\FinalYearPrj\prj\NISAR_original\NISAR_L2_PR_GCOV_028_077_D_083_4005_DHDH_A_20260818T131953_20260818T132027_P05023_N_F_J_001.h5"

HH_PATH = "science/LSAR/GCOV/grids/frequencyA/HHHH"
HV_PATH = "science/LSAR/GCOV/grids/frequencyA/HVHV"
MASK_PATH = "science/LSAR/GCOV/grids/frequencyA/mask"

# Wayanad window from your previous calculation
R0, R1 = 2772, 10553
C0, C1 = 325, 10100

with h5py.File(FILE, "r") as f:

    hh = f[HH_PATH]
    hv = f[HV_PATH]
    mask = f[MASK_PATH]

    print("=" * 80)
    print("WAYANAD VALID-DATA SEARCH")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Check HDF5 fill-value attributes
    # ---------------------------------------------------------
    print("\nHDF5 ATTRIBUTES")
    print("-" * 80)

    for name, ds in [("HHHH", hh), ("HVHV", hv), ("MASK", mask)]:
        print(f"\n{name}")
        for key in ds.attrs.keys():
            try:
                value = ds.attrs[key]
                print(f"  {key}: {value}")
            except Exception:
                pass

    # ---------------------------------------------------------
    # 2. Divide Wayanad into coarse tiles
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCANNING WAYANAD FOR VALID DATA")
    print("=" * 80)

    tile_h = 500
    tile_w = 500

    valid_tiles = []

    for r in range(R0, R1, tile_h):

        rr1 = min(r + tile_h, R1)

        for c in range(C0, C1, tile_w):

            cc1 = min(c + tile_w, C1)

            hh_data = hh[r:rr1, c:cc1]
            hv_data = hv[r:rr1, c:cc1]
            mask_data = mask[r:rr1, c:cc1]

            hh_finite = np.isfinite(hh_data)
            hv_finite = np.isfinite(hv_data)

            hh_count = np.count_nonzero(hh_finite)
            hv_count = np.count_nonzero(hv_finite)

            total = hh_data.size

            if hh_count > 0 or hv_count > 0:

                valid_tiles.append(
                    (
                        r,
                        rr1,
                        c,
                        cc1,
                        hh_count,
                        hv_count,
                        total
                    )
                )

                print(
                    f"VALID TILE: "
                    f"rows {r}:{rr1}, "
                    f"cols {c}:{cc1}, "
                    f"HH={hh_count:,}, "
                    f"HV={hv_count:,}"
                )

    # ---------------------------------------------------------
    # 3. Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Total valid tiles: {len(valid_tiles)}")

    if not valid_tiles:
        print("\nNO FINITE HH/HV PIXELS FOUND ANYWHERE IN WAYANAD WINDOW.")
        print("\nThis means we need to investigate the mask/no-data encoding")
        print("and the actual geographic coverage before continuing.")

    else:

        print("\nFirst valid tiles:")

        for tile in valid_tiles[:20]:
            print(tile)

        # -----------------------------------------------------
        # Find overall extent of valid tiles
        # -----------------------------------------------------
        rows = []
        cols = []

        for r0, r1, c0, c1, _, _, _ in valid_tiles:
            rows.extend([r0, r1])
            cols.extend([c0, c1])

        print("\nApproximate valid raster extent:")
        print(f"  rows = {min(rows)} -> {max(rows)}")
        print(f"  cols = {min(cols)} -> {max(cols)}")

    print("\n" + "=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)