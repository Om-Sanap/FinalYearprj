import h5py
import numpy as np

FILE = r"D:\FinalYearPrj\prj\NISAR_original\NISAR_L2_PR_GCOV_028_005_D_082_4005_DHDH_A_20260813T132738_20260813T132813_P05023_N_F_J_001.h5"

BASE = "science/LSAR/GCOV"

with h5py.File(FILE, "r") as f:

    print("=" * 80)
    print("NISAR GEOSPATIAL INSPECTION")
    print("=" * 80)

    # ---------------------------------------------------------
    # IDENTIFICATION
    # ---------------------------------------------------------

    print("\nIDENTIFICATION")
    print("-" * 80)

    paths = [
        "science/LSAR/identification/missionId",
        "science/LSAR/identification/platformName",
        "science/LSAR/identification/productLevel",
        "science/LSAR/identification/productType",
        "science/LSAR/identification/productVersion",
        "science/LSAR/identification/radarBand",
        "science/LSAR/identification/orbitPassDirection",
        "science/LSAR/identification/lookDirection",
        "science/LSAR/identification/trackNumber",
        "science/LSAR/identification/frameNumber",
        "science/LSAR/identification/boundingPolygon",
    ]

    for path in paths:
        value = f[path][()]
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        print(f"{path}")
        print(f"  {value}")

    # ---------------------------------------------------------
    # FREQUENCY A
    # ---------------------------------------------------------

    print("\nFREQUENCY A")
    print("-" * 80)

    group = f[f"{BASE}/grids/frequencyA"]

    x = group["xCoordinates"][:]
    y = group["yCoordinates"][:]

    print("X coordinates:")
    print(f"  count  = {len(x)}")
    print(f"  first  = {x[0]}")
    print(f"  last   = {x[-1]}")
    print(f"  min    = {np.min(x)}")
    print(f"  max    = {np.max(x)}")

    print("\nY coordinates:")
    print(f"  count  = {len(y)}")
    print(f"  first  = {y[0]}")
    print(f"  last   = {y[-1]}")
    print(f"  min    = {np.min(y)}")
    print(f"  max    = {np.max(y)}")

    print("\nSpacing:")
    print("  X spacing =", group["xCoordinateSpacing"][()])
    print("  Y spacing =", group["yCoordinateSpacing"][()])

    print("\nProjection:")
    print(" ", group["projection"][()])

    # ---------------------------------------------------------
    # BOUNDING BOX
    # ---------------------------------------------------------

    print("\nBOUNDING BOX")
    print("-" * 80)

    bbox_path = f"{BASE}/metadata/ceosAnalysisReadyData/boundingBox"

    bbox = f[bbox_path][()]

    if isinstance(bbox, bytes):
        bbox = bbox.decode(errors="replace")

    print(bbox)

    # ---------------------------------------------------------
    # PRODUCT PROCESSING INFORMATION
    # ---------------------------------------------------------

    print("\nPROCESSING INFORMATION")
    print("-" * 80)

    processing = [
        "radiometricTerrainCorrectionApplied",
        "noiseCorrectionApplied",
        "rfiMitigationApplied",
        "polarimetricSymmetrizationApplied",
        "polarizationOrientationCorrectionApplied",
        "shadowMaskingApplied",
        "preprocessingMultilookingApplied",
    ]

    base = f"{BASE}/metadata/processingInformation/parameters"

    for name in processing:
        path = f"{base}/{name}"

        if path in f:
            value = f[path][()]
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            print(f"{name}: {value}")

    # ---------------------------------------------------------
    # SAMPLE DATA VALUES
    # ---------------------------------------------------------

    print("\nHH / HV DATA SAMPLE")
    print("-" * 80)

    for pol in ["HHHH", "HVHV"]:

        path = f"{BASE}/grids/frequencyA/{pol}"
        ds = f[path]

        print(f"\n{pol}")
        print("Shape:", ds.shape)
        print("dtype:", ds.dtype)

        # Small sample only
        rows = min(1000, ds.shape[0])
        cols = min(1000, ds.shape[1])

        sample = ds[:rows, :cols]

        finite = sample[np.isfinite(sample)]

        if len(finite) > 0:
            print("Sample minimum:", np.min(finite))
            print("Sample maximum:", np.max(finite))
            print("Sample mean:", np.mean(finite))
            print("Sample median:", np.median(finite))

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80) 