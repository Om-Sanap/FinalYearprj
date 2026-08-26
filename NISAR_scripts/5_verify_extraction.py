from pathlib import Path
import rasterio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "NISAR_output"

FILES = {
    "HH": OUTPUT_DIR / "wayanad_HH.tif",
    "HV": OUTPUT_DIR / "wayanad_HV.tif",
    "MASK": OUTPUT_DIR / "wayanad_mask.tif",
}

print("=" * 70)
print("WAYANAD EXTRACTION VERIFICATION")
print("=" * 70)

for name, filepath in FILES.items():

    print(f"\n--- {name} ---")
    print("File:", filepath)

    with rasterio.open(filepath) as src:

        data = src.read(1)

        print("CRS:", src.crs)
        print("Shape:", src.height, "x", src.width)
        print("Pixel size:", src.res)
        print("Bounds:", src.bounds)
        print("Data type:", src.dtypes[0])
        print("NoData:", src.nodata)

        finite = np.isfinite(data)

        print("Finite pixels:", np.sum(finite))
        print("Non-finite pixels:", np.sum(~finite))

        if name != "MASK":

            valid = data[finite]

            if len(valid) > 0:
                print("Min:", np.min(valid))
                print("Max:", np.max(valid))
                print("Mean:", np.mean(valid))
                print("Std:", np.std(valid))

        else:

            unique, counts = np.unique(data, return_counts=True)

            print("Mask values:")

            for value, count in zip(unique, counts):
                print(f"  {value}: {count}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)