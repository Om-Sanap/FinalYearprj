import h5py
from pathlib import Path

# Folder containing the NISAR HDF5 file
DATA_DIR = Path(r"D:\FinalYearPrj\prj\NISAR_original")

# Automatically find HDF5 files
files = list(DATA_DIR.glob("*.h5"))

if not files:
    raise FileNotFoundError(f"No .h5 file found in {DATA_DIR}")

if len(files) > 1:
    print("Multiple HDF5 files found:")
    for i, file in enumerate(files, 1):
        print(f"{i}: {file.name}")

FILE = files[0]

print("=" * 80)
print("NISAR HDF5 FILE")
print("=" * 80)
print(FILE)
print("=" * 80)

with h5py.File(FILE, "r") as f:

    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(
                f"{name} | "
                f"shape={obj.shape} | "
                f"dtype={obj.dtype}"
            )

    f.visititems(show)

print("\nInspection complete.")