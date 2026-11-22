import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(RESULTS_DIR, exist_ok=True)


FILES = {
    "HH dB": (
        os.path.join(INPUT_DIR, "wayanad_HH_dB.tif"),
        "wayanad_HH_dB_map.png",
        "HH Backscatter (dB)"
    ),
    "HV dB": (
        os.path.join(INPUT_DIR, "wayanad_HV_dB.tif"),
        "wayanad_HV_dB_map.png",
        "HV Backscatter (dB)"
    ),
    "HH/HV Ratio": (
        os.path.join(INPUT_DIR, "wayanad_HH_HV_ratio.tif"),
        "wayanad_HH_HV_ratio_map.png",
        "HH / HV Ratio"
    ),
    "Normalized Difference": (
        os.path.join(INPUT_DIR, "wayanad_normalized_difference.tif"),
        "wayanad_normalized_difference_map.png",
        "(HH-HV)/(HH+HV)"
    )
}


print("=" * 70)
print("WAYANAD NISAR FEATURE VISUALIZATION")
print("=" * 70)


for name, (input_file, output_name, colorbar_label) in FILES.items():

    print(f"\nProcessing: {name}")

    with rasterio.open(input_file) as src:
        data = src.read(1).astype(np.float32)

    valid = np.isfinite(data)

    if not np.any(valid):
        print("WARNING: No valid pixels")
        continue

    # Robust display limits
    low = np.percentile(data[valid], 2)
    high = np.percentile(data[valid], 98)

    print("2nd percentile :", low)
    print("98th percentile:", high)

    plt.figure(figsize=(10, 12))

    plt.imshow(
        data,
        cmap="gray",
        vmin=low,
        vmax=high
    )

    plt.title(f"NISAR Wayanad - {name}")
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")

    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, output_name)

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


print("\n" + "=" * 70)
print("FEATURE VISUALIZATION COMPLETE")
print("=" * 70)

print("\nResults saved in:")
print(RESULTS_DIR)