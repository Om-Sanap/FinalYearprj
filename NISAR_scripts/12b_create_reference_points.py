
"""
WAYANAD NISAR - REFERENCE POINT CREATION

Converts the reference-sample CSV into:
1. GeoPackage (.gpkg) - recommended primary GIS format
2. ESRI Shapefile (.shp) - optional compatibility format

The reference labels are intentionally not assigned by this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "NISAR_results"

INPUT_CSV = RESULTS_DIR / "wayanad_reference_samples.csv"

OUTPUT_GPKG = RESULTS_DIR / "wayanad_reference_samples.gpkg"
OUTPUT_SHP = RESULTS_DIR / "wayanad_reference_samples.shp"

GPKG_LAYER = "reference_samples"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WAYANAD NISAR REFERENCE POINT CREATION")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_CSV.is_file():
    raise FileNotFoundError(
        f"Reference sample CSV not found:\n{INPUT_CSV}\n\n"
        "Run 12b_create_reference_points.py first."
    )


# ============================================================
# LOAD AND VALIDATE CSV
# ============================================================

print("\nLoading reference samples...")

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError("The reference sample CSV contains no rows.")

required_columns = [
    "sample_id",
    "x",
    "y",
    "predicted_class",
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The CSV is missing required columns: "
        + ", ".join(missing_columns)
    )

# Coordinates must be numeric and finite.
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")

bad_coordinates = (
    df["x"].isna()
    | df["y"].isna()
    | ~np.isfinite(df["x"])
    | ~np.isfinite(df["y"])
)

if bad_coordinates.any():
    bad_count = int(bad_coordinates.sum())
    raise ValueError(
        f"{bad_count} sample(s) have missing or invalid coordinates. "
        "Correct the CSV before exporting."
    )

if df["sample_id"].isna().any():
    raise ValueError("Some samples have a missing sample_id.")

if df["sample_id"].duplicated().any():
    raise ValueError("sample_id values must be unique.")

# Read CRS from the CSV if available; otherwise use the known
# extraction CRS as a fallback.
if "crs" in df.columns:
    crs_values = df["crs"].dropna().astype(str).str.strip()
    crs_values = crs_values[crs_values != ""].unique()

    if len(crs_values) > 1:
        raise ValueError(
            f"The CSV contains multiple CRS values: {crs_values.tolist()}"
        )

    if len(crs_values) == 1:
        source_crs = crs_values[0]
    else:
        source_crs = "EPSG:32643"
        print("Warning: CRS column is empty; using EPSG:32643.")
else:
    source_crs = "EPSG:32643"
    print("Warning: CSV has no CRS column; using EPSG:32643.")

try:
    parsed_crs = gpd.GeoSeries([], crs=source_crs).crs
except Exception as exc:
    raise ValueError(f"Could not interpret CRS: {source_crs}") from exc

if parsed_crs is None:
    raise ValueError("Could not determine the coordinate reference system.")

print("Samples:", len(df))
print("Source CRS:", parsed_crs)


# ============================================================
# CREATE POINT GEOMETRY
# ============================================================

print("\nCreating point geometry...")

geometry = [
    Point(x, y)
    for x, y in zip(df["x"], df["y"])
]

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs=parsed_crs,
)

if gdf.geometry.is_empty.any() or (~gdf.geometry.is_valid).any():
    raise ValueError("One or more generated point geometries are invalid.")


# ============================================================
# SAVE GEOPACKAGE
# ============================================================

print("\nSaving GeoPackage...")

# Replace an existing GeoPackage only after input validation.
if OUTPUT_GPKG.exists():
    OUTPUT_GPKG.unlink()

gdf.to_file(
    OUTPUT_GPKG,
    layer=GPKG_LAYER,
    driver="GPKG",
    index=False,
)

print("Saved:", OUTPUT_GPKG)
print("Layer:", GPKG_LAYER)


# ============================================================
# SAVE SHAPEFILE
# ============================================================

print("\nSaving Shapefile...")

# Shapefiles consist of multiple files sharing the same base name.
# Remove old components to avoid mixing old and new exports.
shapefile_extensions = [
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".qix",
]

for extension in shapefile_extensions:
    component = OUTPUT_SHP.with_suffix(extension)

    if component.exists():
        component.unlink()

gdf.to_file(
    OUTPUT_SHP,
    driver="ESRI Shapefile",
    index=False,
)

print("Saved:", OUTPUT_SHP)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE POINT CREATION COMPLETE")
print("=" * 70)

print("Total points:", len(gdf))
print("CRS:", gdf.crs)

print("\nPredicted cluster distribution:")
print(
    gdf["predicted_class"]
    .value_counts()
    .sort_index()
    .to_string()
)

labeled = (
    gdf["reference_class"].notna()
    & gdf["reference_class"].astype(str).str.strip().ne("")
)

print("\nSamples with reference_class already populated:", int(labeled.sum()))
print("Samples still awaiting reference labels:", int((~labeled).sum()))

print("\nCreated:")
print(OUTPUT_GPKG)
print(OUTPUT_SHP)

print("\nNEXT STEP")
print("-" * 70)
print(
    "Open wayanad_reference_samples.gpkg in QGIS and inspect "
    "the sample locations against independent reference information."
)
print(
    "Assign reference_class, reference_class_name, confidence, "
    "and notes based on that evidence. Do not copy predicted_class "
    "into reference_class."
)
print("=" * 70)
