import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# WAYANAD NISAR REFERENCE POINT CREATION
# ============================================================

print("=" * 70)
print("WAYANAD NISAR REFERENCE POINT CREATION")
print("=" * 70)


BASE_DIR = r"D:\FinalYearPrj\prj"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "NISAR_results",
    "wayanad_reference_samples.csv"
)

OUTPUT_GPKG = os.path.join(
    BASE_DIR,
    "NISAR_results",
    "wayanad_reference_samples.gpkg"
)

OUTPUT_SHP = os.path.join(
    BASE_DIR,
    "NISAR_results",
    "wayanad_reference_samples.shp"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"\nReference CSV not found:\n{INPUT_CSV}"
    )


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading reference samples...")

df = pd.read_csv(INPUT_CSV)

print("Samples:", len(df))


# ============================================================
# CREATE GEOMETRY
# ============================================================

geometry = [
    Point(x, y)
    for x, y in zip(
        df["x"],
        df["y"]
    )
]


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:32643"
)


# ============================================================
# SAVE GEOPACKAGE
# ============================================================

print("\nSaving GeoPackage...")

if os.path.exists(OUTPUT_GPKG):
    os.remove(OUTPUT_GPKG)

gdf.to_file(
    OUTPUT_GPKG,
    layer="reference_samples",
    driver="GPKG"
)

print("Saved:")
print(OUTPUT_GPKG)


# ============================================================
# SAVE SHAPEFILE
# ============================================================

print("\nSaving Shapefile...")

# Remove old shapefile components if they exist
for ext in [
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg"
]:
    path = OUTPUT_SHP.replace(".shp", ext)

    if os.path.exists(path):
        os.remove(path)


gdf.to_file(
    OUTPUT_SHP,
    driver="ESRI Shapefile"
)

print("Saved:")
print(OUTPUT_SHP)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("REFERENCE POINT CREATION COMPLETE")
print("=" * 70)

print("\nTotal points:", len(gdf))

print("\nCRS:")
print(gdf.crs)

print("\nPredicted class distribution:")

print(
    gdf["predicted_class"]
    .value_counts()
    .sort_index()
)


print("\nCreated files:")

print(OUTPUT_GPKG)
print(OUTPUT_SHP)

print("\n")
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print("""
Open the GeoPackage in QGIS.

Layer:
    reference_samples

Use independent reference information to determine
the actual land-cover/reference class at each point.

DO NOT simply copy predicted_class.

Fill:

    reference_class
    reference_class_name
    confidence
    notes

The reference labels must come from independent
geographic/reference information.
""")

print("=" * 70)