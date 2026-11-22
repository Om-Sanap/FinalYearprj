import pandas as pd
import rasterio
from pyproj import Transformer
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time


# ============================================================
# FILE PATHS
# ============================================================

import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent)

INPUT_CSV = os.path.join(BASE_DIR, "NISAR_results", "wayanad_reference_samples_labeled.csv")
INPUT_TIF = os.path.join(BASE_DIR, "NISAR_output", "wayanad_HH_dB.tif") # Using the dB image or just wayanad_HH if it exists, let me check. Actually in earlier scripts it was wayanad_HH_dB.tif

OUTPUT_CSV = os.path.join(BASE_DIR, "NISAR_results", "wayanad_reference_samples_with_location.csv")


# ============================================================
# 1. READ CSV
# ============================================================

print("Reading CSV...")

df = pd.read_csv(INPUT_CSV)

print("\nOriginal CSV columns:")
print(df.columns.tolist())

print(f"\nNumber of samples: {len(df)}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "sample_id",
    "row",
    "column",
    "x",
    "y",
    "predicted_class",
    "predicted_class_name",
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )

print("\nAll required CSV columns are present.")


# ============================================================
# 3. READ TIFF CRS
# ============================================================

print("\nReading TIFF information...")

with rasterio.open(INPUT_TIF) as src:

    raster_crs = src.crs
    raster_width = src.width
    raster_height = src.height
    raster_transform = src.transform
    raster_bounds = src.bounds

print("\nTIFF CRS:")
print(raster_crs)

print("\nTIFF size:")
print(f"Width  : {raster_width}")
print(f"Height : {raster_height}")

print("\nTIFF bounds:")
print(raster_bounds)

print("\nTIFF transform:")
print(raster_transform)


# ============================================================
# 4. CREATE CRS TRANSFORMER
# ============================================================

# Your TIFF uses:
# EPSG:32643 = WGS 84 / UTM Zone 43N
#
# We convert:
# UTM X/Y
#      ↓
# WGS84
#      ↓
# Longitude / Latitude

transformer = Transformer.from_crs(
    raster_crs,
    "EPSG:4326",
    always_xy=True
)


# ============================================================
# 5. CONVERT X/Y TO LATITUDE/LONGITUDE
# ============================================================

print("\nConverting X/Y coordinates to latitude/longitude...")

latitudes = []
longitudes = []

for index, row_data in df.iterrows():

    x = row_data["x"]
    y = row_data["y"]

    try:

        longitude, latitude = transformer.transform(
            x,
            y
        )

        longitudes.append(longitude)
        latitudes.append(latitude)

    except Exception as e:

        print(
            f"Coordinate conversion failed for "
            f"sample {row_data['sample_id']}: {e}"
        )

        longitudes.append(None)
        latitudes.append(None)


# Add the new columns

df["latitude"] = latitudes
df["longitude"] = longitudes


# ============================================================
# 6. DISPLAY COORDINATE RESULTS
# ============================================================

print("\nCoordinate conversion completed.")

print("\nSample coordinate results:")

print(
    df[
        [
            "sample_id",
            "x",
            "y",
            "latitude",
            "longitude"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 7. REVERSE GEOCODING
# ============================================================

print("\nStarting reverse geocoding...")
print("This may take some time because each request is rate-limited.")


geolocator = Nominatim(
    user_agent="nisar_wayanad_location_analysis_2026"
)


cities = []
countries = []


for index, row_data in df.iterrows():

    latitude = row_data["latitude"]
    longitude = row_data["longitude"]

    sample_id = row_data["sample_id"]

    print(
        f"Processing sample {index + 1}/{len(df)} "
        f"(sample_id={sample_id})..."
    )

    if pd.isna(latitude) or pd.isna(longitude):

        cities.append("")
        countries.append("")

        continue

    try:

        location = geolocator.reverse(
            (latitude, longitude),
            exactly_one=True,
            language="en",
            zoom=10,
            addressdetails=True
        )

        if location is not None:

            address = location.raw.get(
                "address",
                {}
            )

            # Try different locality fields because
            # rural areas may not have a "city".

            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
                or address.get("district")
                or address.get("county")
                or address.get("state_district")
                or ""
            )

            country = address.get(
                "country",
                ""
            )

        else:

            city = ""
            country = ""

    except GeocoderTimedOut:

        print(
            f"Geocoding timeout for sample {sample_id}"
        )

        city = ""
        country = ""

    except GeocoderServiceError as e:

        print(
            f"Geocoding service error for sample "
            f"{sample_id}: {e}"
        )

        city = ""
        country = ""

    except Exception as e:

        print(
            f"Unexpected error for sample "
            f"{sample_id}: {e}"
        )

        city = ""
        country = ""

    cities.append(city)
    countries.append(country)

    # Nominatim requires rate limiting.
    time.sleep(1)


# ============================================================
# 8. ADD CITY AND COUNTRY
# ============================================================

df["city"] = cities
df["country"] = countries


# ============================================================
# 9. ARRANGE COLUMNS
# ============================================================

# Keep ALL original columns exactly as they were,
# then add latitude, longitude, city and country.

final_columns = [col for col in df.columns]

df = df[final_columns]


# ============================================================
# 10. SAVE FINAL CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)

print(f"\nInput CSV  : {INPUT_CSV}")
print(f"Input TIFF : {INPUT_TIF}")
print(f"Output CSV : {OUTPUT_CSV}")

print(f"\nTotal samples processed: {len(df)}")

print("\nFinal columns:")

for column in df.columns:
    print(f" - {column}")


print("\nFirst 10 rows of final CSV:")

print(
    df.head(10).to_string(index=False)
)

print("\nOutput file created successfully!")