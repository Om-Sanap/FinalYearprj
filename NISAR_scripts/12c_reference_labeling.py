"""
WAYANAD NISAR - INTERACTIVE REFERENCE LABELING

Input:
    NISAR_results/wayanad_reference_samples.csv

Feature rasters:
    NISAR_output/wayanad_HH_dB.tif
    NISAR_output/wayanad_HV_dB.tif
    NISAR_output/wayanad_HH_HV_ratio.tif
    NISAR_output/wayanad_normalized_difference.tif

Output:
    NISAR_results/wayanad_reference_samples_labeled.csv

Opens a local browser interface for independent reference labeling.
The original generated sample CSV is never modified.
"""

from pathlib import Path
from html import escape
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser
import threading

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "NISAR_scripts"
OUTPUT_DIR = BASE_DIR / "NISAR_output"
RESULTS_DIR = BASE_DIR / "NISAR_results"

CSV_FILE = RESULTS_DIR / "wayanad_reference_samples.csv"
OUTPUT_FILE = RESULTS_DIR / "wayanad_reference_samples_labeled.csv"

RASTER_FILES = {
    "HH_dB": OUTPUT_DIR / "wayanad_HH_dB.tif",
    "HV_dB": OUTPUT_DIR / "wayanad_HV_dB.tif",
    "HH_HV_ratio": OUTPUT_DIR / "wayanad_HH_HV_ratio.tif",
    "normalized_difference": OUTPUT_DIR / "wayanad_normalized_difference.tif",
}

PORT = 8765

REFERENCE_CLASSES = {
    "1": "Forest / dense vegetation",
    "2": "Agricultural / mixed vegetation",
    "3": "Built-up",
    "4": "Water / smooth surface",
    "5": "Bare / exposed surface",
    "6": "Wetland / waterlogged vegetation",
}

CONFIDENCE_LEVELS = ["High", "Medium", "Low"]


# ============================================================
# STARTUP CHECKS
# ============================================================

print("=" * 70)
print("WAYANAD NISAR - INTERACTIVE REFERENCE LABELING")
print("=" * 70)

if not CSV_FILE.is_file():
    raise FileNotFoundError(
        f"Sample CSV not found:\n{CSV_FILE}\n\n"
        "Run 12a_create_reference_samples.py first."
    )

for name, path in RASTER_FILES.items():
    if not path.is_file():
        raise FileNotFoundError(
            f"Required feature raster not found ({name}):\n{path}"
        )


# ============================================================
# LOAD GENERATED SAMPLES
# ============================================================

df = pd.read_csv(CSV_FILE, dtype={"sample_id": str})

if df.empty:
    raise ValueError("The reference-sample CSV is empty.")

required = [
    "sample_id",
    "x",
    "y",
    "predicted_class",
 ]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required CSV columns: " + ", ".join(missing)
    )

if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
    raise ValueError("sample_id must be present and unique for every sample.")

df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")

if (
    df["x"].isna().any()
    or df["y"].isna().any()
    or not np.isfinite(df["x"]).all()
    or not np.isfinite(df["y"]).all()
):
    raise ValueError("The CSV contains invalid x/y coordinates.")

for column in [
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes",
]:
    if column not in df.columns:
        df[column] = ""

# Normalize empty cells to strings so the HTML form is predictable.
for column in [
    "reference_class",
    "reference_class_name",
    "confidence",
    "notes",
]:
    df[column] = df[column].fillna("").astype(str)


# ============================================================
# PRESERVE PREVIOUSLY ENTERED LABELS
# ============================================================

# If this labeling tool was previously used, preserve labels by sample_id.
# The current generated CSV remains the authoritative source of coordinates.

if OUTPUT_FILE.is_file():
    try:
        old_df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"sample_id": str},
            keep_default_na=False,
        )

        if "sample_id" in old_df.columns:
            label_columns = [
                "reference_class",
                "reference_class_name",
                "confidence",
                "notes",
            ]

            old_labels = old_df.drop_duplicates("sample_id").set_index(
                "sample_id"
            )

            preserved = 0

            for column in label_columns:
                if column in old_labels.columns:
                    old_values = old_labels[column].to_dict()

                    for i, sample_id in enumerate(df["sample_id"]):
                        old_value = str(old_values.get(sample_id, "")).strip()

                        if old_value:
                            df.at[i, column] = old_value
                            if column == "reference_class":
                                preserved += 1

            print(
                "Existing labels recovered for matching sample IDs."
            )
            print(
                "Previously labeled samples preserved:",
                preserved,
            )

    except Exception as exc:
        print(
            "Warning: Could not restore previous labels:",
            exc,
        )
        print("The existing output file has not been deleted.")


# ============================================================
# OPEN AND VALIDATE FEATURE RASTERS
# ============================================================

print("\nOpening feature rasters...")

rasters = {}

try:
    for name, path in RASTER_FILES.items():
        rasters[name] = rasterio.open(path)

    reference = rasters["HH_dB"]

    if reference.crs is None:
        raise ValueError("HH_dB raster has no CRS.")

    for name, src in rasters.items():
        if src.crs != reference.crs:
            raise ValueError(
                f"CRS mismatch: {name} uses {src.crs}, "
                f"but HH_dB uses {reference.crs}."
            )

        if (
            src.width != reference.width
            or src.height != reference.height
        ):
            raise ValueError(
                f"Dimension mismatch in {name}. "
                "All feature rasters must share the same grid."
            )

        if src.transform != reference.transform:
            raise ValueError(
                f"Transform mismatch in {name}. "
                "All feature rasters must share the same grid."
            )

    print("CRS:", reference.crs)
    print("Raster dimensions:", reference.width, "x", reference.height)
    print("All four feature rasters share the same grid.")

except Exception:
    for src in rasters.values():
        src.close()
    raise


# ============================================================
# EXTRACT VALUES AT EACH SAMPLE LOCATION
# ============================================================

print("\nExtracting feature values at sample locations...")

feature_columns = [
    "HH_dB",
    "HV_dB",
    "HH_HV_ratio",
    "normalized_difference",
]

# Recompute values from the current feature rasters.
# Read only a single pixel at each sample location.
extracted = {name: [] for name in feature_columns}
inside_count = 0
invalid_count = 0

for _, sample in df.iterrows():
    x = float(sample["x"])
    y = float(sample["y"])

    row, col = reference.index(x, y)

    if not (
        0 <= row < reference.height
        and 0 <= col < reference.width
    ):
        for name in feature_columns:
            extracted[name].append(np.nan)
        invalid_count += 1
        continue

    inside_count += 1
    pixel_values = {}
    pixel_is_valid = True

    for name, src in rasters.items():
        value = src.read(
            1,
            window=Window(col, row, 1, 1),
            masked=True,
        )[0, 0]

        if np.ma.is_masked(value):
            pixel_is_valid = False
            pixel_values[name] = np.nan
        else:
            value = float(value)
            if not np.isfinite(value):
                pixel_is_valid = False
                pixel_values[name] = np.nan
            else:
                pixel_values[name] = value

    if not pixel_is_valid:
        invalid_count += 1

    for name in feature_columns:
        value = pixel_values[name]
        extracted[name].append(
            value if pixel_is_valid else np.nan
        )

for name in feature_columns:
    df[name] = extracted[name]

for src in rasters.values():
    src.close()

print("Samples loaded:", len(df))
print("Coordinates inside raster bounds:", inside_count)
print("Outside bounds or invalid feature pixels:", invalid_count)

if inside_count == 0:
    raise ValueError(
        "No sample coordinates fall inside the feature raster. "
        "Check the CSV coordinates and raster CRS."
    )


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_data():
    """Save labels without modifying the original sample CSV."""
    temp_file = OUTPUT_FILE.with_name(
        OUTPUT_FILE.stem + ".tmp" + OUTPUT_FILE.suffix
    )

    df.to_csv(temp_file, index=False)
    temp_file.replace(OUTPUT_FILE)


# ============================================================
# HTML HELPERS
# ============================================================

def safe(value):
    """Escape text before inserting it into HTML."""
    if value is None or pd.isna(value):
        return ""
    return escape(str(value), quote=True)


def display_number(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return f"{value:.5f}"
    except (TypeError, ValueError):
        pass
    return "No valid pixel value"


def selected_option(value, option):
    return "selected" if str(value) == str(option) else ""


# ============================================================
# BUILD BROWSER PAGE
# ============================================================

def make_page(index, message=""):
    row = df.iloc[index]

    sample_id = safe(row["sample_id"])
    x = float(row["x"])
    y = float(row["y"])

    predicted = safe(row.get("predicted_class", ""))
    predicted_name = safe(row.get("predicted_class_name", ""))

    reference_class = str(row.get("reference_class", "")).strip()
    reference_name = str(row.get("reference_class_name", "")).strip()
    confidence = str(row.get("confidence", "")).strip()
    notes = str(row.get("notes", "")).strip()

    valid_values = all(
        np.isfinite(pd.to_numeric(row[name], errors="coerce"))
        for name in feature_columns
    )

    metrics_html = ""
    for name in feature_columns:
        metrics_html += (
            f"<div class='metric'><b>{safe(name)}</b><br>"
            f"{display_number(row[name])}</div>"
        )

    class_options = [
        "<option value=''>-- Select reference class --</option>"
    ]

    for class_id, class_name in REFERENCE_CLASSES.items():
        class_options.append(
            f"<option value='{class_id}' "
            f"{selected_option(reference_class, class_id)}>"
            f"{class_id} - {safe(class_name)}</option>"
        )

    confidence_options = [
        "<option value=''>-- Select confidence --</option>"
    ]

    for level in CONFIDENCE_LEVELS:
        confidence_options.append(
            f"<option value='{level}' "
            f"{selected_option(confidence, level)}>"
            f"{level}</option>"
        )

    message_html = (
        f"<div class='message'>{safe(message)}</div>"
        if message else ""
    )

    data_warning = ""
    if not valid_values:
        data_warning = (
            "<div class='warning'>One or more feature values are "
            "invalid or NoData. Check this sample before labeling it.</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wayanad NISAR Reference Labeling</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background: #f2f4f7;
    margin: 0;
    padding: 24px;
    color: #202124;
}}
.container {{
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 28px;
    border-radius: 10px;
    box-shadow: 0 2px 12px #0001;
}}
h1 {{ margin-top: 0; }}
.info {{
    background: #f0f2f5;
    padding: 16px;
    border-radius: 8px;
    margin: 14px 0;
}}
.metric {{
    display: inline-block;
    vertical-align: top;
    width: 42%;
    padding: 10px 2%;
    overflow-wrap: anywhere;
}}
label {{
    display: block;
    font-weight: bold;
    margin-top: 14px;
}}
select, textarea, input[type=text] {{
    width: 100%;
    padding: 10px;
    margin-top: 6px;
    box-sizing: border-box;
}}
button {{
    padding: 11px 18px;
    margin: 12px 6px 0 0;
    border: 0;
    border-radius: 5px;
    cursor: pointer;
}}
.save {{ background: #217346; color: white; }}
.nav {{ background: #245b9e; color: white; }}
.message {{ background: #e8f4e8; padding: 12px; margin-bottom: 12px; }}
.warning {{ background: #fff1d6; padding: 12px; margin: 12px 0; }}
.small {{ color: #555; font-size: 0.92em; }}
</style>
</head>
<body>
<div class="container">
<h1>Wayanad NISAR Reference Labeling</h1>
{message_html}

<div class="info">
<b>Sample:</b> {index + 1} of {len(df)}<br>
<b>Sample ID:</b> {sample_id}<br>
<b>Coordinates:</b> X = {x:.2f}, Y = {y:.2f}<br>
<b>CRS:</b> {safe(str(reference.crs) if 'reference' in globals() else 'See raster metadata')}<br>
<b>Predicted K-Means cluster:</b> {predicted}<br>
<b>Predicted class name, if provided:</b> {predicted_name or 'Not provided'}
</div>

{data_warning}

<div class="info">
<h3>NISAR feature values</h3>
{metrics_html}
</div>

<p class="small">
Assign the reference class using independent reference imagery or
reliable geographic data. Do not copy the predicted cluster into
the reference class. If evidence is insufficient, leave the class
blank and explain why in Notes.
</p>

<form method="POST" action="/save">
<input type="hidden" name="index" value="{index}">

<label>Reference class</label>
<select name="reference_class">
{''.join(class_options)}
</select>

<label>Reference class name</label>
<input type="text" name="reference_class_name"
value="{safe(reference_name)}"
placeholder="Optional descriptive name">

<label>Confidence</label>
<select name="confidence">
{''.join(confidence_options)}
</select>

<label>Notes / evidence source</label>
<textarea name="notes" rows="4"
placeholder="Reference imagery, observation date, uncertainty, etc.">{safe(notes)}</textarea>

<button class="save" type="submit">Save and go to next sample</button>
</form>

<div>
<form method="GET" action="/" style="display:inline">
<input type="hidden" name="index" value="{max(0, index - 1)}">
<button class="nav" type="submit">Previous sample</button>
</form>

<form method="GET" action="/" style="display:inline">
<input type="hidden" name="index" value="{min(len(df) - 1, index + 1)}">
<button class="nav" type="submit">Next sample without saving</button>
</form>
</div>

<p class="small">
Labels saved to: {safe(str(OUTPUT_FILE))}
</p>
</div>
</body>
</html>"""


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format_string, *args):
        # Keep the terminal output concise.
        return

    def send_html(self, html_text, status=200):
        content = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path != "/":
            self.send_html("Not found", 404)
            return

        params = parse_qs(parsed.query)

        try:
            index = int(params.get("index", ["0"])[0])
        except (ValueError, TypeError):
            index = 0

        index = max(0, min(index, len(df) - 1))
        self.send_html(make_page(index))

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/save":
            self.send_html("Not found", 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = parse_qs(body, keep_blank_values=True)

            index = int(data["index"][0])
            if not 0 <= index < len(df):
                raise ValueError("Invalid sample index.")

            reference_class = data.get("reference_class", [""])[0].strip()
            reference_name = data.get("reference_class_name", [""])[0].strip()
            confidence = data.get("confidence", [""])[0].strip()
            notes = data.get("notes", [""])[0].strip()

            if reference_class and reference_class not in REFERENCE_CLASSES:
                raise ValueError("Invalid reference class selected.")

            if confidence and confidence not in CONFIDENCE_LEVELS:
                raise ValueError("Invalid confidence value selected.")

            # Set a default name for selected categories when left blank.
            if reference_class and not reference_name:
                reference_name = REFERENCE_CLASSES[reference_class]

            df.at[index, "reference_class"] = reference_class
            df.at[index, "reference_class_name"] = reference_name
            df.at[index, "confidence"] = confidence
            df.at[index, "notes"] = notes

            save_data()

            next_index = min(index + 1, len(df) - 1)
            self.send_response(303)
            self.send_header("Location", f"/?index={next_index}")
            self.end_headers()

        except Exception as exc:
            self.send_html(
                "<html><body><h2>Could not save sample</h2>"
                f"<p>{safe(str(exc))}</p>"
                "<p><a href='/'>Return to labeling</a></p></body></html>",
                400,
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    save_data()

    server = HTTPServer(("127.0.0.1", PORT), Handler)

    url = f"http://127.0.0.1:{PORT}/"

    print("\n" + "=" * 70)
    print("REFERENCE LABELING INTERFACE READY")
    print("=" * 70)
    print("Browser URL:", url)
    print("Samples:", len(df))
    print("Output CSV:", OUTPUT_FILE)
    print("\nUse independent reference information to assign labels.")
    print("Save each sample before stopping the server.")
    print("Press CTRL+C in this terminal to stop.")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeling server...")
    finally:
        server.server_close()
        print("Server stopped.")