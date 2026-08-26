import os
import webbrowser
import pandas as pd
import numpy as np
import rasterio
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\FinalYearPrj\prj"

CSV_FILE = os.path.join(
    BASE_DIR,
    "NISAR_results",
    "wayanad_reference_samples.csv"
)

HH_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_HH_dB.tif"
)

HV_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_HV_dB.tif"
)

RATIO_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_HH_HV_ratio.tif"
)

ND_FILE = os.path.join(
    BASE_DIR,
    "NISAR_output",
    "wayanad_normalized_difference.tif"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "NISAR_results",
    "wayanad_reference_samples_labeled.csv"
)

# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("WAYANAD NISAR REFERENCE LABELING")
print("=" * 70)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV not found:\n{CSV_FILE}")

df = pd.read_csv(CSV_FILE)

print(f"Samples loaded: {len(df)}")

# ------------------------------------------------------------
# Find coordinate columns automatically
# ------------------------------------------------------------

possible_x = [
    "x",
    "X",
    "easting",
    "Easting",
    "utm_x",
    "utm_easting"
]

possible_y = [
    "y",
    "Y",
    "northing",
    "Northing",
    "utm_y",
    "utm_northing"
]

x_col = next((c for c in possible_x if c in df.columns), None)
y_col = next((c for c in possible_y if c in df.columns), None)

if x_col is None or y_col is None:
    raise ValueError(
        "Could not find coordinate columns in CSV.\n"
        f"Columns found: {list(df.columns)}"
    )

print(f"X coordinate column: {x_col}")
print(f"Y coordinate column: {y_col}")

# ============================================================
# ADD LABEL COLUMNS IF NECESSARY
# ============================================================

if "reference_class" not in df.columns:
    df["reference_class"] = ""

if "reference_class_name" not in df.columns:
    df["reference_class_name"] = ""

if "confidence" not in df.columns:
    df["confidence"] = ""

if "notes" not in df.columns:
    df["notes"] = ""

# ============================================================
# LOAD RASTERS
# ============================================================

print("\nLoading NISAR rasters...")

hh_src = rasterio.open(HH_FILE)
hv_src = rasterio.open(HV_FILE)
ratio_src = rasterio.open(RATIO_FILE)
nd_src = rasterio.open(ND_FILE)

print("Rasters loaded.")
print(f"CRS: {hh_src.crs}")

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    1: "Mixed vegetation / agricultural surface",
    2: "Built-up / exposed structured surface",
    3: "Strong structural / double-bounce-like scattering",
    4: "Low-backscatter smooth surface",
    5: "Forest / dense vegetation"
}

# ============================================================
# GET RASTER VALUES
# ============================================================

def get_values(x, y):

    try:
        row, col = hh_src.index(x, y)

        if (
            row < 0 or
            col < 0 or
            row >= hh_src.height or
            col >= hh_src.width
        ):
            return np.nan, np.nan, np.nan, np.nan

        hh = hh_src.read(1)[row, col]
        hv = hv_src.read(1)[row, col]
        ratio = ratio_src.read(1)[row, col]
        nd = nd_src.read(1)[row, col]

        return hh, hv, ratio, nd

    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ============================================================
# PRE-CALCULATE RASTER VALUES
# ============================================================

print("\nExtracting NISAR values for samples...")

hh_values = []
hv_values = []
ratio_values = []
nd_values = []

for _, row in df.iterrows():

    hh, hv, ratio, nd = get_values(
        row[x_col],
        row[y_col]
    )

    hh_values.append(hh)
    hv_values.append(hv)
    ratio_values.append(ratio)
    nd_values.append(nd)

df["HH_dB"] = hh_values
df["HV_dB"] = hv_values
df["HH_HV_ratio"] = ratio_values
df["normalized_difference"] = nd_values

# ============================================================
# SAVE FUNCTION
# ============================================================

def save_data():

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved progress:")
    print(OUTPUT_FILE)


# ============================================================
# HTML INTERFACE
# ============================================================

def make_page(index):

    row = df.iloc[index]

    predicted = row.get("predicted_class", "")
    predicted_name = CLASS_NAMES.get(
        int(predicted) if str(predicted).isdigit() else 0,
        ""
    )

    reference = row.get("reference_class", "")
    reference_name = row.get("reference_class_name", "")
    confidence = row.get("confidence", "")
    notes = row.get("notes", "")

    hh = row["HH_dB"]
    hv = row["HV_dB"]
    ratio = row["HH_HV_ratio"]
    nd = row["normalized_difference"]

    return f"""
<!DOCTYPE html>

<html>

<head>

<title>Wayanad NISAR Reference Labeling</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f4f4f4;
}}

.container {{
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 10px;
}}

h1 {{
    margin-top: 0;
}}

.info {{
    background: #eeeeee;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}}

.metric {{
    display: inline-block;
    width: 180px;
    padding: 10px;
}}

label {{
    font-weight: bold;
}}

select,
textarea,
input {{
    width: 100%;
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 15px;
    box-sizing: border-box;
}}

button {{
    padding: 12px 20px;
    margin: 5px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}}

.save {{
    background: #4CAF50;
    color: white;
}}

.next {{
    background: #2196F3;
    color: white;
}}

.prev {{
    background: #777;
    color: white;
}}

</style>

</head>

<body>

<div class="container">

<h1>Wayanad NISAR Reference Labeling</h1>

<div class="info">

<b>Sample:</b> {index + 1} / {len(df)}<br>

<b>Coordinates:</b>
{x_col} = {row[x_col]:.2f},
{y_col} = {row[y_col]:.2f}

<br><br>

<b>Predicted K-Means Class:</b>
{predicted}

<br>

<b>Provisional interpretation:</b>
{predicted_name}

</div>

<div class="info">

<h3>NISAR characteristics</h3>

<div class="metric">
<b>HH dB</b><br>
{hh:.4f}
</div>

<div class="metric">
<b>HV dB</b><br>
{hv:.4f}
</div>

<div class="metric">
<b>HH/HV Ratio</b><br>
{ratio:.4f}
</div>

<div class="metric">
<b>Normalized Difference</b><br>
{nd:.4f}
</div>

</div>

<form method="POST" action="/save">

<input type="hidden" name="index" value="{index}">

<label>Reference Class</label>

<select name="reference_class">

<option value="">-- Select --</option>

<option value="1" {"selected" if str(reference) == "1" else ""}>
1 - Forest / dense vegetation
</option>

<option value="2" {"selected" if str(reference) == "2" else ""}>
2 - Agricultural / mixed vegetation
</option>

<option value="3" {"selected" if str(reference) == "3" else ""}>
3 - Built-up
</option>

<option value="4" {"selected" if str(reference) == "4" else ""}>
4 - Water / smooth surface
</option>

<option value="5" {"selected" if str(reference) == "5" else ""}>
5 - Bare / exposed surface
</option>

</select>

<label>Reference Class Name</label>

<input
type="text"
name="reference_class_name"
value="{reference_name}"
placeholder="Example: Forest">

<label>Confidence</label>

<select name="confidence">

<option value="">-- Select --</option>

<option {"selected" if confidence == "High" else ""}>
High
</option>

<option {"selected" if confidence == "Medium" else ""}>
Medium
</option>

<option {"selected" if confidence == "Low" else ""}>
Low
</option>

</select>

<label>Notes</label>

<textarea
name="notes"
rows="5"
placeholder="Describe why this reference label was assigned..."
>{notes}</textarea>

<button class="save" type="submit">
Save
</button>

</form>

<form method="GET" action="/" style="display:inline;">

<input
type="hidden"
name="index"
value="{max(0, index - 1)}">

<button class="prev">
Previous
</button>

</form>

<form method="GET" action="/" style="display:inline;">

<input
type="hidden"
name="index"
value="{min(len(df) - 1, index + 1)}">

<button class="next">
Next
</button>

</form>

</div>

</body>

</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        parsed = urlparse(self.path)

        if parsed.path == "/":

            params = parse_qs(parsed.query)

            try:
                index = int(params.get("index", [0])[0])
            except:
                index = 0

            index = max(
                0,
                min(index, len(df) - 1)
            )

            page = make_page(index)

            self.send_response(200)
            self.send_header(
                "Content-type",
                "text/html; charset=utf-8"
            )
            self.end_headers()

            self.wfile.write(
                page.encode("utf-8")
            )

    def do_POST(self):

        global df

        parsed = urlparse(self.path)

        if parsed.path == "/save":

            length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(length).decode()

            data = parse_qs(body)

            index = int(
                data["index"][0]
            )

            reference_class = data.get(
                "reference_class",
                [""]
            )[0]

            reference_name = data.get(
                "reference_class_name",
                [""]
            )[0]

            confidence = data.get(
                "confidence",
                [""]
            )[0]

            notes = data.get(
                "notes",
                [""]
            )[0]

            df.at[
                index,
                "reference_class"
            ] = reference_class

            df.at[
                index,
                "reference_class_name"
            ] = reference_name

            df.at[
                index,
                "confidence"
            ] = confidence

            df.at[
                index,
                "notes"
            ] = notes

            save_data()

            next_index = min(
                len(df) - 1,
                index + 1
            )

            self.send_response(302)

            self.send_header(
                "Location",
                f"/?index={next_index}"
            )

            self.end_headers()


# ============================================================
# START
# ============================================================

save_data()

PORT = 8765

server = HTTPServer(
    ("localhost", PORT),
    Handler
)

url = f"http://localhost:{PORT}/"

print("\n" + "=" * 70)
print("REFERENCE LABELING INTERFACE READY")
print("=" * 70)

print(f"\nOpen this address in your browser:")
print(url)

print("\nUse the browser interface to:")
print("1. Inspect each sample")
print("2. Determine the independent reference class")
print("3. Select confidence")
print("4. Add notes")
print("5. Click Save")
print("6. Continue to the next sample")

print("\nIMPORTANT:")
print("Do NOT simply use the predicted K-Means class.")
print("Use independent geographic/reference information.")

print("\nPress CTRL+C in this terminal to stop the server.")

webbrowser.open(url)

server.serve_forever()