from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.io

# -----------------------
# --- Configuration ----
# -----------------------

# Base directory containing data files
BASE_DIR = Path("./data")

# Names of input Excel file, .mat subdirectory, and output CSV file
EXCEL_NAME = "3 dimensional data concentrations.xlsx"
MAT_SUBDIR = ""          # leave empty if .mat files are directly under BASE_DIR
OUT_NAME = "data.csv"

EXCEL_PATH = BASE_DIR / EXCEL_NAME
MAT_DIR    = BASE_DIR / MAT_SUBDIR
OUT_PATH   = OUT_NAME

# Window size for moving‐average smoothing
SMOOTH_WIN = 5

# Mapping of essential reagents from Excel labels to output column names
ESSENTIALS = {
    "TMB":  "tmb",
    "HRP":  "hrp",
    "H2O2": "h2o2",
}

# List of real additive labels as they appear in Excel (case/spacing normalized later)
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PAA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PVA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]

def sanitize(label: str) -> str:
    """
    Convert a raw label into a clean prefix for column names:
      - Replace non-alphanumeric characters with underscores
      - Convert to lowercase
      - Strip leading/trailing underscores
    """
    cleaned = re.sub(r'[^0-9A-Za-z]+', '_', label)
    return cleaned.lower().strip('_')

# Build a mapping from normalized Excel label to clean prefix
ADDITIVE_MAP = {
    label.upper().replace(" ", ""): sanitize(label)
    for label in ADDITIVE_LABELS
}

# Combine essential reagents and additives into one lookup dictionary
ALL_COMPONENTS = {**ESSENTIALS, **ADDITIVE_MAP}


# -----------------------
# --- Helper Functions --
# -----------------------

def smooth_curve(y: np.ndarray, window: int = SMOOTH_WIN) -> np.ndarray:
    """
    Apply moving‐average smoothing to a 1D signal.
    """
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

def metrics_from_curve(y: np.ndarray, window: int = SMOOTH_WIN):
    """
    Compute metrics from a time‐series curve:
      - t90      : index where smoothed curve first reaches 90% of its max
      - I_max    : maximum of the smoothed curve
      - tail_pct : mean of the last 5 raw points as a percent of I_max
    Returns (t90, I_max, tail_pct).
    """
    y_s = smooth_curve(y, window)
    if y_s.size == 0:
        return len(y) - 1, 0.0, 0.0

    ymax = float(y_s.max())
    t90 = int(np.argmax(y_s >= 0.9 * ymax))
    tail_pct = float(y[-5:].mean() / ymax * 100.0) if ymax > 0 else 0.0
    return t90, ymax, tail_pct

def collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    """
    Starting from label_row, read up to n_rows of 24 numeric values each:
      - Only columns 1..24 (df.iloc[:, 1:25]) are used.
      - Stop early if the next row’s first cell matches another component label.
      - If fewer than n_rows are found, pad by repeating the last row or using zeros.
    Returns an (n_rows × 24) numpy array.
    """
    rows = []
    r = label_row

    while r < len(df) and len(rows) < n_rows:
        first_cell = str(df.iat[r, 0]).upper().replace(" ", "")
        if r != label_row and first_cell in ALL_COMPONENTS:
            break

        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(dtype=float, copy=True))
        r += 1

    # Pad if necessary
    while len(rows) < n_rows:
        rows.append(rows[-1].copy() if rows else np.zeros(24))

    return np.vstack(rows[:n_rows])

def parse_sheet(df: pd.DataFrame) -> dict:
    """
    Parse one Excel sheet into a dict of component → (16×24) concentration array.
    Components not found are returned as zeros.
    """
    comps = {}
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")

    for label in ALL_COMPONENTS:
        matches = df.index[col0 == label]
        if not matches.empty:
            comps[label] = collect_block(df, matches[0])
        else:
            comps[label] = np.zeros((16, 24), dtype=float)

    return comps

def load_curve_cube(mat_path: Path) -> np.ndarray:
    """
    Load the MATLAB .mat file, extract `Output_c`, and stack into a (T × 16 × 24) array.
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # Each entry in data["Output_c"] is a 16×24 array for one timepoint
    return np.stack(list(data["Output_c"]), axis=0)


# -----------------------
# --- Main Processing ---
# -----------------------

def main():
    # Verify the Excel file exists
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # Read all sheets
    xls = pd.ExcelFile(EXCEL_PATH)
    records = []

    for sheet_name in xls.sheet_names:
        # Match sheets named "Trial N"
        m = re.match(r"Trial\s*(\d+)", sheet_name, flags=re.I)
        if not m:
            print(f"⚠ Skipping sheet {sheet_name}")
            continue

        trial = int(m.group(1))
        mat_path = MAT_DIR / f"Trial {trial}.mat"
        if not mat_path.exists():
            print(f"⚠ Missing {mat_path}, skipping Trial {trial}")
            continue

        # Parse Excel sheet for concentration blocks
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df)

        # Load kinetic curves and compute metrics for each grid point
        cube = load_curve_cube(mat_path)  # shape: (time, 16, 24)
        for r in range(16):
            for c in range(24):
                t90, I_max, tail_pct = metrics_from_curve(cube[:, r, c], SMOOTH_WIN)

                row = []
                # Essential reagents
                for label in ESSENTIALS:
                    row.append(comps[label][r, c])
                # Additives: presence flag and concentration
                for label in sorted(ADDITIVE_MAP):
                    lvl = comps[label][r, c]
                    row.extend([int(lvl > 0), lvl])
                # Curve metrics (保留 t90, I_max, tail_pct)
                row.extend([t90, I_max, tail_pct])
                records.append(row)

    # Build output column names (去掉 "Trial" 和 "valid")
    columns = (
        list(ESSENTIALS.values())
        + [
            f"{ADDITIVE_MAP[label]}_{suffix}"
            for label in sorted(ADDITIVE_MAP)
            for suffix in ("sw", "lvl")
        ]
        + ["t90", "I_max", "tail_pct"]
    )

    # Create DataFrame and write to CSV
    df_out = pd.DataFrame.from_records(records, columns=columns)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"✅ Exported {OUT_PATH} with {len(records)} rows")



if __name__ == "__main__":
    main()
