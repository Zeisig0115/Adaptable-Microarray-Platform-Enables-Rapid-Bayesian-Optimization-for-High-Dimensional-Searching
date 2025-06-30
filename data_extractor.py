from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.io

BASE_DIR = Path("./data")

EXCEL_NAME = "3 dimensional data concentrations.xlsx"
MAT_SUBDIR = ""
OUT_NAME = "bofire_ready.csv"

EXCEL_PATH = BASE_DIR / EXCEL_NAME
MAT_DIR = BASE_DIR / MAT_SUBDIR
OUT_PATH = BASE_DIR / OUT_NAME

SMOOTH_WIN = 5

# Map Excel labels to BoFire column names
ESSENTIALS = {"TMB": "e1", "HRP": "e2", "H2O2": "e3"}
ADDITIVE_MAP = {f"A{i:02d}": f"a{i:02d}" for i in range(1, 23)}
ALL_COMPONENTS = {**ESSENTIALS, **ADDITIVE_MAP}


def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    return np.convolve(y, np.ones(window) / window, mode="same")


def metrics_from_curve(y: np.ndarray, window: int = 5):
    y_s = smooth_curve(y, window)
    ymax = float(y_s.max()) if y_s.size > 0 else 0.0
    t90 = int(np.argmax(y_s >= 0.9 * ymax)) if ymax > 0 else len(y_s) - 1
    tail = float(y[-5:].mean() / ymax * 100.0) if ymax > 0 else 0.0
    return t90, ymax, tail


def collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    rows = []
    r = label_row
    while r < len(df) and len(rows) < n_rows:
        first = str(df.iat[r, 0]).upper().replace(" ", "")
        if r != label_row and first in ALL_COMPONENTS:
            break
        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(float, copy=True))
        r += 1
    while len(rows) < n_rows:
        rows.append(rows[-1].copy() if rows else np.zeros(24))
    return np.vstack(rows[:n_rows])


def parse_sheet(df: pd.DataFrame) -> dict:
    comps = {}
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")
    for label in ALL_COMPONENTS:
        idx = df.index[col0 == label]
        if not idx.empty:
            comps[label] = collect_block(df, int(idx[0]))
        else:
            comps[label] = np.zeros((16, 24), dtype=float)
    return comps


def load_curve_cube(mat_path: Path) -> np.ndarray:
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return np.stack([mat for mat in data["Output_c"]], axis=0)


def main():
    excel_file = EXCEL_PATH
    if not excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    xls = pd.ExcelFile(excel_file)
    records = []

    for sheet in xls.sheet_names:
        m = re.match(r"Trial\s*(\d+)", sheet, flags=re.I)
        if not m:
            print(f"⚠  Skipping sheet {sheet}")
            continue
        trial = int(m.group(1))
        mat_path = MAT_DIR / f"Trial {trial}.mat"
        if not mat_path.exists():
            print(f"⚠  Missing {mat_path}, skipping Trial {trial}")
            continue

        df = pd.read_excel(xls, sheet_name=sheet, header=None)
        comps = parse_sheet(df)
        cube = load_curve_cube(mat_path)

        for r in range(16):
            for c in range(24):
                t90, imax, tail = metrics_from_curve(cube[:, r, c], SMOOTH_WIN)

                row = [trial]
                for label in ESSENTIALS:
                    row.append(comps[label][r, c])
                for label in sorted(ADDITIVE_MAP):
                    lvl = comps[label][r, c]
                    row.extend([int(lvl > 0), lvl])
                row.extend([t90, imax, tail, True])
                records.append(row)

    columns = (
        ["Trial"]
        + list(ESSENTIALS.values())
        + [f"{ADDITIVE_MAP[l]}_{suffix}"
           for l in sorted(ADDITIVE_MAP) for suffix in ("sw", "lvl")]
        + ["t90", "I_max", "tail_pct", "valid"]
    )

    pd.DataFrame.from_records(records, columns=columns) \
      .to_csv(OUT_PATH, index=False)
    print(f"✅  Exported {OUT_PATH} with {len(records)} rows")


if __name__ == "__main__":
    main()
