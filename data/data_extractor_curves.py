from pathlib import Path
import re

import numpy as np
import pandas as pd
import scipy.io

# -----------------------
# --- Configuration ----
# -----------------------

BASE_DIR = Path("./data")
EXCEL_NAME = "3 dimensional data concentrations.xlsx"
MAT_SUBDIR = ""
OUT_NAME = "curves.csv"             # ★ 新文件名

EXCEL_PATH = BASE_DIR / EXCEL_NAME
MAT_DIR    = BASE_DIR / MAT_SUBDIR
OUT_PATH   = BASE_DIR / OUT_NAME

SMOOTH_WIN = 5
TS_LEN = 60                                           # ★ 提取 I_0:59，共 60 点

ESSENTIALS = {"TMB": "tmb", "HRP": "hrp", "H2O2": "h2o2"}

ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PAA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PVA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]

def sanitize(label: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z]+', '_', label)
    return cleaned.lower().strip('_')

ADDITIVE_MAP = {lbl.upper().replace(" ", ""): sanitize(lbl)
                for lbl in ADDITIVE_LABELS}
ALL_COMPONENTS = {**ESSENTIALS, **ADDITIVE_MAP}

# ---------- 工具函数 ----------
def smooth_curve(y, window=SMOOTH_WIN):
    return np.convolve(y, np.ones(window)/window, mode="same")

def metrics_from_curve(y, window=SMOOTH_WIN):
    y_s = smooth_curve(y, window)
    if y_s.size == 0:
        return len(y)-1, 0.0, 0.0
    ymax = float(y_s.max())
    t90 = int(np.argmax(y_s >= 0.9*ymax))
    tail_pct = float(y[-5:].mean()/ymax*100) if ymax else 0.0
    return t90, ymax, tail_pct

def collect_block(df, label_row, n_rows=16):
    rows, r = [], label_row
    while r < len(df) and len(rows) < n_rows:
        first = str(df.iat[r,0]).upper().replace(" ", "")
        if r != label_row and first in ALL_COMPONENTS:
            break
        vals = pd.to_numeric(df.iloc[r,1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(float, copy=True))
        r += 1
    while len(rows) < n_rows:
        rows.append(rows[-1].copy() if rows else np.zeros(24))
    return np.vstack(rows[:n_rows])

def parse_sheet(df):
    comps, col0 = {}, df.iloc[:,0].astype(str).str.upper().str.replace(" ", "")
    for lbl in ALL_COMPONENTS:
        idx = df.index[col0==lbl]
        comps[lbl] = collect_block(df, idx[0]) if not idx.empty \
                     else np.zeros((16,24))
    return comps

def load_curve_cube(mat_path: Path):
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return np.stack(list(data["Output_c"]), axis=0)        # (time, 16, 24)

# -----------------------
# --- 主流程 ------------
# -----------------------
def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(EXCEL_PATH)

    xls = pd.ExcelFile(EXCEL_PATH)
    records = []

    for sheet_name in xls.sheet_names:
        m = re.match(r"Trial\s*(\d+)", sheet_name, flags=re.I)
        if not m:
            print(f"⚠ Skipping sheet {sheet_name}")
            continue

        trial_no = int(m.group(1))
        mat_path = MAT_DIR / f"Trial {trial_no}.mat"
        if not mat_path.exists():
            print(f"⚠ Missing {mat_path}, skipping Trial {trial_no}")
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df)
        cube = load_curve_cube(mat_path)                   # (T,16,24)

        for r in range(16):
            for c in range(24):
                curve = cube[:, r, c]
                t90, I_max, tail_pct = metrics_from_curve(curve)

                row = []
                # 1) essentials (3)
                for lbl in ESSENTIALS:
                    row.append(comps[lbl][r, c])
                # 2) additive levels (22)
                for lbl in sorted(ADDITIVE_MAP):
                    row.append(comps[lbl][r, c])
                # 3) raw intensity I_0..I_59 (60)
                if len(curve) >= TS_LEN:
                    row.extend(curve[:TS_LEN])
                else:                                      # ★ 不足用 NaN 填
                    row.extend(np.concatenate([curve, np.full(TS_LEN-len(curve), np.nan)]))
                # 4) outputs (3)
                row.extend([t90, I_max, tail_pct])
                records.append(row)

    # ---------- 列名 ----------
    columns = (
        list(ESSENTIALS.values()) +
        [ADDITIVE_MAP[lbl] for lbl in sorted(ADDITIVE_MAP)] +
        [f"I_{i}" for i in range(TS_LEN)] +               # ★ 60 个 I_t
        ["t90", "I_max", "tail_pct"]
    )
    assert len(columns) == 25 + TS_LEN + 3 == 88

    df_out = pd.DataFrame.from_records(records, columns=columns)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"✅ Exported {OUT_PATH} | shape = {df_out.shape} (25+60+3 columns)")

if __name__ == "__main__":
    main()
