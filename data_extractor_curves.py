import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

BASE_DIR = Path("./data")
DEFAULT_EXCEL = BASE_DIR / "3 dimensional data concentrations.xlsx"
DEFAULT_MAT_DIR = BASE_DIR
DEFAULT_OUT = BASE_DIR / "sequence_data.csv"

# ---------- 1. Moving Average ----------
def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply a moving average to 1D array, keeping length fixed.
    """
    return np.convolve(y, np.ones(window) / window, mode="same")

# ---------- 2. Compute Three Metrics ----------
def three_metrics(y: np.ndarray, window: int = 5):
    """
    Compute (t90, I_max, tail_pct) from a curve.
    """
    y_s = smooth_curve(y, window)
    max_val = float(y_s.max()) if y_s.size > 0 else 0.0
    t90 = int(np.argmax(y_s >= 0.9 * max_val)) if max_val > 0 else len(y_s) - 1
    tail_pct = float(y[-5:].mean() / max_val * 100.0) if max_val > 0 else 0.0
    return t90, max_val, tail_pct

# -------- Excel Parsing Helpers --------
def _collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    """
    Extract up to n_rows of data from a label row in a sheet.
    """
    rows = []
    for r in range(label_row, len(df)):
        # stop at next label
        first = str(df.iat[r, 0]).upper().replace(" ", "")
        if first in ("TMB", "HRP", "H2O2") and r != label_row:
            break
        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(float))
        if len(rows) == n_rows:
            break
    # pad with last row if needed
    while len(rows) < n_rows:
        rows.append(rows[-1].copy())
    return np.vstack(rows[:n_rows])

def parse_sheet(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Parse concentrations of TMB, HRP, H2O2 from a sheet.
    """
    comps: dict[str, np.ndarray] = {}
    for label in ("TMB", "HRP", "H2O2"):
        idx = df.index[df[0].astype(str).str.upper() == label]
        if idx.empty:
            raise ValueError(f"Label {label!r} not found in sheet")
        comps[label] = _collect_block(df, idx[0])
    return comps

# -------- .mat Loader --------
def load_cube(mat_path: Path) -> np.ndarray:
    """
    Load Output_c from a .mat file as a (60,16,24) array.
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return np.stack([m for m in data["Output_c"]], axis=0)

# -------- Main Routine --------
def main():
    """
    Export Trial curves to CSV with metrics.
    """
    ap = argparse.ArgumentParser(
        description="Export Trial curves: Trial, TMB, HRP, H2O2, I0–I59, t90, I_max, tail_pct"
    )
    ap.add_argument(
        "--excel",
        default=str(DEFAULT_EXCEL),
        help="Path to the Excel file (default: ./data/3 dimensional data concentrations.xlsx)"
    )
    ap.add_argument(
        "--mat-dir",
        default=str(DEFAULT_MAT_DIR),
        help="Directory containing Trial .mat files (default: ./data/)"
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output CSV file path (default: ./data/all_curves.csv)"
    )
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for smoothing (default: 5)"
    )
    args = ap.parse_args()

    excel_path = Path(args.excel)
    mat_dir = Path(args.mat_dir)
    out_path = Path(args.out)

    if not excel_path.exists():
        print(f"[ERROR] Excel file not found: {excel_path}", file=sys.stderr)
        sys.exit(1)
    if not mat_dir.exists():
        print(f"[ERROR] .mat directory not found: {mat_dir}", file=sys.stderr)
        sys.exit(1)

    sheets = pd.read_excel(excel_path, sheet_name=None, header=None)
    records: list[list[float]] = []

    for trial in (1, 2, 3):
        # 找到对应的 Trial sheet
        sheet_name = next(
            (s for s in sheets if re.sub(r"\s+", "", s).lower() == f"trial{trial}"),
            None
        )
        if sheet_name is None:
            print(f"[WARN] Sheet for Trial {trial} not found", file=sys.stderr)
            continue

        conc = parse_sheet(sheets[sheet_name])
        mat_path = mat_dir / f"Trial {trial}.mat"
        if not mat_path.exists():
            print(f"[WARN] Missing .mat file: {mat_path}", file=sys.stderr)
            continue

        cube = load_cube(mat_path)

        for r in range(16):
            for c in range(24):
                curve = cube[:, r, c]
                t90, imax, tail = three_metrics(curve, window=args.smooth_window)
                records.append(
                    [trial, conc["TMB"][r, c], conc["HRP"][r, c], conc["H2O2"][r, c]]
                    + curve.tolist()
                    + [t90, imax, tail]
                )

    cols = (
        ["Trial", "TMB", "HRP", "H2O2"]
        + [f"I{t}" for t in range(60)]
        + ["t90", "I_max", "tail_pct"]
    )
    pd.DataFrame.from_records(records, columns=cols) \
      .to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} with {len(records)} curves")

if __name__ == "__main__":
    main()
