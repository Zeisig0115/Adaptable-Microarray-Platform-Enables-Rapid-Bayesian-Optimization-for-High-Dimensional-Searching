import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# -----------------------
# --- Configuration ----
# -----------------------
BASE_DIR = Path("./data")
DEFAULT_EXCEL = BASE_DIR / "3 dimensional data concentrations.xlsx"
DEFAULT_MAT_DIR = BASE_DIR
DEFAULT_OUT = BASE_DIR / "sequence_data.csv"
SMOOTH_WIN = 5  # 默认平滑窗口大小

# 必要试剂标签
ESSENTIAL_LABELS = ["TMB", "HRP", "H2O2"]
# 添加剂标签列表
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PAA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PVA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]

def sanitize(label: str) -> str:
    """
    将原始标签转成干净的前缀（用于列名，如果需要）：
    - 非字母数字替换为下划线
    - 转小写
    - 去除首尾下划线
    """
    cleaned = re.sub(r'[^0-9A-Za-z]+', '_', label)
    return cleaned.lower().strip('_')

# 构造添加剂的标签映射：如 "PEG20K" -> "peg20k"
ADDITIVE_MAP = {
    label.upper().replace(" ", ""): sanitize(label)
    for label in ADDITIVE_LABELS
}
# 合并所有要在 Excel 中搜索的成分标签
ALL_COMPONENTS = {lbl: lbl for lbl in ESSENTIAL_LABELS}
ALL_COMPONENTS.update(ADDITIVE_MAP)

# ---------- 平滑与指标计算 ----------
def smooth_curve(y: np.ndarray, window: int = SMOOTH_WIN) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

def metrics_from_curve(y: np.ndarray, window: int = SMOOTH_WIN):
    """
    计算 t90, I_max, tail_pct
    """
    y_s = smooth_curve(y, window)
    if y_s.size == 0:
        return len(y) - 1, 0.0, 0.0
    ymax = float(y_s.max())
    t90 = int(np.argmax(y_s >= 0.9 * ymax))
    tail_pct = float(y[-5:].mean() / ymax * 100.0) if ymax > 0 else 0.0
    return t90, ymax, tail_pct

# --------- Excel 解析函数 ---------
def collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    """
    从 label_row 开始，读取最多 n_rows 行，每行 24 个数值。
    碰到下一个成分标签则停止；不足时用最后一行或零填充。
    """
    rows = []
    r = label_row
    while r < len(df) and len(rows) < n_rows:
        first = str(df.iat[r, 0]).upper().replace(" ", "")
        if r != label_row and first in ALL_COMPONENTS:
            break
        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(dtype=float, copy=True))
        r += 1
    # pad
    while len(rows) < n_rows:
        if rows:
            rows.append(rows[-1].copy())
        else:
            rows.append(np.zeros(24))
    return np.vstack(rows[:n_rows])

def parse_sheet(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    解析整个 sheet，返回每个成分标签对应的 16×24 浓度矩阵，
    如果没找到则用零矩阵填充。
    """
    comps: dict[str, np.ndarray] = {}
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")
    for label in ALL_COMPONENTS:
        matches = df.index[col0 == label]
        if not matches.empty:
            comps[label] = collect_block(df, matches[0])
        else:
            comps[label] = np.zeros((16, 24), dtype=float)
    return comps

# -------- .mat 文件加载 --------
def load_curve_cube(mat_path: Path) -> np.ndarray:
    """
    把 MATLAB .mat 中的 Output_c 堆叠成 (T × 16 × 24) 数组
    """
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return np.stack(list(data["Output_c"]), axis=0)

# -------- 主流程 --------
def main():
    ap = argparse.ArgumentParser(
        description="Export Trial curves: Trial, TMB/HRP/H2O2, I0–I{T-1}, t90, I_max, tail_pct"
    )
    ap.add_argument(
        "--excel", default=str(DEFAULT_EXCEL),
        help="Path to the Excel file"
    )
    ap.add_argument(
        "--mat-dir", default=str(DEFAULT_MAT_DIR),
        help="Directory containing Trial .mat files"
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="Output CSV file path"
    )
    ap.add_argument(
        "--smooth-window", type=int, default=SMOOTH_WIN,
        help="Window size for smoothing"
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

    # 先打开 Excel，不一次性读所有 sheet，改为逐个读取
    xls = pd.ExcelFile(excel_path)
    records = []
    timepoints = None

    for sheet_name in xls.sheet_names:
        m = re.match(r"Trial\s*(\d+)", sheet_name, flags=re.I)
        if not m:
            print(f"[WARN] Skipping sheet {sheet_name}", file=sys.stderr)
            continue
        trial = int(m.group(1))

        mat_path = mat_dir / f"Trial {trial}.mat"
        if not mat_path.exists():
            print(f"[WARN] Missing .mat file: {mat_path}", file=sys.stderr)
            continue

        # 解析这一 sheet
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df)

        # 加载曲线立方体
        cube = load_curve_cube(mat_path)
        if timepoints is None:
            timepoints = cube.shape[0]

        # 对每个 (r,c) 计算指标并保存整条曲线
        for r in range(16):
            for c in range(24):
                curve = cube[:, r, c]
                t90, I_max, tail_pct = metrics_from_curve(curve, window=args.smooth_window)
                records.append(
                    [trial,
                     comps["TMB"][r, c], comps["HRP"][r, c], comps["H2O2"][r, c]]
                    + curve.tolist()
                    + [t90, I_max, tail_pct]
                )

    # 构建列名：Trial, TMB, HRP, H2O2, I0..I{T-1}, t90, I_max, tail_pct
    cols = (
        ["Trial", "TMB", "HRP", "H2O2"]
        + [f"I{i}" for i in range(timepoints or 0)]
        + ["t90", "I_max", "tail_pct"]
    )
    pd.DataFrame.from_records(records, columns=cols).to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} with {len(records)} curves")


if __name__ == "__main__":
    main()
