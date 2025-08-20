#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_pipeline_from_t_and_images.py

要点：
  - 每个 sheet 强制重新提取，覆盖保存 ./data/<sheet> b.npy
  - 只用 ./data/<sheet> t.npy（必须存在）
  - 处理前对每个 sheet 交互输入 ROI 半径
  - 不保存 sd.npy / meta.json（BlueSD 仅内存用于统计）
  - data.csv: sheet, well="[r,c]", 25D 浓度, auc_mean, roi_sd_q25, roi_sd_q50, roi_sd_q75, n_repeats, auc_mean_std
"""

import os, glob, math, re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ============== 配置 ==============
DATA_DIR          = Path("./data")
IMG_BASE_DIR      = Path("./img")
EXCEL_NAME        = "3 dimensional data concentrations.xlsx"

FILE_GLOB         = "IMG_*.JPG"
NUM_REFS          = 1       # 标定参考帧个数
NUM_BG            = 1       # 每参考帧背景点个数
NUM_VIS           = 4       # ROI 预览帧数

DIVIDE_BY_TMAX    = True    # AUC 归一化：True→/max(t)；False→/(t[-1]-t[0])
ROWCOL_ONE_BASED  = False   # True → well 输出 1-based；False → 0-based

# 3 + 22 组件（顺序固定）
ESSENTIALS = {"TMB": "tmb", "HRP": "hrp", "H2O2": "h2o2"}
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PVA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PAA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]
ADDITIVE_KEYS_UPPER = [lab.upper().replace(" ", "") for lab in ADDITIVE_LABELS]
# =================================

# ---------- 小工具 ----------
def sanitize(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).lower().strip("_")

def numeric_key_from_filename(path: str) -> int:
    name = os.path.basename(path)
    m = re.findall(r'(\d+)', name)
    return int(m[-1]) if m else 0

def safe_crop(img, cx, cy, r):
    h, w = img.shape[:2]
    x1 = max(int(round(cx)) - r, 0)
    y1 = max(int(round(cy)) - r, 0)
    x2 = min(int(round(cx)) + r + 1, w)
    y2 = min(int(round(cy)) + r + 1, h)
    patch = img[y1:y2, x1:x2]
    rh = y2 - y1; rw = x2 - x1
    yy, xx = np.ogrid[:rh, :rw]
    cy0 = min(r, rh - 1); cx0 = min(r, rw - 1)
    inds = np.where((xx - cx0)**2 + (yy - cy0)**2 <= r*r)
    return patch, inds

# ---------- 交互标定 + 提取 BlueOnly & BlueSD（仅保存 BlueOnly） ----------
def blue_and_sd_from_frames(
    frames: List[str],
    spot_radius: int,
    num_refs: int = 2,
    num_bg: int = 1,
    num_vis: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      BlueOnly: (T,16,24)
      BlueSD:   (T,16,24)   ← ROI 内像素蓝度（背景扣除后）的标准差（仅内存使用）
    """
    T = len(frames)
    # --- 标定 ---
    ref_idx = np.linspace(0, T - 1, num_refs, dtype=int).tolist()
    gridXY, bgXY_list = [], []
    for ridx in ref_idx:
        img = cv2.imread(frames[ridx])
        if img is None:
            raise RuntimeError(f'Failed to read image: {frames[ridx]}')
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Ref frame {ridx} / file: {os.path.basename(frames[ridx])}\n'
                  f'Click 4 corners (TL, TR, BR, BL) + {num_bg} bg pts')
        pts = plt.ginput(4 + num_bg, timeout=0)
        plt.close()
        if len(pts) != 4 + num_bg:
            raise RuntimeError('Click count mismatch. Need 4 corners + NUM_BG background points.')
        corners = np.array(pts[:4], np.float32)
        bgXY_list.append([tuple(p) for p in pts[4:]])
        dst = np.array([[0, 0], [23, 0], [23, 15], [0, 15]], np.float32)
        H   = cv2.getPerspectiveTransform(dst, corners)
        U, V = np.meshgrid(np.arange(24), np.arange(16))
        uv   = np.stack([U.ravel(), V.ravel()], -1).astype(np.float32)
        mp   = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
        Xk, Yk = mp[:, 0].reshape(16, 24), mp[:, 1].reshape(16, 24)
        gridXY.append((Xk, Yk))
    print('Calibration done.')

    # --- 预览 ROI 布局 ---
    vis_idx = np.linspace(0, T - 1, min(num_vis, T), dtype=int)
    cols = int(math.ceil(math.sqrt(len(vis_idx))))
    rows = int(math.ceil(len(vis_idx) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axs = np.atleast_1d(axs).flatten()
    for ax, tidx in zip(axs, vis_idx):
        img = cv2.imread(frames[tidx])
        if img is None:
            ax.set_title(f'Failed to read frame {tidx}'); ax.axis('off'); continue
        ref_id = int(np.argmin(np.abs(np.array(ref_idx) - tidx)))
        Xk, Yk = gridXY[ref_id]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {tidx+1} → Ref #{ref_id}')
        ax.axis('off')
        for i in range(16):
            for j in range(24):
                circ = plt.Circle((Xk[i, j], Yk[i, j]), spot_radius,
                                  edgecolor='r', facecolor='none', linewidth=0.7)
                ax.add_patch(circ)
    for ax in axs[len(vis_idx):]:
        ax.axis('off')
    plt.tight_layout(); plt.show()

    # --- 提取 ---
    BlueOnly = np.zeros((T, 16, 24), np.float32)
    BlueSD   = np.zeros((T, 16, 24), np.float32)
    for t_idx, fname in enumerate(frames):
        img = cv2.imread(fname)
        if img is None:
            raise RuntimeError(f'Failed to read image: {fname}')
        ref_id = int(np.argmin(np.abs(np.array(ref_idx) - t_idx)))
        Xk, Yk = gridXY[ref_id]
        bg_pts = bgXY_list[ref_id]

        # 背景基线（蓝度）
        blue_base_vals = []
        for cx, cy in bg_pts:
            patch, inds = safe_crop(img, cx, cy, spot_radius)
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
            b = lab[..., 2] - 128.0
            blue_base_vals.append(np.mean((-b)[inds]))
        b_bg = float(np.mean(blue_base_vals)) if blue_base_vals else 0.0

        for i in range(16):
            for j in range(24):
                cx, cy = Xk[i, j], Yk[i, j]
                patch, inds = safe_crop(img, cx, cy, spot_radius)
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                b = lab[..., 2] - 128.0
                vals = (-b)[inds] - b_bg      # 像素级蓝度（背景扣除）
                mu = float(np.mean(vals))
                sd = float(np.std(vals))      # 总体标准差（ddof=0）
                BlueOnly[t_idx, i, j] = max(0.0, mu)
                BlueSD[t_idx, i, j]   = sd

        if (t_idx + 1) % 10 == 0 or t_idx == T - 1:
            print(f'Processed {t_idx + 1}/{T} frames')
    print('BlueOnly & BlueSD extraction finished.')
    return BlueOnly, BlueSD

# ---------- Excel 解析（16×24×(3+22)） ----------
def collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    rows = []
    r = label_row
    ALL = set(list(ESSENTIALS.keys()) + ADDITIVE_KEYS_UPPER)
    while r < len(df) and len(rows) < n_rows:
        first_cell_norm = str(df.iat[r, 0]).upper().replace(" ", "")
        if r != label_row and first_cell_norm in ALL:
            break
        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(dtype=float, copy=True))
        r += 1
    if not rows:
        rows = [np.zeros(24, dtype=float) for _ in range(n_rows)]
    while len(rows) < n_rows:
        rows.append(rows[-1].copy())
    return np.vstack(rows[:n_rows])

def parse_sheet(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    comps: Dict[str, np.ndarray] = {}
    ALL = set(list(ESSENTIALS.keys()) + ADDITIVE_KEYS_UPPER)
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")
    for label in ALL:
        matches = df.index[col0 == label]
        if not matches.empty:
            comps[label] = collect_block(df, matches[0])
        else:
            comps[label] = np.zeros((16, 24), dtype=float)
    return comps

# ---------- AUC/t ----------
def auc_mean_general(xx: np.ndarray, yy: np.ndarray, divide_by_tmax: bool=True) -> float:
    if xx is None or yy is None or xx.size != yy.size or xx.size < 2:
        return np.nan
    m = np.isfinite(xx) & np.isfinite(yy)
    if not np.any(m):
        return np.nan
    xs = xx[m].astype(float); ys = yy[m].astype(float)
    order = np.argsort(xs); xs = xs[order]; ys = ys[order]
    area = float(np.trapezoid(ys, xs))
    denom = float(np.max(xs) if divide_by_tmax else (xs[-1] - xs[0]))
    return np.nan if denom <= 0 else area / denom

# ---------- 输入半径 ----------
def ask_radius_for_sheet(sheet: str) -> int:
    while True:
        s = input(f"请输入 '{sheet}' 的 ROI 半径（像素，正整数，例如 30）：").strip()
        try:
            r = int(s)
            if r > 0:
                return r
        except Exception:
            pass
        print("无效输入，请输入正整数。")

# ---------- 主流程 ----------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xls_path = DATA_DIR / EXCEL_NAME
    if not xls_path.exists():
        raise FileNotFoundError(f"Excel not found: {xls_path}")
    xls = pd.ExcelFile(xls_path)

    feature_cols = [ESSENTIALS[k] for k in ["TMB", "HRP", "H2O2"]]
    additive_cols = [sanitize(lab) for lab in ADDITIVE_LABELS]
    # 列顺序：sheet, well, 25D, auc_mean, roi_sd_q25, roi_sd_q50, roi_sd_q75, n_repeats, auc_mean_std
    columns = ["sheet", "well"] + feature_cols + additive_cols + \
              ["auc_mean", "roi_sd_q25", "roi_sd_q50", "roi_sd_q75", "n_repeats", "auc_mean_std"]

    records = []

    for sheet_name in xls.sheet_names:
        if sheet_name.strip().lower() == "time points":
            continue
        exp = sheet_name.strip()

        # 时间轴：必须存在
        t_path = DATA_DIR / f"{exp} t.npy"
        if not t_path.exists():
            print(f"⚠ 缺少时间轴：{t_path}，跳过 {exp}")
            continue
        t_seconds = np.load(t_path)
        if t_seconds.ndim != 1 or t_seconds.size < 2:
            print(f"⚠ 非法时间轴形状：{t_seconds.shape}，跳过 {exp}")
            continue

        # 半径输入
        spot_radius = ask_radius_for_sheet(exp)

        # 图片
        img_dir = IMG_BASE_DIR / exp
        frames = sorted(glob.glob(str(img_dir / FILE_GLOB)), key=numeric_key_from_filename)
        if not frames:
            print(f"⚠ 未找到图片：{img_dir}/{FILE_GLOB}，跳过 {exp}")
            continue

        # 强制重新计算（不复用、不跳过）
        blue_only, blue_sd = blue_and_sd_from_frames(
            frames=frames,
            spot_radius=spot_radius,
            num_refs=NUM_REFS,
            num_bg=NUM_BG,
            num_vis=NUM_VIS
        )

        # 对齐长度（以时间轴为准）
        T = min(len(t_seconds), blue_only.shape[0], blue_sd.shape[0])
        if T < len(t_seconds) or T < blue_only.shape[0] or T < blue_sd.shape[0]:
            print(f"ℹ 对齐长度到 T={T}（按时间轴截断）")
        t_seconds = t_seconds[:T]
        blue_only = blue_only[:T]
        blue_sd   = blue_sd[:T]

        # 始终覆盖保存 BlueOnly（b.npy）
        b_path = DATA_DIR / f"{exp} b.npy"
        np.save(b_path, blue_only)
        print(f"✓ Overwritten {b_path.name} with shape {blue_only.shape}")

        # 解析该 sheet 的配方矩阵
        df_comp = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df_comp)

        # 逐井记录
        for r in range(16):
            for c in range(24):
                r_out = (r + 1) if ROWCOL_ONE_BASED else r
                c_out = (c + 1) if ROWCOL_ONE_BASED else c
                well_str = f"[{r_out},{c_out}]"

                vec_3  = [comps["TMB"][r, c], comps["HRP"][r, c], comps["H2O2"][r, c]]
                vec_22 = [comps[k][r, c] for k in ADDITIVE_KEYS_UPPER]

                y = blue_only[:, r, c]
                sd_curve = blue_sd[:, r, c]

                auc_val = float(auc_mean_general(t_seconds, y, divide_by_tmax=DIVIDE_BY_TMAX))
                if np.isfinite(sd_curve).any():
                    q25, q50, q75 = np.nanpercentile(sd_curve, [25, 50, 75])
                    q25 = float(q25); q50 = float(q50); q75 = float(q75)
                else:
                    q25 = q50 = q75 = float('nan')

                records.append([exp, well_str] + vec_3 + vec_22 +
                               [auc_val, q25, q50, q75, np.nan, np.nan])

    if not records:
        print("没有可写入的记录。请检查 Excel、t.npy 或图片目录。")
        return

    # 合并并做重复统计（按 25D 浓度向量分组，与 well 无关）
    df = pd.DataFrame.from_records(records, columns=columns)

    key_cols = [ESSENTIALS[k] for k in ["TMB", "HRP", "H2O2"]] + [sanitize(lab) for lab in ADDITIVE_LABELS]
    key_array = np.round(df[key_cols].to_numpy(dtype=float), 12)
    keys = [tuple(row.tolist()) for row in key_array]

    from collections import defaultdict
    groups = defaultdict(list)
    for idx, k in enumerate(keys):
        groups[k].append(idx)

    n_repeats = np.zeros(len(df), dtype=int)
    auc_std   = np.full(len(df), np.nan, dtype=float)
    for _, idxs in groups.items():
        n = len(idxs)
        n_repeats[idxs] = n
        if n >= 2:
            vals = df.iloc[idxs]["auc_mean"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                auc_std[idxs] = float(np.std(vals, ddof=1))
    df["n_repeats"] = n_repeats
    df["auc_mean_std"] = auc_std

    out_path = DATA_DIR / "data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ 写入 {out_path}，共 {len(df)} 行；列：{', '.join(df.columns)}")


if __name__ == "__main__":
    main()
