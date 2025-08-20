
# -*- coding: utf-8 -*-
"""
Sheet-based extrapolation evaluation (print only, allows explicit test/train sheets).

Train on the first N_train unique `sheet` groups (by first appearance in the CSV)
and test on the last N_test groups by default; or explicitly specify which `sheet`s
are used for testing (and optionally training). Reuses the same training & metrics
pipeline as `fold_eval_learn_noise` from main.py.

Examples:
# (A) Explicitly choose test sheets (training = all other sheets)
python3 sheet_extrap_eval.py --csv data.csv --target auc_mean \
  --models saas_gp gp --test_sheets Sheet5,Sheet6

# (B) Explicit train + test sheets
python3 sheet_extrap_eval.py --csv data.csv --target auc_mean \
  --models saas_gp --train_sheets Sheet1,Sheet2,Sheet3,Sheet4 --test_sheets Sheet5,Sheet6

# (C) Fallback by counts (first 4 train, last 2 test)
python3 sheet_extrap_eval.py --csv data.csv --target auc_mean \
  --models saas_gp gp --n_train 4 --n_test 2
"""
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Reuse the fold-level evaluation (model fitting + metrics) from the existing codebase
from main import fold_eval_learn_noise

FEATURE_WHITELIST = [
    "tmb","hrp","h2o2","etoh","peg20k","dmso","pl127","bsa","pva","tw80","glycerol","tw20",
    "imidazole","tx100","edta","mgcl2","sucrose","cacl2","zn2","paa","mn2","peg200k","fe2",
    "peg5k","peg400"
]


def _load_numeric_with_sheet(csv_path: str, target_col: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load features + target similar to main.load_numeric_table_no_noise, but also returns the 'sheet' column.

    Returns:
        feat_cols: chosen feature names (case-insensitive whitelist, preserving original casing)
        X: float64 numpy array [n, d]
        y: float64 numpy array [n, 1]
        sheets: object numpy array [n] (labels)
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"未找到目标列 '{target_col}'；现有列：{list(df.columns)}")
    if "sheet" not in df.columns:
        raise ValueError("未找到列 'sheet'，无法做基于sheet的外推评估。")

    # case-insensitive feature mapping
    col_map = {c.lower(): c for c in df.columns if c != target_col and c != "sheet"}
    feat_cols = [col_map[c] for c in FEATURE_WHITELIST if c in col_map]
    if not feat_cols:
        raise ValueError("白名单特征在数据中一个都没找到；请检查列名或大小写。")

    # numeric coercion for features + target; keep original 'sheet' labels
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y_ser = pd.to_numeric(df[target_col], errors="coerce")
    sheet_ser = df["sheet"]

    # clean rows with inf/NaN in features or target, and require non-null sheet
    all_df = pd.concat([X_df, y_ser.rename(target_col), sheet_ser.rename("sheet")], axis=1)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + [target_col, "sheet"], how="any")

    X = all_df[feat_cols].to_numpy(dtype=float)
    y = all_df[target_col].to_numpy(dtype=float).reshape(-1, 1)
    sheets = all_df["sheet"].to_numpy()

    if X.shape[1] == 0:
        raise ValueError("白名单中的特征在数据里均为空或无法转为数值。")
    if X.shape[0] == 0:
        raise ValueError("清洗后没有有效样本；请检查数据是否含有 NaN/±inf 或目标/特征是否可转换为数值。")

    return feat_cols, X, y, sheets


def _parse_sheets_arg(arg_val: Optional[str]) -> Optional[List[str]]:
    if arg_val is None:
        return None
    items = [s.strip() for s in arg_val.split(",")]
    items = [s for s in items if s != ""]
    return items if len(items) > 0 else None


def sheet_extrapolation_eval(
    csv_path: str,
    target_col: str,
    models: List[str],
    seed: int = 42,
    warmup_steps: int = 128,
    num_samples: int = 64,
    thinning: int = 16,
    crps_T: int = 512,
    n_train: int = 4,
    n_test: int = 2,
    train_sheet_names: Optional[List[str]] = None,
    test_sheet_names: Optional[List[str]] = None,
    device: str = "auto"
) -> pd.DataFrame:
    """Evaluate extrapolation by `sheet`.

    If `test_sheet_names` is provided, test = rows where sheet ∈ test_sheet_names.
    If `train_sheet_names` is also provided, train = rows where sheet ∈ train_sheet_names.
    Otherwise, train = all rows where sheet ∉ test_sheet_names.

    If `test_sheet_names` is None, fall back to first/last split by counts.
    """
    # load data
    feat_cols, X_np, y_np, sheets_raw = _load_numeric_with_sheet(csv_path, target_col)

    # normalize to string for robust matching
    sheets = pd.Series(sheets_raw).astype(str).to_numpy()
    unique_sheets = pd.Index(sheets).unique().tolist()

    # choose train/test sheets
    if test_sheet_names is not None:
        # normalize input names to str
        test_sheet_names = [str(s) for s in test_sheet_names]
        missing = [s for s in test_sheet_names if s not in unique_sheets]
        if missing:
            raise ValueError(f"以下 test_sheets 不存在于数据中：{missing}。现有 sheet：{unique_sheets}")

        if train_sheet_names is not None:
            train_sheet_names = [str(s) for s in train_sheet_names]
            missing_tr = [s for s in train_sheet_names if s not in unique_sheets]
            if missing_tr:
                raise ValueError(f"以下 train_sheets 不存在于数据中：{missing_tr}。现有 sheet：{unique_sheets}")
            if set(train_sheet_names) & set(test_sheet_names):
                overlap = sorted(set(train_sheet_names) & set(test_sheet_names))
                raise ValueError(f"train_sheets 与 test_sheets 交集非空：{overlap}")
            train_sheets = train_sheet_names
        else:
            # default: train on all other sheets
            train_sheets = [s for s in unique_sheets if s not in set(test_sheet_names)]

        test_sheets = test_sheet_names
    else:
        # fallback to by-count split
        if len(unique_sheets) < (n_train + n_test):
            raise ValueError(
                f"唯一 sheet 数量为 {len(unique_sheets)}，不足以执行 n_train={n_train} & n_test={n_test} 的切分。"
                " 请检查数据或调整参数。"
            )
        train_sheets = unique_sheets[:n_train]
        test_sheets = unique_sheets[-n_test:]

    # boolean masks -> indices
    idx_train = np.where(pd.Series(sheets).isin(train_sheets).to_numpy())[0]
    idx_test = np.where(pd.Series(sheets).isin(test_sheets).to_numpy())[0]

    if len(idx_train) == 0 or len(idx_test) == 0:
        raise ValueError(f"切分后训练/测试样本为空：train={len(idx_train)}, test={len(idx_test)}")

    # run evaluation for each model using the shared fold evaluation
    records = []
    for m in models:
        res = fold_eval_learn_noise(
            X_all=X_np, y_all=y_np,
            idx_train=idx_train, idx_test=idx_test,
            warmup=warmup_steps, num_samples=num_samples, thinning=thinning,
            seed=seed, crps_T=crps_T, model_type=m,
            device=device,  # 先尝试带 device
        )
        records.append({
            "model": m,
            "train_sheets": ",".join(map(str, train_sheets)),
            "test_sheets": ",".join(map(str, test_sheets)),
            "n_train": int(len(idx_train)),
            "n_test": int(res.n_test),
            "RMSE": float(res.rmse),
            "R2": float(res.r2),
            "NLL": float(res.nll),
            "MSLL": float(res.msll),
            "CRPS": float(res.crps),
        })

    df_out = pd.DataFrame.from_records(records)
    # Sort by RMSE ascending as a default view
    df_out = df_out.sort_values(by="RMSE", ascending=True, ignore_index=True)
    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data.csv")
    parser.add_argument("--target", type=str, default="auc_mean")
    parser.add_argument("--models", type=str, nargs="+", default=["gp", "saasbo", "fullyb_gp", "saas_gp", "prf", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--warmup_steps", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--thinning", type=int, default=16)
    parser.add_argument("--crps_T", type=int, default=512)

    # Either explicit sheets or fallback by counts
    parser.add_argument("--test_sheets", type=str, default="pairwise 1, pairwise 2")
    parser.add_argument("--train_sheets", type=str, default="H2O2, HRP, TMB, single additive")
    parser.add_argument("--n_train", type=int, default=4, help="Used only if --test_sheets not provided")
    parser.add_argument("--n_test", type=int, default=2, help="Used only if --test_sheets not provided")
    args = parser.parse_args()

    test_sheets = _parse_sheets_arg(args.test_sheets)
    train_sheets = _parse_sheets_arg(args.train_sheets)

    print(train_sheets, test_sheets)

    df = sheet_extrapolation_eval(
        csv_path=args.csv,
        target_col=args.target,
        models=args.models,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        num_samples=args.num_samples,
        thinning=args.thinning,
        crps_T=args.crps_T,
        n_train=args.n_train,
        n_test=args.n_test,
        train_sheet_names=train_sheets,
        test_sheet_names=test_sheets,
        device=args.device
    )

    # Print only (no saving)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
