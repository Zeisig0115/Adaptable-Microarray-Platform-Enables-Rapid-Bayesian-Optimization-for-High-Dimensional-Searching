import argparse
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Hyperparameters & Reproducibility ----------
SEED = 42
KF_SPLITS = 5
USE_GROUPS = False

# set seeds for numpy and random
np.random.seed(SEED)
random.seed(SEED)

# if torch is available, set its seed too
try:
    import torch

    torch.manual_seed(SEED)
except ImportError:
    print("[Info] torch not available, skipping torch seed", file=sys.stderr)


def get_args():
    """
    Parse command-line arguments for data path, grouping, and output.

    Returns:
        Namespace with attributes:
            data: path to input CSV
            group_col: column name for grouping folds
            model_out: filename for saving the final model
    """
    p = argparse.ArgumentParser("5-fold CV for CatBoost surrogate")
    p.add_argument("-d", "--data", default="all_curves.csv",
                   help="CSV file (with TMB, HRP, H2O2, t90, I_max, tail_pct)")
    p.add_argument("-g", "--group_col", default="Trial",
                   help="Column name for grouping when USE_GROUPS=True")
    p.add_argument("-o", "--model_out", default="catboost_surrogate_full.joblib",
                   help="Path to save the final model fitted on all data")
    return p.parse_args()


def load_dataset(path: str, group_col: str):
    """
    Load features, targets, and optional group labels from CSV.

    Args:
        path: path to CSV file.
        group_col: column name to use for groups if enabled.

    Returns:
        X: numpy array of shape (n_samples, 3)
        y: numpy array of shape (n_samples, 3)
        groups: array of group labels or None
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")
    df = pd.read_csv(path)
    X = df[["TMB", "HRP", "H2O2"]].values
    y = df[["t90", "I_max", "tail_pct"]].values
    groups = df[group_col].values if USE_GROUPS else None
    return X, y, groups


def build_model():
    """
    Construct a Pipeline with StandardScaler and CatBoostRegressor.

    CatBoostRegressor uses MultiRMSE for multi-output regression.

    Returns:
        sklearn Pipeline object
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("cat", CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiRMSE",
            random_seed=SEED,
            thread_count=-1,
            verbose=0
        ))
    ])


def cross_validate(X: np.ndarray, y: np.ndarray, groups=None):
    """
    Perform K-fold (or GroupKFold) cross-validation and print metrics.

    Args:
        X: feature matrix
        y: target matrix
        groups: group labels for GroupKFold or None

    Returns:
        List of dicts containing per-fold statistics
    """
    cv = GroupKFold(n_splits=KF_SPLITS) if USE_GROUPS else \
        KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)
    fold_stats = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        model = build_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        stats = {
            "fold": fold,
            "t90_r2": r2_score(y[te][:, 0], pred[:, 0]),
            "t90_mae": mean_absolute_error(y[te][:, 0], pred[:, 0]),
            "imax_r2": r2_score(y[te][:, 1], pred[:, 1]),
            "imax_mae": mean_absolute_error(y[te][:, 1], pred[:, 1]),
            "tail_r2": r2_score(y[te][:, 2], pred[:, 2]),
            "tail_mae": mean_absolute_error(y[te][:, 2], pred[:, 2]),
        }
        fold_stats.append(stats)
        print(f"[Fold {fold}] "
              f"t90_R2={stats['t90_r2']:.3f}  "
              f"Imax_R2={stats['imax_r2']:.3f}  "
              f"Tail_R2={stats['tail_r2']:.3f}")

    # summarize mean ± std for each metric
    def mean_std(lst, key):
        vals = np.array([d[key] for d in lst])
        return vals.mean(), vals.std(ddof=1)

    print("\n===== 5-fold CV Results (mean±sigma) =====")
    for tgt, lbl in zip(["t90", "imax", "tail"], ["t90", "I_max", "tail_pct"]):
        r2_m, r2_s = mean_std(fold_stats, f"{tgt}_r2")
        mae_m, mae_s = mean_std(fold_stats, f"{tgt}_mae")
        print(f"{lbl:<8}  R²={r2_m:.3f}±{r2_s:.3f}   "
              f"MAE={mae_m:.3f}±{mae_s:.3f}")

    return fold_stats


def main():
    """
    Main entry point:
    - Parse CLI arguments
    - Load dataset
    - Run cross-validation
    - Optionally train and save final model on full data
    """
    args = get_args()
    X, y, groups = load_dataset(args.data, args.group_col)
    stats = cross_validate(X, y, groups)

    # train on full data and save model
    model = build_model()
    model.fit(X, y)
    joblib.dump(model, args.model_out)
    print(f"✅ Trained full model and saved to {args.model_out}")


if __name__ == "__main__":
    main()
