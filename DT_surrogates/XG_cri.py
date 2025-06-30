import argparse
import os
import random

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor  # pip install xgboost

# ---------- Hyperparameters & Reproducibility ----------
SEED       = 42
KF_SPLITS  = 5
USE_GROUPS = False     # set True to group folds by Trial
BATCH_SIZE = 512       # placeholder, not used by XGBoost

np.random.seed(SEED)
random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
except Exception:
    # skip if torch not available
    pass


def get_args():
    """
    Parse command-line arguments.

    Returns:
        args.data: path to input CSV
        args.group_col: column name for grouping folds
        args.model_out: path to save final model
    """
    p = argparse.ArgumentParser("5-fold CV for XGBoost surrogate")
    p.add_argument("-d", "--data", default="all_curves.csv",
                   help="CSV with TMB, HRP, H2O2, t90, I_max, tail_pct")
    p.add_argument("-g", "--group_col", default="Trial",
                   help="Group column if USE_GROUPS=True")
    p.add_argument("-o", "--model_out", default="xgb_surrogate_full.joblib",
                   help="Path to save model trained on full data")
    return p.parse_args()


def load_dataset(path, group_col):
    """
    Load features, targets, and optional group labels from CSV.

    Args:
        path: CSV file path.
        group_col: column for grouping CV folds.

    Returns:
        X: array of shape (n_samples, 3)
        y: array of shape (n_samples, 3)
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
    Construct a Pipeline with scaler and XGBoost regressors.

    Uses MultiOutputRegressor to train one XGB per target.
    """
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    )
    return Pipeline([
        ("scaler", StandardScaler()),  # scaling has no impact on trees but is harmless
        ("xgb", MultiOutputRegressor(xgb))
    ])


def cross_validate(X, y, groups=None):
    """
    Perform K-fold (or GroupKFold) cross-validation and print metrics.

    Args:
        X: feature matrix
        y: target matrix
        groups: group labels or None

    Returns:
        List of dicts with per-fold statistics.
    """
    cv = GroupKFold(n_splits=KF_SPLITS) if USE_GROUPS else \
         KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)

    fold_stats = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        model = build_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        stats = {
            "fold": fold,
            "t90_r2":   r2_score(y[te][:, 0], pred[:, 0]),
            "t90_mae":  mean_absolute_error(y[te][:, 0], pred[:, 0]),
            "imax_r2":  r2_score(y[te][:, 1], pred[:, 1]),
            "imax_mae": mean_absolute_error(y[te][:, 1], pred[:, 1]),
            "tail_r2":  r2_score(y[te][:, 2], pred[:, 2]),
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
    - parse CLI args
    - load dataset
    - run cross-validation
    - train and save final model on all data
    """
    args = get_args()
    X, y, groups = load_dataset(args.data, args.group_col)
    stats = cross_validate(X, y, groups)

    # train final model on full data and save
    model = build_model()
    model.fit(X, y)
    joblib.dump(model, args.model_out)
    print(f"✅ Trained full model and saved to {args.model_out}")


if __name__ == "__main__":
    main()
