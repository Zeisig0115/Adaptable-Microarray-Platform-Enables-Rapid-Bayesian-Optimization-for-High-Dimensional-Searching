import argparse, pandas as pd, joblib
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- 超参 & 可复现性 ----------
SEED       = 42
KF_SPLITS  = 5
USE_GROUPS = False         # 若想按 Trial 分组，把它改为 True
BATCH_SIZE = 512

# ---------- Seed ----------
import random, os, numpy as np
np.random.seed(SEED)
random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
except Exception as e:
    print("[Info] torch 未安装或 CUDA 无法加载，仅跳过 torch seed：", e)


# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser("5-fold CV for RF surrogate")
    p.add_argument("-d", "--data", default="all_curves.csv",
                   help="CSV path (需含 TMB, HRP, H2O2, t90, I_max, tail_pct)")
    p.add_argument("-g", "--group_col", default="Trial",
                   help="列名，若 USE_GROUPS=True 时作为分组折叠键")
    p.add_argument("-o", "--model_out", default="rf_surrogate_full.joblib",
                   help="输出：在全部样本上再次拟合的最终模型")
    return p.parse_args()


# ---------- 读取数据 ----------
def load_dataset(path, group_col):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} 不存在")
    df = pd.read_csv(path)
    X = df[["TMB", "HRP", "H2O2"]].values
    y = df[["t90", "I_max", "tail_pct"]].values
    groups = df[group_col].values if USE_GROUPS else None
    return X, y, groups


# ---------- 构建模型 ----------
def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                random_state=SEED,
                n_jobs=-1
            )
        ))
    ])


# ---------- 交叉验证 ----------
def cross_validate(X, y, groups=None):
    cv = GroupKFold(n_splits=KF_SPLITS) if USE_GROUPS else \
         KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)

    fold_stats = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        model = build_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        stats = {
            "fold": fold,
            "t90_r2"   : r2_score(y[te][:, 0], pred[:, 0]),
            "t90_mae"  : mean_absolute_error(y[te][:, 0], pred[:, 0]),
            "imax_r2"  : r2_score(y[te][:, 1], pred[:, 1]),
            "imax_mae" : mean_absolute_error(y[te][:, 1], pred[:, 1]),
            "tail_r2"  : r2_score(y[te][:, 2], pred[:, 2]),
            "tail_mae" : mean_absolute_error(y[te][:, 2], pred[:, 2]),
        }
        fold_stats.append(stats)
        print(f"[Fold {fold}] "
              f"t90_R2={stats['t90_r2']:.3f}  "
              f"Imax_R2={stats['imax_r2']:.3f}  "
              f"Tail_R2={stats['tail_r2']:.3f}")

    # 汇总
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


# ---------- 主程序 ----------
def main():
    args = get_args()
    X, y, groups = load_dataset(args.data, args.group_col)
    cross_validate(X, y, groups)


if __name__ == "__main__":
    main()
