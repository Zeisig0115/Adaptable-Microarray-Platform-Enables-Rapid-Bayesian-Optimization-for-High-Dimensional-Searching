"""
xgb_cv.py —— 纯 XGBoost GBDT · 5-fold CV + 测试曲线可视化
"""
import numpy as np, pandas as pd, random, time, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------- Config ----------
CSV_PATH   = "./data/curves.csv"     # 改成自己的文件名
N_SPLITS   = 5                  # KFold 折数
TEST_RATIO = 0.15               # 外部独立测试集占比
SEED       = 42
np.random.seed(SEED);  random.seed(SEED)

# ---------- 1. 读取并拆分数据 ----------
df = pd.read_csv(CSV_PATH)

# 根据自己的列名取 60 个输出；下面示例既兼容 `I_0` 又兼容 `I0`
I_cols = [c for c in df.columns if c.upper().startswith("I")][:60]
feature_cols = [c for c in df.columns if c not in I_cols][:25]

X = df[feature_cols].values.astype(np.float32)
Y = df[I_cols].values.astype(np.float32)

# 外部 hold-out 测试集
X_trval, X_test, Y_trval, Y_test = train_test_split(
    X, Y, test_size=TEST_RATIO, random_state=SEED)

# （可选）对输入做标准化；树模型一般不敏感，这里举例带上
scaler = StandardScaler().fit(X_trval)
X_trval_s = scaler.transform(X_trval)
X_test_s  = scaler.transform(X_test)

# ---------- 2. 定义 XGBRegressor 超参 ----------
xgb_base = xgb.XGBRegressor(
    n_estimators     = 600,
    max_depth        = 4,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_lambda       = 1.0,
    objective        = "reg:squarederror",
    tree_method      = "hist",          # CPU hist 加速; GPU 可改 "gpu_hist"
    random_state     = SEED
)
# MultiOutputRegressor 会自动克隆 base，为每个输出训练一个子模型
gbr_multi = MultiOutputRegressor(xgb_base, n_jobs=-1)

# ---------- 3. 5-fold CV ----------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_mse, fold_r2, fold_models = [], [], []

for f, (tr_idx, val_idx) in enumerate(kf.split(X_trval_s), 1):
    t0 = time.time()
    X_tr, X_val = X_trval_s[tr_idx], X_trval_s[val_idx]
    Y_tr, Y_val = Y_trval[tr_idx], Y_trval[val_idx]

    model = MultiOutputRegressor(xgb_base, n_jobs=-1)
    model.fit(X_tr, Y_tr)

    Y_val_pred = model.predict(X_val)
    mse = mean_squared_error(Y_val, Y_val_pred)
    r2  = r2_score(Y_val, Y_val_pred, multioutput="uniform_average")
    fold_mse.append(mse);  fold_r2.append(r2);  fold_models.append(model)

    print(f"Fold {f}:  MSE={mse:.2f} | R²={r2:.3f} | {time.time()-t0:.1f}s")

# ---------- 4. CV 汇总 ----------
print("\n=== 5-Fold CV Summary ===")
print(f"MSE  mean ± std : {np.mean(fold_mse):.2f} ± {np.std(fold_mse):.2f}")
print(f"R²   mean ± std : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")

# 选验证 MSE 最低那折
best_fold = int(np.argmin(fold_mse))
best_model = fold_models[best_fold]

# ---------- 5. 外部测试集评估 ----------
Y_test_pred = best_model.predict(X_test_s)
test_mse = mean_squared_error(Y_test, Y_test_pred)
test_r2  = r2_score(Y_test, Y_test_pred, multioutput="uniform_average")
print(f"\n=== Hold-out Test ===\nMSE={test_mse:.2f} | R²={test_r2:.3f}")

# ---------- 6. 可视化 9 条曲线 ----------
idx_vis = random.sample(range(X_test.shape[0]), min(9, X_test.shape[0]))
for idx in idx_vis:
    plt.figure(figsize=(6,3))
    plt.plot(Y_test[idx], label="True")
    plt.plot(Y_test_pred[idx], "--", label="Pred")
    plt.title(f"Test sample {idx}")
    plt.xlabel("Time step"); plt.ylabel("Intensity")
    plt.legend(); plt.tight_layout(); plt.show()
