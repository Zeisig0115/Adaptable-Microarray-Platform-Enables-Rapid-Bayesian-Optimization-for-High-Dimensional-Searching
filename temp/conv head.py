"""
gbdt_conv_cv.py  ——  5-fold CV + 测试曲线可视化
依赖: pandas numpy scikit-learn torch matplotlib
> pip install pandas numpy scikit-learn torch matplotlib
"""

import numpy as np, pandas as pd, random, time, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ------------------ Config ------------------
CSV_PATH     = "./data/curves.csv"       # 数据文件
N_SPLITS     = 5                    # KFold
TEST_RATIO   = 0.15                 # 外部独立测试
EPOCHS       = 300                  # Conv 头最多迭代
PATIENCE     = 40                   # Early-Stopping
SEED         = 42
CHN, KERNEL  = 8, 3                 # ConvHead 超参

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Conv Head ------------------
class ConvHead(nn.Module):
    """
    复制边界填充 + 残差: 输出 = x + Conv(Conv(x))
    保证长度恒 60，避免首尾塌陷
    """
    def __init__(self, channels=CHN, k=KERNEL):
        super().__init__()
        self.k = k
        self.pad = nn.ReplicationPad1d(k // 2)
        self.conv1 = nn.Conv1d(1, channels, k, padding=0)
        self.conv2 = nn.Conv1d(channels, 1, k, padding=0)

    def forward(self, x):               # x: [B,1,60]
        y = F.relu(self.conv1(self.pad(x)))
        y = self.conv2(self.pad(y))
        return x + y                    # 残差细调

# helper
to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(1)

# ------------------ 1. 读数据 ------------------
df = pd.read_csv(CSV_PATH)
I_cols = [f"I_{i}" if f"I_{i}" in df.columns else f"I{i}" for i in range(60)]
feature_cols = [c for c in df.columns if c not in I_cols][:25]  # 若列少于 25 自动截取

X = df[feature_cols].values.astype(np.float32)
Y = df[I_cols].values.astype(np.float32)

# 外部 hold-out 测试
X_trval, X_test, Y_trval, Y_test = train_test_split(
    X, Y, test_size=TEST_RATIO, random_state=SEED
)

# 归一化仅对 GBDT 可选（树模型不敏感），这里保留示例
scaler = StandardScaler().fit(X_trval)
X_trval_s = scaler.transform(X_trval)
X_test_s  = scaler.transform(X_test)

# ------------------ 2. 5-fold CV ------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_mse, fold_r2, fold_states = [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_trval_s), 1):
    t0 = time.time()
    X_tr, X_val = X_trval_s[tr_idx], X_trval_s[val_idx]
    Y_tr, Y_val = Y_trval[tr_idx], Y_trval[val_idx]

    # ---- 2.1  Gradient Boosting ----
    gbr = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED
        )
    )
    gbr.fit(X_tr, Y_tr)

    Y_tr_pred = gbr.predict(X_tr).astype(np.float32)
    Y_val_pred = gbr.predict(X_val).astype(np.float32)

    # ---- 2.2  Conv Head 训练 ----
    conv = ConvHead().to(device)
    opt  = optim.Adam(conv.parameters(), lr=5e-3)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.MSELoss()

    Xtr_t = to_tensor(Y_tr_pred)
    Ytr_t = to_tensor(Y_tr)
    Xval_t= to_tensor(Y_val_pred)
    Yval_t= to_tensor(Y_val)

    best, bad = 1e9, 0
    for ep in range(1, EPOCHS+1):
        conv.train();  opt.zero_grad()
        loss = crit(conv(Xtr_t), Ytr_t); loss.backward(); opt.step()

        conv.eval()
        with torch.no_grad():
            v_loss = crit(conv(Xval_t), Yval_t).item()
        sch.step(v_loss)

        if v_loss < best - 1e-6:
            best, bad = v_loss, 0
            best_state = {k: v.cpu() for k, v in conv.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:  break

    conv.load_state_dict(best_state)
    with torch.no_grad():
        Y_val_ref = conv(Xval_t).cpu().squeeze(1).numpy()

    mse = mean_squared_error(Y_val, Y_val_ref)
    r2  = r2_score(Y_val, Y_val_ref, multioutput='uniform_average')
    fold_mse.append(mse); fold_r2.append(r2); fold_states.append((gbr, best_state))
    print(f"Fold {fold}: MSE={mse:.2f} | R²={r2:.3f} | {time.time()-t0:.1f}s")

# ------------------ 3. CV 汇总 ------------------
print("\n=== 5-Fold CV Summary ===")
print(f"MSE  mean ± std : {np.mean(fold_mse):.2f} ± {np.std(fold_mse):.2f}")
print(f"R²   mean ± std : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")

# 选取验证 MSE 最低那一折的模型
best_fold = int(np.argmin(fold_mse))
gbr_best, conv_state = fold_states[best_fold]
conv_best = ConvHead().to(device)
conv_best.load_state_dict(conv_state)

# ------------------ 4. 在外部测试集评估 + 可视化 ------------------
Y_test_pred = gbr_best.predict(X_test_s).astype(np.float32)
with torch.no_grad():
    Y_test_ref = conv_best(to_tensor(Y_test_pred)).cpu().squeeze(1).numpy()

test_mse = mean_squared_error(Y_test, Y_test_ref)
test_r2  = r2_score(Y_test, Y_test_ref, multioutput='uniform_average')
print(f"\n=== Hold-out Test ===\nMSE={test_mse:.2f} | R²={test_r2:.3f}")

# 画 9 条随机曲线
idx_vis = random.sample(range(X_test.shape[0]), 9)
for i, idx in enumerate(idx_vis, 1):
    plt.figure()
    plt.plot(Y_test[idx], label='True')
    plt.plot(Y_test_ref[idx], label='Pred', linestyle='--')
    plt.title(f'Test Sample {idx}')
    plt.xlabel('Time step'); plt.ylabel('Intensity'); plt.legend()
    plt.tight_layout(); plt.show()
