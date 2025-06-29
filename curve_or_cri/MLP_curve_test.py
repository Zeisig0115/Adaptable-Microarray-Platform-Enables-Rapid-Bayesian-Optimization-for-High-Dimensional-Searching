import pandas as pd, numpy as np, torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 可复现性 ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
EPOCHS      = 500
PATIENCE    = 50

BEST_HIDDEN = 256
BEST_DEPTH  = 5
BEST_LR     = 1e-3

# ---------- Dataset ----------
class KineticDS(Dataset):
    def __init__(self, df, scaler=None, fit_scaler=False):
        x = df[["TMB","HRP","H2O2"]].to_numpy(dtype="float32")
        if fit_scaler: scaler.fit(x)
        self.x = scaler.transform(x).astype("float32")
        self.y = df[[f"I{i}" for i in range(60)]].to_numpy(dtype="float32")
    def __len__(self):  return len(self.x)
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

# ---------- Model ----------
class ResidualBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d,d), nn.BatchNorm1d(d)
        )
        self.act = nn.ReLU()
    def forward(self,x): return self.act(self.net(x)+x)

class KineticNet(nn.Module):
    def __init__(self, h=256, depth=4):
        super().__init__()
        self.input  = nn.Sequential(nn.Linear(3,h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h,60)
    def forward(self,x): return self.output(self.blocks(self.input(x)))

# ---------- 读数据 & 三折划分 ----------
df = pd.read_csv("./all_curves.csv")

train_full = df.sample(frac=0.85, random_state=SEED)          # 85 % 先留给 train+val
test_df    = df.drop(train_full.index)                        # 15 % 做最终测试

val_frac   = 0.1                                              # 🔖 “train_full” 再拆 10 % 作验证
val_df     = train_full.sample(frac=val_frac, random_state=SEED)
train_df   = train_full.drop(val_df.index)

scaler = StandardScaler()
train_ds = KineticDS(train_df, scaler, fit_scaler=True)
val_ds   = KineticDS(val_df, scaler)
test_ds  = KineticDS(test_df, scaler)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------- 训练 ----------
net   = KineticNet(h=BEST_HIDDEN, depth=BEST_DEPTH).to(device)
opt   = torch.optim.AdamW(net.parameters(), lr=BEST_LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
loss_fn = nn.MSELoss()

best_val, bad = np.inf, 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS+1):
    # --- train phase ---
    net.train()
    running_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * len(xb)
    train_loss = running_loss / len(train_ds)
    train_losses.append(train_loss)

    # --- validation phase 🔖 ---
    net.eval()
    with torch.no_grad():
        val_loss = sum(
            loss_fn(net(xb.to(device)), yb.to(device)).item() * len(xb)
            for xb, yb in val_dl
        ) / len(val_ds)
    val_losses.append(val_loss)

    # --- scheduler & early-stopping ---
    sched.step(val_loss)
    if val_loss < best_val - 1e-4:
        best_val, bad = val_loss, 0
        best_state = net.state_dict()
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (val loss={val_loss:.4f})")
            break

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(train_losses, label="Train")   # 直接用列表即可
plt.plot(val_losses,   label="Val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# ---------- 测试 & 三指标对比 + 曲线可视化 ----------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 收集预测曲线
net.load_state_dict(best_state)
net.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        preds.append(net(xb).cpu().numpy())
P = np.vstack(preds)   # shape [N,60]

# 2. 从 test_df 获得真实曲线和 CSV 中的三指标真值
true_curves       = test_df[[f"I{i}" for i in range(60)]].to_numpy(dtype="float32")  # [N,60]
true_criteria_csv = test_df[["t90","I_max","tail_pct"]].to_numpy(dtype="float32")    # [N,3]

# 3. 对预测曲线计算三指标
def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    return np.convolve(y, np.ones(window)/window, mode="same")

def three_metrics(y: np.ndarray, window: int = 5):
    y_s     = smooth_curve(y, window)
    max_val = float(y_s.max()) if y_s.max()>0 else 0.0
    t90     = int(np.argmax(y_s >= 0.9*max_val)) if max_val>0 else 0
    tail_pct= float(y_s[-5:].mean()/max_val*100) if max_val>0 else 0.0
    return t90, max_val, tail_pct

metrics_pred_curve = np.array([three_metrics(c) for c in P], dtype="float32")  # [N,3]

# 4. 随机挑 9 个样本：打印 CSV 真值 vs 预测曲线算得
indices = np.random.choice(len(P), 9, replace=False)
print("=== 9 Random Samples：True Value vs From Reconstructed Curves ===\n")
for i in indices:
    t90_p, imax_p, tail_p = metrics_pred_curve[i]
    t90_c, imax_c, tail_c = true_criteria_csv[i]
    print(f"[Sample {i:3d}]")
    print(f"  • True Values：    t90={int(t90_c):3d}, I_max={imax_c:7.3f}, tail%={tail_c:6.2f}")
    print(f"  • From Curves：    t90={int(t90_p):3d}, I_max={imax_p:7.3f}, tail%={tail_p:6.2f}\n")

# 5. 全量 MAE & R²
mae = mean_absolute_error(true_criteria_csv, metrics_pred_curve, multioutput="raw_values")
r2  = r2_score(true_criteria_csv, metrics_pred_curve, multioutput="raw_values")
names = ["t90 (steps)", "I_max", "tail_pct (%)"]
print("=== 预测曲线算得指标 vs CSV 中指标 ===")
for j, name in enumerate(names):
    print(f"{name:12s} → MAE = {mae[j]:6.2f} | R² = {r2[j]:6.3f}")
print()

# 6. 可视化：画出这 9 个样本的真实 vs 预测曲线
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, i in zip(axes.flatten(), indices):
    ax.plot(range(60), true_curves[i], label="True")
    ax.plot(range(60), P[i],           label="Pred")
    ax.set_title(f"Sample {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("I(t)")
    ax.legend()
plt.tight_layout()
plt.show()



