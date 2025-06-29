#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- 可复现性 & 设备 ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS     = 500
PATIENCE   = 50
BEST_HIDDEN= 256
BEST_DEPTH = 5
BEST_LR    = 1e-3

# ---------- Dataset 支持 y_scaler ----------
class KineticDS(Dataset):
    def __init__(self, df, scaler_x, fit_x, scaler_y=None, fit_y=False):
        x = df[["TMB","HRP","H2O2"]].to_numpy(dtype="float32")
        if fit_x: scaler_x.fit(x)
        self.x = scaler_x.transform(x).astype("float32")

        y = df[["t90","I_max","tail_pct"]].to_numpy(dtype="float32")
        if scaler_y is None:
            self.y = y
        else:
            if fit_y: scaler_y.fit(y)
            self.y = scaler_y.transform(y).astype("float32")

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

# ---------- 模型 ----------
class ResidualBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d,d), nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()
    def forward(self,x): return self.act(self.net(x)+x)

class KineticNet(nn.Module):
    def __init__(self,h=256,depth=5):
        super().__init__()
        self.input  = nn.Sequential(nn.Linear(3,h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h,3)
    def forward(self,x): return self.output(self.blocks(self.input(x)))

# ---------- 读 CSV & 划分 ----------
df = pd.read_csv("all_curves.csv")
train_full = df.sample(frac=0.85, random_state=SEED)
test_df    = df.drop(train_full.index)
val_df     = train_full.sample(frac=0.10, random_state=SEED)
train_df   = train_full.drop(val_df.index)

# 归一化器
scaler_x = StandardScaler()
scaler_y = StandardScaler()

train_ds = KineticDS(train_df, scaler_x, True,  scaler_y, True)
val_ds   = KineticDS(val_df,   scaler_x, False, scaler_y, False)
test_ds  = KineticDS(test_df,  scaler_x, False, scaler_y, False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ---------- 训练 ----------
net   = KineticNet(h=BEST_HIDDEN, depth=BEST_DEPTH).to(device)
opt   = torch.optim.AdamW(net.parameters(), lr=BEST_LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
loss_fn = nn.MSELoss()

best_val, bad = np.inf, 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS+1):
    # train
    net.train(); running=0
    for xb,yb in train_dl:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        l = loss_fn(net(xb), yb)
        l.backward(); opt.step()
        running += l.item()*xb.size(0)
    train_loss = running/len(train_ds)
    train_losses.append(train_loss)

    # val
    net.eval(); total=0
    with torch.no_grad():
        for xb,yb in val_dl:
            xb,yb = xb.to(device), yb.to(device)
            total += loss_fn(net(xb), yb).item()*xb.size(0)
    val_loss = total/len(val_ds)
    val_losses.append(val_loss)

    sched.step(val_loss)
    if val_loss < best_val-1e-4:
        best_val, bad = val_loss, 0
        best_state = net.state_dict()
    else:
        bad += 1
        if bad>=PATIENCE:
            print(f"Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
            break

    if epoch==1 or epoch%20==0:
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

# 损失曲线
plt.figure(figsize=(8,6))
plt.plot(train_losses,label="Train")
plt.plot(val_losses,  label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Train vs Val Loss"); plt.legend(); plt.grid(True)
plt.show()

# ---------- 测试评估 ----------
net.load_state_dict(best_state)
net.eval()
preds_s, truths_s = [], []
with torch.no_grad():
    for xb,yb in test_dl:
        xb,yb = xb.to(device), yb.to(device)
        preds_s.append(net(xb).cpu().numpy())
        truths_s.append(yb.cpu().numpy())
P_s = np.vstack(preds_s)
T_s = np.vstack(truths_s)
# inverse scale
P = scaler_y.inverse_transform(P_s)
T = scaler_y.inverse_transform(T_s)

# 全局指标
mse_all = mean_squared_error(T,P)
r2_all  = r2_score(T,P, multioutput="uniform_average")
mae_all = mean_absolute_error(T,P)
print(f"Test combined → MSE={mse_all:.4f}, R²={r2_all:.4f}, MAE={mae_all:.4f}")

# 分指标
names = ["t90","I_max","tail_pct"]
mae_per = mean_absolute_error(T,P, multioutput="raw_values")
r2_per  = r2_score(T,P, multioutput="raw_values")
print("\nPer-metric:")
for i,n in enumerate(names):
    print(f"  {n:8s} MAE={mae_per[i]:6.3f}, R²={r2_per[i]:6.3f}")

# 散点图
fig,axes = plt.subplots(1,3,figsize=(15,5))
for i,ax in enumerate(axes):
    ax.scatter(T[:,i],P[:,i],alpha=0.5)
    mn,mx = T[:,i].min(), T[:,i].max()
    ax.plot([mn,mx],[mn,mx],'r--')
    ax.set_title(f"{names[i]} True vs Pred")
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
plt.tight_layout(); plt.show()
