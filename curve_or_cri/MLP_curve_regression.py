
import pandas as pd, numpy as np, torch, random, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 基础配置 ----------
CSV_PATH   = "./data/curves.csv"
KF_SPLITS  = 5          # 五折交叉验证
USE_GROUPS = False      # 若有 “Trial” 分组列可设 True
EPOCHS     = 500
BATCH_SIZE = 256
PATIENCE   = 50
SEED       = 41
device     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------- 1. 读数据 ----------------
df = pd.read_csv('./data/curves.csv')          # ← 这里按你的路径
I_cols       = [f'I_{i}' for i in range(60)]   # 60-point 序列列名
feature_cols = [c for c in df.columns if c not in I_cols][:25]  # 前 25 维浓度

TARGET_COLS  = I_cols                          # 目标输出列
IN_DIM       = len(feature_cols)               # =25

# ---------------- 2. Dataset ----------------
class KineticDS(Dataset):
    def __init__(self, df, scaler, fit_scaler=False):
        x = df[feature_cols].to_numpy(dtype='float32')

        if fit_scaler:           # 只有训练集会执行
            scaler.fit(x)
        self.x = scaler.transform(x).astype('float32')
        self.y = df[TARGET_COLS].to_numpy(dtype='float32')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

# ---------------- 3. 网络 ----------------
class ResidualBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d,d), nn.BatchNorm1d(d)
        ); self.act = nn.ReLU()
    def forward(self, x): return self.act(self.net(x) + x)

class KineticNet(nn.Module):
    def __init__(self, h=256, depth=5):
        super().__init__()
        self.input  = nn.Sequential(nn.Linear(IN_DIM, h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h, 60)
    def forward(self, x): return self.output(self.blocks(self.input(x)))

# ---------- 4. 单折训练 / 验证 ----------
def train_one_fold(train_df, val_df):
    scaler = StandardScaler()

    tr_ds  = KineticDS(train_df, scaler, fit_scaler=True)
    val_ds = KineticDS(val_df,   scaler, fit_scaler=False)
    tr_dl  = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, BATCH_SIZE, drop_last=False)

    net   = KineticNet().to(device)
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_loss, bad = np.inf, 0
    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        net.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        # ---- val ----
        net.eval(); val_loss = 0.0; preds = []; truths = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = net(xb)
                val_loss += loss_fn(out, yb).item() * len(xb)
                preds.append(out.cpu()); truths.append(yb.cpu())
        val_loss /= len(val_ds); sch.step(val_loss)

        # ---- early-stop ----
        if val_loss < best_loss - 1e-4:
            best_loss, bad, best_state = val_loss, 0, net.state_dict()
        else:
            bad += 1
            if bad >= PATIENCE: break

    net.load_state_dict(best_state)
    P = torch.cat(preds).numpy(); T = torch.cat(truths).numpy()
    val_r2 = r2_score(T, P, multioutput='uniform_average')
    return best_loss, val_r2

# ---------- 5. K-Fold 主流程 ----------
df  = pd.read_csv(CSV_PATH)
groups = df["Trial"] if USE_GROUPS else None
splitter = (
    GroupKFold(n_splits=df["Trial"].nunique())
    if USE_GROUPS else
    KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)
)

fold_mse, fold_r2 = [], []
for k, (tr, va) in enumerate(splitter.split(df, groups=groups), 1):
    t0 = time.time()
    mse, r2 = train_one_fold(df.iloc[tr], df.iloc[va])
    fold_mse.append(mse); fold_r2.append(r2)
    print(f"Fold {k}: MSE={mse:.3f} | R²={r2:.3f} | {time.time()-t0:.1f}s")

print("\n=== 25D 输入·交叉验证汇总 ===")
print(f"MSE  mean ± std : {np.mean(fold_mse):.3f} ± {np.std(fold_mse):.3f}")
print(f"R²   mean ± std : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
