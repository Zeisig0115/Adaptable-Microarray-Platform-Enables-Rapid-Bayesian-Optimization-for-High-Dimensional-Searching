#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------- Reproducibility & Device --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"  # choose GPU if available

# -------------------- Hyperparameters --------------------
BATCH_SIZE  = 256
EPOCHS      = 500
PATIENCE    = 50
BEST_HIDDEN = 256
BEST_DEPTH  = 5
BEST_LR     = 1e-3

# -------------------- Dataset with Optional y-Scaling --------------------
class KineticDS(Dataset):
    """
    PyTorch Dataset for kinetic data.
    - Scales inputs (x) and optionally targets (y) with provided StandardScalers.
    """
    def __init__(self, df, scaler_x, fit_x, scaler_y=None, fit_y=False):
        # extract feature matrix and fit/transform scaler_x if requested
        x = df[["TMB", "HRP", "H2O2"]].to_numpy(dtype="float32")
        if fit_x:
            scaler_x.fit(x)
        self.x = scaler_x.transform(x).astype("float32")

        # extract target matrix and fit/transform scaler_y if provided
        y = df[["t90", "I_max", "tail_pct"]].to_numpy(dtype="float32")
        if scaler_y is None:
            self.y = y
        else:
            if fit_y:
                scaler_y.fit(y)
            self.y = scaler_y.transform(y).astype("float32")

    def __len__(self):
        """Return number of samples."""
        return len(self.x)

    def __getitem__(self, i):
        """Return a single sample (feature, target) as torch tensors."""
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

# -------------------- Residual Block Definition --------------------
class ResidualBlock(nn.Module):
    """
    Single residual block:
    - Two linear layers with BatchNorm, ReLU, and dropout
    - Skip connection adds input to output before final activation
    """
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        """Apply block and add input (skip), then activate."""
        return self.act(self.net(x) + x)

# -------------------- Main Network Definition --------------------
class KineticNet(nn.Module):
    """
    Deep feedforward network for kinetic predictions:
    - Input layer projects 3 features to hidden dimension
    - Series of ResidualBlocks for feature extraction
    - Output layer predicts 3 target values
    """
    def __init__(self, h=BEST_HIDDEN, depth=BEST_DEPTH):
        super().__init__()
        self.input  = nn.Sequential(
            nn.Linear(3, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h, 3)

    def forward(self, x):
        """Forward pass through input, residual blocks, and output layer."""
        return self.output(self.blocks(self.input(x)))

# -------------------- Load Data & Split --------------------
df = pd.read_csv("all_curves.csv")
# 85% train, 10% validation, 15% holdout test
train_full = df.sample(frac=0.85, random_state=SEED)
test_df    = df.drop(train_full.index)
val_df     = train_full.sample(frac=0.10, random_state=SEED)
train_df   = train_full.drop(val_df.index)

# initialize scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# create Dataset and DataLoader for each split
train_ds = KineticDS(train_df, scaler_x, True,  scaler_y, True)
val_ds   = KineticDS(val_df,   scaler_x, False, scaler_y, False)
test_ds  = KineticDS(test_df,  scaler_x, False, scaler_y, False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# -------------------- Training Loop --------------------
net     = KineticNet().to(device)
opt     = torch.optim.AdamW(net.parameters(), lr=BEST_LR, weight_decay=1e-4)
sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
loss_fn = nn.MSELoss()

best_val, bad = np.inf, 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # --- Training Phase ---
    net.train()
    running_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_ds)
    train_losses.append(train_loss)

    # --- Validation Phase ---
    net.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            val_running += loss_fn(net(xb), yb).item() * xb.size(0)
    val_loss = val_running / len(val_ds)
    val_losses.append(val_loss)

    # adjust learning rate and check early stopping
    sched.step(val_loss)
    if val_loss < best_val - 1e-4:
        best_val, bad = val_loss, 0
        best_state = net.state_dict()
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
            break

    # periodic logging
    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

# -------------------- Plot Loss Curves --------------------
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Test Set Evaluation --------------------
net.load_state_dict(best_state)
net.eval()
preds_list, truths_list = [], []

with torch.no_grad():
    for xb, yb in test_dl:
        xb, yb = xb.to(device), yb.to(device)
        preds_list.append(net(xb).cpu().numpy())
        truths_list.append(yb.cpu().numpy())

P_s = np.vstack(preds_list)
T_s = np.vstack(truths_list)

# inverse-transform targets and predictions
P = scaler_y.inverse_transform(P_s)
T = scaler_y.inverse_transform(T_s)

# compute global metrics
mse_all = mean_squared_error(T, P)
r2_all  = r2_score(T, P, multioutput="uniform_average")
mae_all = mean_absolute_error(T, P)
print(f"Test combined → MSE={mse_all:.4f}, R²={r2_all:.4f}, MAE={mae_all:.4f}")

# compute per-target metrics
names   = ["t90", "I_max", "tail_pct"]
mae_per = mean_absolute_error(T, P, multioutput="raw_values")
r2_per  = r2_score(T, P, multioutput="raw_values")
print("\nPer-metric results:")
for i, name in enumerate(names):
    print(f"  {name:8s} MAE={mae_per[i]:6.3f}, R²={r2_per[i]:6.3f}")

# -------------------- Scatter Plots of Predictions vs True --------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.scatter(T[:, i], P[:, i], alpha=0.5)
    mn, mx = T[:, i].min(), T[:, i].max()
    ax.plot([mn, mx], [mn, mx], 'r--')  # reference diagonal
    ax.set_title(f"{names[i]}: True vs Predicted")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
plt.tight_layout()
plt.show()
