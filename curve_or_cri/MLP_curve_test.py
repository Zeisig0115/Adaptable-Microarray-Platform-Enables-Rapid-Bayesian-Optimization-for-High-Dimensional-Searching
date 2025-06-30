import pandas as pd
import numpy as np
import torch
import random
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ---------- Reproducibility & Device Setup ----------
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Hyperparameters ----------
BATCH_SIZE    = 256
EPOCHS        = 500
PATIENCE      = 50
BEST_HIDDEN   = 256
BEST_DEPTH    = 5
BEST_LR       = 1e-3
VAL_FRAC      = 0.10  # fraction of training set used for validation

# ---------- Dataset Definition ----------
class KineticDS(Dataset):
    """
    PyTorch Dataset for kinetic intensity curves.
    - Scales 3 input features using a StandardScaler.
    - Loads 60-dimensional intensity outputs I0…I59.
    """
    def __init__(self, df, scaler=None, fit_scaler=False):
        # Extract features and optionally fit the scaler
        x = df[["TMB", "HRP", "H2O2"]].to_numpy(dtype="float32")
        if fit_scaler:
            scaler.fit(x)
        self.x = scaler.transform(x).astype("float32")

        # Extract all 60 intensity values as target
        self.y = df[[f"I{i}" for i in range(60)]].to_numpy(dtype="float32")

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """
        Return one sample:
        - features: torch.FloatTensor of shape (3,)
        - targets:  torch.FloatTensor of shape (60,)
        """
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


# ---------- Model Components ----------
class ResidualBlock(nn.Module):
    """
    Single residual block:
    - Two linear layers with BatchNorm, ReLU, and Dropout
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
        """Forward pass: apply block, add skip connection, then activate."""
        return self.act(self.net(x) + x)


class KineticNet(nn.Module):
    """
    Deep feedforward network for predicting full intensity curves:
    - Input layer: projects 3 features to hidden dimension
    - Sequence of ResidualBlocks for representation learning
    - Output layer: maps hidden features to 60 output values
    """
    def __init__(self, h=BEST_HIDDEN, depth=BEST_DEPTH):
        super().__init__()
        self.input  = nn.Sequential(
            nn.Linear(3, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h, 60)

    def forward(self, x):
        """Forward pass through input, residual blocks, and output layer."""
        return self.output(self.blocks(self.input(x)))


# ---------- Metric Extraction from Curves ----------
def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply a moving-average filter of given window size to smooth a 1D curve.
    """
    return np.convolve(y, np.ones(window) / window, mode="same")

def three_metrics(y: np.ndarray, window: int = 5):
    """
    Compute three summary metrics from a smoothed curve:
    - t90: index where curve first reaches 90% of its max
    - I_max: maximum smoothed intensity
    - tail_pct: mean of last 5 points divided by I_max, as percentage
    """
    y_s      = smooth_curve(y, window)
    max_val  = float(y_s.max()) if y_s.max() > 0 else 0.0
    t90      = int(np.argmax(y_s >= 0.9 * max_val)) if max_val > 0 else 0
    tail_pct = float(y_s[-5:].mean() / max_val * 100) if max_val > 0 else 0.0
    return t90, max_val, tail_pct


# ---------- Data Loading & Splitting ----------
df          = pd.read_csv("./all_curves.csv")
train_full  = df.sample(frac=0.85, random_state=SEED)   # 85% to train+val
test_df     = df.drop(train_full.index)                 # 15% hold-out test
val_df      = train_full.sample(frac=VAL_FRAC, random_state=SEED)
train_df    = train_full.drop(val_df.index)

# Initialize and fit scaler on training features
scaler      = StandardScaler()
train_ds    = KineticDS(train_df, scaler,      fit_scaler=True)
val_ds      = KineticDS(val_df,   scaler,      fit_scaler=False)
test_ds     = KineticDS(test_df,  scaler,      fit_scaler=False)

train_dl    = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl      = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_dl     = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ---------- Model Training ----------
net         = KineticNet(h=BEST_HIDDEN, depth=BEST_DEPTH).to(device)
opt         = torch.optim.AdamW(net.parameters(), lr=BEST_LR, weight_decay=1e-4)
sched       = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
loss_fn     = nn.MSELoss()

best_val, bad = np.inf, 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # -- Training Phase --
    net.train()
    running = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_ds)
    train_losses.append(train_loss)

    # -- Validation Phase --
    net.eval()
    with torch.no_grad():
        val_running = sum(
            loss_fn(net(xb.to(device)), yb.to(device)).item() * xb.size(0)
            for xb, yb in val_dl
        )
    val_loss = val_running / len(val_ds)
    val_losses.append(val_loss)

    # -- Scheduler & Early Stopping --
    sched.step(val_loss)
    if val_loss < best_val - 1e-4:
        best_val, bad = val_loss, 0
        best_state = net.state_dict()
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (val_loss={val_loss:.4f})")
            break

    if epoch == 1 or epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

# ---------- Plot Training & Validation Loss ----------
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ---------- Test Evaluation & Curve Metrics ----------
# Restore best model and collect predictions
net.load_state_dict(best_state)
net.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        preds.append(net(xb.to(device)).cpu().numpy())
P = np.vstack(preds)  # predicted curves shape [N, 60]

# True curves and criteria from CSV
true_curves       = test_df[[f"I{i}" for i in range(60)]].to_numpy(dtype="float32")
true_criteria_csv = test_df[["t90", "I_max", "tail_pct"]].to_numpy(dtype="float32")

# Compute metrics from predicted curves
metrics_pred_curve = np.array([three_metrics(c) for c in P], dtype="float32")

# Print comparisons for 9 random samples
indices = np.random.choice(len(P), 9, replace=False)
print("=== 9 Random Samples: CSV vs From Predicted Curves ===")
for i in indices:
    t90_p, imax_p, tail_p = metrics_pred_curve[i]
    t90_c, imax_c, tail_c = true_criteria_csv[i]
    print(f"[Sample {i:3d}] True:    t90={int(t90_c):3d}, I_max={imax_c:7.3f}, tail%={tail_c:6.2f}")
    print(f"           FromCurves t90={int(t90_p):3d}, I_max={imax_p:7.3f}, tail%={tail_p:6.2f}\n")

# Compute overall MAE & R² for each criterion
mae_vals = mean_absolute_error(true_criteria_csv, metrics_pred_curve, multioutput="raw_values")
r2_vals  = r2_score(true_criteria_csv, metrics_pred_curve, multioutput="raw_values")
names    = ["t90 (steps)", "I_max", "tail_pct (%)"]
print("=== Overall Metrics (Predicted Curves vs CSV Criteria) ===")
for j, name in enumerate(names):
    print(f"{name:12s} → MAE={mae_vals[j]:6.2f}, R²={r2_vals[j]:6.3f}")
print()

# Plot true vs predicted curves for those 9 samples
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax, i in zip(axes.flatten(), indices):
    ax.plot(range(60), true_curves[i], label="True")
    ax.plot(range(60), P[i],           label="Pred")
    ax.set_title(f"Sample {i}")
    ax.set_xlabel("Time Point")
    ax.set_ylabel("Intensity")
    ax.legend()
plt.tight_layout()
plt.show()
