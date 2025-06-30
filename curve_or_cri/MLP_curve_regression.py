import pandas as pd
import numpy as np
import torch, time, random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Configuration & Reproducibility ----------
CSV_PATH   = "./all_curves.csv"
KF_SPLITS  = 5
USE_GROUPS = False
EPOCHS     = 500
BATCH_SIZE = 256
PATIENCE   = 50
SEED       = 41

# set random seeds and device
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Dataset ----------
class KineticDS(Dataset):
    """
    PyTorch Dataset for kinetic intensity curves.
    - Scales 3 input features with a StandardScaler.
    - Loads 60-dimensional intensity outputs.
    """
    def __init__(self, df, scaler=None, fit_scaler=False):
        x = df[["TMB", "HRP", "H2O2"]].to_numpy(dtype="float32")
        if fit_scaler:
            scaler.fit(x)
        # transform features and store as float32
        self.x = scaler.transform(x).astype("float32")
        # load 60 time-point intensities as targets
        self.y = df[[f"I{i}" for i in range(60)]].to_numpy(dtype="float32")

    def __len__(self):
        """Return number of samples."""
        return len(self.x)

    def __getitem__(self, i):
        """
        Return one sample (features, targets) as torch tensors.
        """
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


# ---------- Model Definition ----------
class ResidualBlock(nn.Module):
    """
    Residual block with two linear layers:
    - BatchNorm, ReLU, and Dropout between layers
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
        """Apply block and add skip connection, then activation."""
        return self.act(self.net(x) + x)


class KineticNet(nn.Module):
    """
    Deep feedforward network for curve prediction:
    - Input layer projects 3 features to hidden size
    - Sequence of ResidualBlocks for representation learning
    - Output layer predicts 60 intensity values
    """
    def __init__(self, h=256, depth=5):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(3, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h, 60)

    def forward(self, x):
        """Forward pass through input layer, residual blocks, and output layer."""
        return self.output(self.blocks(self.input(x)))


# ---------- Train & Validate One Fold ----------
def train_one_fold(train_df, val_df):
    """
    Train the model on one fold and evaluate on validation set.
    Returns:
        best_val_loss (float): lowest validation MSE achieved
        val_r2        (float): R² score on validation set
    """
    scaler = StandardScaler()

    # create datasets and loaders
    tr_ds  = KineticDS(train_df, scaler, fit_scaler=True)
    val_ds = KineticDS(val_df,   scaler, fit_scaler=False)
    tr_dl  = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, BATCH_SIZE)

    # initialize model, optimizer, scheduler, and loss
    net    = KineticNet().to(device)
    opt    = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_val_loss, bad_epochs = np.inf, 0
    # training loop with early stopping
    for epoch in range(1, EPOCHS + 1):
        # --- training phase ---
        net.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        # --- validation phase ---
        net.eval()
        val_loss, preds, truths = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = net(xb)
                val_loss += loss_fn(out, yb).item() * len(xb)
                preds.append(out.cpu())
                truths.append(yb.cpu())
        val_loss /= len(val_ds)
        sched.step(val_loss)

        # check for improvement
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = net.state_dict()
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    # restore best model state and compute R² on validation
    net.load_state_dict(best_state)
    P = torch.cat(preds).numpy()
    T = torch.cat(truths).numpy()
    val_r2 = r2_score(T, P, multioutput='uniform_average')
    return best_val_loss, val_r2


# ---------- Main Cross-Validation Flow ----------
df = pd.read_csv(CSV_PATH)
groups = df["Trial"] if USE_GROUPS else None

# choose KFold or GroupKFold based on USE_GROUPS flag
splitter = (
    GroupKFold(n_splits=df["Trial"].nunique())
    if USE_GROUPS else
    KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)
)

fold_mse, fold_r2 = [], []
for k, (tr_idx, val_idx) in enumerate(splitter.split(df, groups=groups), start=1):
    t0 = time.time()
    mse, r2 = train_one_fold(df.iloc[tr_idx], df.iloc[val_idx])
    fold_mse.append(mse)
    fold_r2.append(r2)
    print(f"Fold {k}:  MSE={mse:.2f} | R²={r2:.3f} | {time.time()-t0:.1f}s")

# summary of cross-validation performance
print("\n=== Cross-Val Summary ===")
print(f"MSE  mean ± std : {np.mean(fold_mse):.2f} ± {np.std(fold_mse):.2f}")
print(f"R²   mean ± std : {np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}")
