import pandas as pd
import numpy as np
import torch, random, time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold

# Set global constants and seeds for reproducibility and device selection
SEED       = 42
KF_SPLITS  = 5
USE_GROUPS = False
EPOCHS     = 500
BATCH_SIZE = 512
PATIENCE   = 50

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


class KineticDS(Dataset):
    """
    Dataset for kinetic measurements.
    Applies feature and target scaling based on provided StandardScaler instances.
    """
    def __init__(self, df, scaler_x, fit_x, scaler_y=None, fit_y=False):
        # Extract input features and optionally fit/transform scaler_x
        x = df[["TMB","HRP","H2O2"]].to_numpy(dtype="float32")
        if fit_x:
            scaler_x.fit(x)
        self.x = scaler_x.transform(x).astype("float32")

        # Extract targets and optionally fit/transform scaler_y
        y = df[["t90","I_max","tail_pct"]].to_numpy(dtype="float32")
        if scaler_y is None:
            self.y = y
        else:
            if fit_y:
                scaler_y.fit(y)
            self.y = scaler_y.transform(y).astype("float32")

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, i):
        """Retrieve a single sample (features and target) as torch tensors."""
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


class ResidualBlock(nn.Module):
    """
    Single residual block with two linear layers, batch normalization,
    ReLU activations, dropout, and a skip connection.
    """
    def __init__(self, d, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d),
            nn.BatchNorm1d(d)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        """Forward pass: apply block and add input (skip connection), then activate."""
        return self.act(self.net(x) + x)


class KineticNet(nn.Module):
    """
    Fully connected network for kinetic modeling:
    - Input layer transforms 3 features to hidden dimension
    - Sequence of residual blocks for deep representation
    - Output layer maps back to 3 target values
    """
    def __init__(self, h=256, depth=5):
        super().__init__()
        self.input  = nn.Sequential(
            nn.Linear(3, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResidualBlock(h) for _ in range(depth)])
        self.output = nn.Linear(h, 3)

    def forward(self, x):
        """Forward pass through input layer, residual blocks, and output layer."""
        return self.output(self.blocks(self.input(x)))


def train_one_fold(train_df, val_df):
    """
    Train the model on a single train/validation split.
    Returns:
        mse (float): Mean squared error on validation set (unscaled).
        r2  (float): R² score on validation set (unscaled).
    """
    # Initialize scalers and datasets for this fold
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    tr_ds = KineticDS(train_df, scaler_x, fit_x=True,  scaler_y=scaler_y, fit_y=True)
    val_ds= KineticDS(val_df,   scaler_x, fit_x=False, scaler_y=scaler_y, fit_y=False)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
    val_dl= DataLoader(val_ds, BATCH_SIZE)

    # Initialize model, optimizer, scheduler, and loss function
    net   = KineticNet().to(device)
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    best_loss, bad_epochs = np.inf, 0
    # Training loop with early stopping based on validation loss
    for epoch in range(1, EPOCHS+1):
        net.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        net.eval()
        total_loss, preds, truths = 0, [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = net(xb)
                total_loss += loss_fn(out, yb).item() * xb.size(0)
                preds.append(out.cpu().numpy())
                truths.append(yb.cpu().numpy())
        avg_loss = total_loss / len(val_ds)
        sched.step(avg_loss)

        # Check for improvement
        if avg_loss < best_loss - 1e-4:
            best_loss, bad_epochs = avg_loss, 0
            best_state = net.state_dict()
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    # Restore best model and unscale predictions for final metrics
    net.load_state_dict(best_state)
    P_scaled = np.vstack(preds)
    T_scaled = np.vstack(truths)
    P = scaler_y.inverse_transform(P_scaled)
    T = scaler_y.inverse_transform(T_scaled)

    mse = mean_squared_error(T, P)
    r2  = r2_score(T, P, multioutput="uniform_average")
    return mse, r2


# Load data and set up cross-validation splitter
df = pd.read_csv("all_curves.csv")
groups = df["Trial"] if USE_GROUPS else None
splitter = (
    GroupKFold(n_splits=df["Trial"].nunique())
    if USE_GROUPS
    else KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)
)

# Perform cross-validation and report fold-wise and overall metrics
fold_mse, fold_r2 = [], []
for k, (tr_idx, val_idx) in enumerate(splitter.split(df, groups=groups), start=1):
    t0 = time.time()
    mse, r2 = train_one_fold(df.iloc[tr_idx], df.iloc[val_idx])
    fold_mse.append(mse)
    fold_r2.append(r2)
    print(f"Fold {k}: MSE={mse:.4f} | R²={r2:.4f} | {(time.time()-t0):.1f}s")

print("\n=== CV Summary ===")
print(f"MSE mean ± std : {np.mean(fold_mse):.4f} ± {np.std(fold_mse):.4f}")
print(f"R²  mean ± std : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
