############################################################
# 0. Imports
############################################################
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from typing import Sequence
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from botorch.models import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from utils import load_chem_bo_data


############################################################
# 2. Load, split 80 / 20
############################################################
csv_path = "./data/botorch_ready.csv"          # 按需修改
X_all, Y_all, meta = load_chem_bo_data(csv_path)

# BoTorch 用 torch，但 sklearn 用 NumPy；这里先做 NumPy split
X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    X_all.numpy(), Y_all.numpy(),
    test_size=0.20, random_state=42, shuffle=True
)

# 转回 torch
X_train = torch.tensor(X_train_np, dtype=torch.double)
Y_train = torch.tensor(Y_train_np, dtype=torch.double)
X_test  = torch.tensor(X_test_np,  dtype=torch.double)

############################################################
# 3. Train Exact GP  (MixedSingleTaskGP, 3 outputs)
############################################################
num_switch = len(meta["switch"])
cat_dims   = list(range(num_switch))           # switch 列在最前 → 0…num_switch-1

gp_model = MixedSingleTaskGP(
    train_X=X_train,
    train_Y=Y_train,
    cat_dims=cat_dims,
    input_transform=Normalize(d=X_train.shape[-1]),
    outcome_transform=Standardize(m=Y_train.shape[-1]),
)

mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
fit_gpytorch_mll(
    mll,
    optimizer_kwargs={"options": {"maxiter": 75}},        # 减迭代数提速
    max_attempts=2,
    options={"num_restarts": 3, "raw_samples": 128},
)

# Predict on test set
gp_model.eval()
with torch.no_grad():
    posterior = gp_model.posterior(X_test)
    Y_gp_mean = posterior.mean.cpu().numpy()

rmse_gp = np.sqrt(
    mean_squared_error(Y_test_np, Y_gp_mean, multioutput="raw_values")
)

############################################################
# 4. Train Random-Forest baseline
############################################################
rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,    # 树数可调
            random_state=42,
            n_jobs=-1,
        ))
rf.fit(X_train_np, Y_train_np)
Y_rf_pred = rf.predict(X_test_np)

rmse_rf = np.sqrt(
    mean_squared_error(Y_test_np, Y_rf_pred, multioutput="raw_values")
)

############################################################
# 5. Report
############################################################
targets = meta["target"]
print("\n===  RMSE on 20 % test set  ===")
for idx, name in enumerate(targets):
    print(f"{name:<10s}  GP: {rmse_gp[idx]:7.4f}   RF: {rmse_rf[idx]:7.4f}")

rel_drop = (rmse_gp - rmse_rf) / rmse_rf * 100
print("\nRelative ↑RMSE vs RF (+ means GP worse):")
for idx, name in enumerate(targets):
    print(f"{name:<10s}:  {rel_drop[idx]:6.1f} %")
