import pandas as pd
import numpy as np
import torch, time
from pathlib import Path
from typing import Sequence, List
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.transforms import Normalize, Standardize

from utils import load_chem_bo_data

tkwargs = {"dtype": torch.double, "device": "cpu"}
torch.set_default_dtype(torch.double)

csv_path = "./data/botorch_ready.csv"
X_all, Y_all, meta = load_chem_bo_data(csv_path)

X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    X_all.numpy(), Y_all.numpy(), test_size=0.20, random_state=42, shuffle=True
)

X_train = torch.tensor(X_train_np, **tkwargs)
Y_train = torch.tensor(Y_train_np, **tkwargs)
X_test  = torch.tensor(X_test_np,  **tkwargs)

############################################################
# 2. Train SAAS GP – one model per output
############################################################
warmup, num_samples = 256, 256       # 轻量版；正式可升到 512/1024+
targets = meta["target"]
rmse_saas: List[float] = []

for idx, tname in enumerate(targets):
    print(f"\n=== Training SAAS GP for output '{tname}' ({idx+1}/{len(targets)}) ===")
    y_i = Y_train[:, [idx]]          # 单输出形状 (N,1)

    model_i = SaasFullyBayesianSingleTaskGP(
        train_X=X_train,
        train_Y=y_i,
        input_transform=Normalize(d=X_train.shape[-1]),
        outcome_transform=Standardize(m=1),
    )

    t0 = time.time()
    fit_fully_bayesian_model_nuts(
        model=model_i,
        warmup_steps=warmup,
        num_samples=num_samples,
        disable_progbar=True,
    )
    print(f"  ↳ fit time: {time.time() - t0:.1f} s")

    # ── Predict on test ──
    model_i.eval()
    with torch.no_grad():
        posterior = model_i.posterior(X_test)
        mu = posterior.mean.squeeze(-1).cpu().numpy()   # (N_test,)
    rmse = np.sqrt(mean_squared_error(Y_test_np[:, idx], mu))
    rmse_saas.append(rmse)
    print(f"  ↳ test RMSE: {rmse:.4f}")

############################################################
# 3. Random-Forest baseline (保持原参数)
############################################################
rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ))
rf.fit(X_train_np, Y_train_np)
Y_rf_pred = rf.predict(X_test_np)
rmse_rf = np.sqrt(
    mean_squared_error(Y_test_np, Y_rf_pred, multioutput="raw_values")
)

############################################################
# 4. Report
############################################################
print("\n===  RMSE on 20 % test set  ===")
for idx, name in enumerate(targets):
    print(f"{name:<10s}  SAAS: {rmse_saas[idx]:7.4f}   RF: {rmse_rf[idx]:7.4f}")

rel_drop = (np.array(rmse_saas) - rmse_rf) / rmse_rf * 100
print("\nRelative ↑RMSE vs RF (+ means SAAS worse):")
for idx, name in enumerate(targets):
    print(f"{name:<10s}:  {rel_drop[idx]:6.1f} %")
