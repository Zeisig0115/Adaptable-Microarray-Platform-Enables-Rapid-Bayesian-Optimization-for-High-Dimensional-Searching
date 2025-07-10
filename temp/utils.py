from pathlib import Path
from typing import Sequence, Tuple, Dict
import pandas as pd
import torch


def load_botorch_data(
    csv_path: str | Path,
    *,
    valid_col: str = "valid",
    switch_suffix: str = "_sw",
    level_suffix: str = "_lvl",
    exclude_level_cols: Sequence[str] = ("tmb", "hrp", "h2o2"),
    essential_cols: Sequence[str] = ("tmb", "hrp", "h2o2"),
    target_cols: Sequence[str] = ("t90", "I_max", "tail_pct"),
    dtype=torch.double,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, list]]:
    """
    Read CSV and split into BoTorch-ready tensors.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.
    valid_col : str
        Name of the boolean column used to filter valid rows.
    switch_suffix : str
        Suffix pattern identifying additive *switch* columns.
    level_suffix : str
        Suffix pattern identifying additive *concentration* columns.
    exclude_level_cols : Sequence[str]
        Level columns to drop from the additive-level set
        (they will still appear as essentials if listed separately).
    essential_cols : Sequence[str]
        Columns always kept as continuous essentials.
    target_cols : Sequence[str]
        Response columns to model (can be single- or multi-output).
    dtype : torch.dtype
        Torch dtype for returned tensors (BoTorch defaults to double).
    verbose : bool
        Print summary shapes for sanity-check.

    Returns
    -------
    train_X : torch.Tensor  [N, d]
    train_Y : torch.Tensor  [N, m]
    meta    : dict          {'switch', 'level', 'essential', 'target'}
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if valid_col in df.columns:
        df = df[df[valid_col]].reset_index(drop=True)

    switch_cols = [c for c in df.columns if c.endswith(switch_suffix)]
    level_cols = [
        c for c in df.columns
        if c.endswith(level_suffix) and c not in exclude_level_cols
    ]
    feature_cols = list(switch_cols) + list(level_cols) + list(essential_cols)

    X_np = df[feature_cols].to_numpy(dtype=float)
    Y_np = df[list(target_cols)].to_numpy(dtype=float)

    train_X = torch.tensor(X_np, dtype=dtype)
    train_Y = torch.tensor(Y_np, dtype=dtype)

    if verbose:
        info = {
            "# of switches": len(switch_cols),
            "# of additives": len(level_cols),
            "# of essentials": len(essential_cols),
            "# of feature dim": train_X.shape[1],
            "# of samples": train_X.shape[0],
        }
        for key, val in info.items():
            print(f"{key:<20}: {val}")

    meta = {
        "switch": switch_cols,
        "level": level_cols,
        "essential": list(essential_cols),
        "target": list(target_cols),
    }
    return train_X, train_Y, meta
