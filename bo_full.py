from __future__ import annotations
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import math
from itertools import combinations, product

import numpy as np
import pandas as pd
import torch

from fit_model import fit_gp

from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed

torch.set_default_dtype(torch.double)

FEATURE_WHITELIST = [
    "tmb", "hrp", "h2o2", "peg20k", "pl127", "bsa", "pva", "tw80",
    "glycerol", "tw20", "imidazole", "tx100", "edta", "mgcl2", "sucrose", "cacl2",
    "znso4", "paa", "mnso4", "peg200k", "feso4", "peg6000", "peg400"
]

ESSENTIALS = ["tmb", "hrp", "h2o2"]
DEFAULT_Q = 40
DEFAULT_OUT_CSV = "bo_candidates_slots_gp.csv"
EPS = 1e-7


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_2d_y(y: np.ndarray | List[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _canonicalize_pairs(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    valid_pairs = [(int(aid), float(np.clip(r, 0.0, 1.0))) for aid, r in pairs if r > 0.0 and int(aid) > 0]
    bucket: Dict[int, float] = {}
    for aid, r in valid_pairs:
        bucket[aid] = float(np.clip(bucket.get(aid, 0.0) + r, 0.0, 1.0))
    return sorted(bucket.items(), key=lambda x: x[0])


class SlotCodec:
    def __init__(self, essentials: List[str], additives: List[str], ranges: Dict[str, Tuple[float, float]],
                 k_slots: int):
        self.E = list(essentials)
        self.adds = list(additives)
        self.A = len(self.adds)
        self.k = int(k_slots)
        self.name2id = {name: i + 1 for i, name in enumerate(self.adds)}
        self.id2name = {i + 1: name for i, name in enumerate(self.adds)}
        self.ranges = dict(ranges)

    def encode_row(self, row: Dict[str, float]) -> np.ndarray:
        z = np.zeros(3 + 2 * self.k, dtype=np.float64)
        for i, e in enumerate(self.E):
            z[i] = row.get(e, 0.0)
        pairs = []
        for a in self.adds:
            v = row.get(a, 0.0)
            if v is not None and abs(v) > EPS:
                lo, hi = self.ranges.get(a, (0.0, 1.0))
                denom = (hi - lo) if hi > lo else 1e-6
                r = np.clip((v - lo) / denom, 0.0, 1.0)
                pairs.append((self.name2id[a], r))
        pairs = _canonicalize_pairs(pairs)[:self.k]
        pairs += [(0, 0.0)] * (self.k - len(pairs))
        for s, (aid, r) in enumerate(pairs):
            z[3 + 2 * s] = float(aid)
            z[3 + 2 * s + 1] = float(r)
        return z

    def encode(self, rows: List[Dict[str, float]]) -> np.ndarray:
        return np.array([self.encode_row(r) for r in rows])

    def decode(self, Z: np.ndarray) -> List[Dict[str, float]]:
        Z = np.atleast_2d(Z)
        decoded_rows: List[Dict[str, float]] = []
        for z in Z:
            row = {e: z[i] for i, e in enumerate(self.E)}
            pairs = []
            for s in range(self.k):
                aid = int(round(z[3 + 2 * s]))
                r = float(z[3 + 2 * s + 1])
                if aid > 0 and r > 0.0:
                    aid = int(np.clip(aid, 1, self.A))
                    pairs.append((aid, np.clip(r, 0.0, 1.0)))
            for aid, r in _canonicalize_pairs(pairs):
                name = self.id2name[aid]
                lo, hi = self.ranges[name]
                row[name] = lo + r * (hi - lo)
            decoded_rows.append(row)
        return decoded_rows

    def postprocess_batch(self, Zt: torch.Tensor) -> torch.Tensor:
        if Zt.dim() not in [2, 3]:
            raise ValueError(f"Input Tensor dims should be 2 or 3, but get{Zt.dim()}")
        Z = Zt.detach().cpu().clone()
        is_2d = Z.dim() == 2
        if is_2d: Z = Z.unsqueeze(0)
        B, q, d = Z.shape
        expected_d = 3 + 2 * self.k
        assert d == expected_d, f"Encoding dims mismatch: expected{expected_d}, got{d}"
        for b in range(B):
            for i in range(q):
                ids = Z[b, i, 3::2].round().clamp_(0, self.A)
                ratios = Z[b, i, 4::2].clamp_(0.0, 1.0)
                ratios.masked_fill_(ids == 0, 0.0)
                pairs = [(int(ids[s].item()), ratios[s].item()) for s in range(self.k)]
                pairs = _canonicalize_pairs(pairs)[:self.k]
                pairs += [(0, 0.0)] * (self.k - len(pairs))
                for s, (aid, r) in enumerate(pairs):
                    Z[b, i, 3 + 2 * s] = float(aid)
                    Z[b, i, 3 + 2 * s + 1] = float(r)
        Z = Z.squeeze(0) if is_2d else Z
        return Z.to(Zt.device, dtype=Zt.dtype)

    def get_bounds(self, Z_hist: np.ndarray, device: torch.device) -> torch.Tensor:
        Z = np.asarray(Z_hist)
        lb_e = np.min(Z[:, :3], axis=0)
        ub_e = np.max(Z[:, :3], axis=0)
        same_bounds = ub_e <= lb_e
        ub_e[same_bounds] = lb_e[same_bounds] + 1e-6
        lb_slots = np.ravel([(0.0, 0.0)] * self.k)
        ub_slots = np.ravel([(float(self.A), 1.0)] * self.k)
        lb = np.concatenate([lb_e, lb_slots])
        ub = np.concatenate([ub_e, ub_slots])
        bounds_np = np.stack([lb, ub])
        return torch.tensor(bounds_np, device=device, dtype=torch.double)


# ----------------- BO -----------------
class SlotBO:
    def __init__(
            self,
            df: pd.DataFrame,
            essentials: List[str],
            additives: List[str],
            ranges: Dict[str, Tuple[float, float]],
            k_slots: int,
            target_col: str,
            device: str = "auto",
            seed: int = 42,
    ):
        self.E = list(essentials)
        self.adds = list(additives)
        self.k = int(k_slots)
        self.target_col = target_col
        self.seed = seed

        # set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        set_seeds(self.seed)

        # prepare data and encoder
        rows = df.to_dict('records')
        self.codec = SlotCodec(self.E, self.adds, ranges, self.k)
        Z_np = self.codec.encode(rows)
        y_np = _ensure_2d_y(df[self.target_col].values)

        unique_Z, unique_indices, inverse_indices = np.unique(
            Z_np, axis=0, return_index=True, return_inverse=True
        )

        if len(unique_Z) < len(Z_np):
            print(f"[Warning] Found {len(Z_np) - len(unique_Z)} duplicate rows in encoded data. Merging them.")
            unique_y = np.array([
                np.mean(y_np[inverse_indices == i]) for i in range(len(unique_Z))
            ])
            Z_np = unique_Z
            y_np = _ensure_2d_y(unique_y)

        self.Z = torch.tensor(Z_np, dtype=torch.double, device=self.device)
        self.y = torch.tensor(y_np, dtype=torch.double, device=self.device)

        # fit gp
        # [e0,e1,e2, id1,r1, id2,r2, id3,r3], categorical variable id 3, 5, 7 ...
        self.cat_dims = list(range(3, 3 + 2 * self.k, 2))
        fit_kwargs = {"seed": self.seed, "cat_dims": self.cat_dims}

        print("Fitting GP...")
        self.model = fit_gp(self.Z, self.y, **fit_kwargs)
        print("GP fitted")

        self.bounds = self.codec.get_bounds(Z_np, self.device)
        self._supports_gpish = isinstance(self.model, (SingleTaskGP, MixedSingleTaskGP))

    def _make_acqf(self, acq_type: str, **opts: Any):
        t = acq_type.lower()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        best_f = self.y.max().item()
        if t in ("qlognei", "lognei"):
            return qLogNoisyExpectedImprovement(self.model, X_baseline=self.Z, sampler=sampler)
        if t in ("qei", "ei"):
            return qLogExpectedImprovement(self.model, best_f=best_f)
        if t in ("ucb", "qucb"):
            return qUpperConfidenceBound(self.model, beta=opts.get("beta", 0.2))
        if t in ("kg", "qkg", "knowledge_gradient"):
            return qKnowledgeGradient(self.model, num_fantasies=opts.get("kg_num_fantasies", 64))

        raise ValueError(f"Unknown Acquisition Function: '{acq_type}'")

    # continuous relaxation
    def ask(self, q: int = 8, num_restarts: int = 20, raw_samples: int = 512,
            acq_types: List[str] | None = None, acq_options: Dict | None = None) -> Tuple[
        List[Dict[str, float]], np.ndarray, np.ndarray]:
        set_seeds(self.seed)

        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}

        # 如果有多个采集函数，确保每个生成的候选点数量合理
        q_per_acq = max(1, math.ceil(q / len(acq_types)))
        all_candidates_tensors = []

        print(f"\nGenerating candidates using {acq_types}...")
        for acq_name in acq_types:
            # 1. 创建采集函数 (和原来一样)
            acqf = self._make_acqf(acq_name, **acq_options)

            # 2. 调用优化器寻找最佳的 "点集"
            # 对于 qKG，candidates 已经是我们想要的那个最优组合了
            candidates, acq_values = optimize_acqf(
                acq_function=acqf,
                bounds=self.bounds,
                q=q_per_acq,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=False,
                post_processing_func=self.codec.postprocess_batch,
            )
            all_candidates_tensors.append(candidates)

        if not all_candidates_tensors:
            return [], np.array([])

        final_candidates_tensor = torch.cat(all_candidates_tensors, dim=0)
        final_candidates_tensor = final_candidates_tensor[:q]

        # --- 新增代码：获取预测的均值和标准差 ---
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            posterior = self.model.posterior(final_candidates_tensor)
            mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
            variance = posterior.variance.squeeze(-1).detach().cpu().numpy()
            sigma = np.sqrt(variance)
        # --- 结束新增代码 ---

        final_rows = self.codec.decode(final_candidates_tensor.detach().cpu().numpy())

        # 返回均值和标准差
        return final_rows, mean, sigma

    # # totally categorical
    # def ask(self, q: int = 8, num_restarts: int = 20, raw_samples: int = 512,
    #         acq_types: List[str] | None = None, acq_options: Dict | None = None) -> Tuple[
    #     List[Dict[str, float]], np.ndarray]:
    #
    #     set_seeds(self.seed)
    #     acq_types = acq_types or ["qlognei"]
    #     acq_options = acq_options or {}
    #     q_per_acq = max(1, math.ceil(q / len(acq_types)))
    #     all_candidates_tensors = []
    #     canonical_combinations = []
    #     additive_ids = range(1, self.codec.A + 1)
    #     for num_additives in range(self.k + 1):
    #         for combo in combinations(additive_ids, num_additives):
    #             padded_combo = combo + (0,) * (self.k - num_additives)
    #             canonical_combinations.append(padded_combo)
    #     print(f"[info] Generated {len(canonical_combinations)} unique additive combinations for optimization")
    #
    #     fixed_features_list = []
    #     for combo in canonical_combinations:
    #         fixed_features = {}
    #         for i, additive_id in enumerate(combo):
    #             id_dim_index = self.cat_dims[i]
    #             ratio_dim_index = id_dim_index + 1
    #             fixed_features[id_dim_index] = float(additive_id)
    #             if additive_id == 0:
    #                 fixed_features[ratio_dim_index] = 0.0
    #         fixed_features_list.append(fixed_features)
    #
    #     for acq_name in acq_types:
    #         acqf = self._make_acqf(acq_name, **acq_options)
    #
    #         candidates, _ = optimize_acqf_mixed(
    #             acq_function=acqf,
    #             bounds=self.bounds,
    #             q=q_per_acq,
    #             fixed_features_list=fixed_features_list,
    #             num_restarts=num_restarts,
    #             raw_samples=raw_samples,
    #             options={"batch_limit": 5, "maxiter": 200},
    #         )
    #         candidates = self.codec.postprocess_batch(candidates)
    #         all_candidates_tensors.append(candidates)
    #
    #     if not all_candidates_tensors:
    #         return [], np.array([])
    #
    #     all_candidates = torch.cat(all_candidates_tensors, dim=0)
    #     all_rows = self.codec.decode(all_candidates.detach().cpu().numpy())
    #     seen_keys, unique_rows, unique_tensors = set(), [], []
    #     for r, t in zip(all_rows, all_candidates):
    #         z = self.codec.encode_row(r)
    #         key = tuple(np.round(z, 6))
    #         if key not in seen_keys:
    #             seen_keys.add(key)
    #             unique_rows.append(r)
    #             unique_tensors.append(t)
    #     if not unique_rows:
    #         return [], np.array([])
    #     unique_candidates_tensor = torch.stack(unique_tensors)
    #     final_acqf = self._make_acqf(acq_types[-1], **acq_options)
    #     with torch.no_grad():
    #         individual_acq_vals = final_acqf(unique_candidates_tensor.unsqueeze(1))
    #     individual_acq_vals = individual_acq_vals.detach().cpu().numpy()
    #     sorted_results = sorted(zip(unique_rows, individual_acq_vals), key=lambda x: x[1], reverse=True)
    #     final_rows = [r for r, v in sorted_results[:q]]
    #     final_vals = np.array([v for r, v in sorted_results[:q]])
    #     return final_rows, final_vals


def _setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian optimization script based on slot encoding and GP")
    # --- base setting ---
    # MODIFIED: Changed --csv to --input for generality
    parser.add_argument("--input", type=str, default="data.xlsx",
                        help="Path to the historical data file (.csv or .xlsx)")
    parser.add_argument("--target", type=str, default="auc", help="Column name for the target variable")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="device")
    # --- BO setting ---
    parser.add_argument("--slots", type=int, default=3, help="# of slots (k)")
    parser.add_argument("--q", type=int, default=DEFAULT_Q, help="# of candidates per BO round")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_CSV)
    # --- AF setting ---
    parser.add_argument("--acq", type=str, default="qei", help="Acquisition Function: qlognei, qei, qucb, qkg")
    parser.add_argument("--acq_opts", type=str, default="{}")
    return parser


# MODIFIED: This function now handles both .csv and .xlsx files
def _prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Tuple[float, float]]]:
    path = Path(file_path)
    if not path.exists():
        # Fallback to check for default files if the specified path doesn't exist
        alt_path_xlsx = Path("./data.xlsx")
        alt_path_csv = Path("./data.csv")
        if alt_path_xlsx.exists():
            print(f"[info] {path} not found, using alternative path {alt_path_xlsx}")
            path = alt_path_xlsx
        elif alt_path_csv.exists():
            print(f"[info] {path} not found, using alternative path {alt_path_csv}")
            path = alt_path_csv
        else:
            raise FileNotFoundError(f"Data file does not exist: {path.resolve()}")

    # Read file based on its extension
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: '{path.suffix}'. Please use .csv or .xlsx files.")

    df.columns = df.columns.str.lower()

    present_cols = [c for c in FEATURE_WHITELIST if c in df.columns]
    assert all(e in present_cols for e in ESSENTIALS), f"essentials {ESSENTIALS} must be included in columns"
    additives = [c for c in present_cols if c not in ESSENTIALS]
    ranges: Dict[str, Tuple[float, float]] = {}
    for a in additives:
        s = df[a].astype(float).values
        lo = float(np.nanmin(s))
        hi = float(np.nanmax(s))
        ranges[a] = (lo, hi if hi > lo else lo + 1e-6)
    return df, ESSENTIALS, additives, ranges


def _save_results(rows: List[Dict[str, float]], means: np.ndarray, sigmas: np.ndarray, essentials: List[str],
                  out_path: str):
    if not rows:
        print("[warning] no candidate is generated")
        return

    out_df = pd.DataFrame(rows).fillna(0.0)
    out_df['pred_mean'] = means
    out_df['pred_sigma'] = sigmas
    add_counts = Counter(k for r in rows for k in r if k not in essentials and r.get(k, 0.0) > EPS)
    sorted_adds = [k for k, _ in sorted(add_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    all_cols = essentials + sorted_adds + ['pred_mean', 'pred_sigma']
    for col in all_cols:
        if col not in out_df:
            out_df[col] = 0.0
    out_df = out_df[all_cols]
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[success] {len(rows)} candidates are written into -> {Path(out_path).resolve()}")
    print(f"[info] column rank: {all_cols}")
    if add_counts:
        print("[info] additives used counter:", ", ".join(f"{k}:{v}" for k, v in add_counts.most_common()))


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    set_seeds(args.seed)

    # MODIFIED: Use args.input instead of args.csv
    df, essentials, additives, ranges = _prepare_data(args.input)

    bo = SlotBO(
        df=df,
        essentials=essentials,
        additives=additives,
        ranges=ranges,
        k_slots=args.slots,
        target_col=args.target,
        device=args.device,
        seed=args.seed,
    )
    acq_types = [s.strip() for s in args.acq.split(",") if s.strip()]
    try:
        acq_opts = json.loads(args.acq_opts)
        assert isinstance(acq_opts, dict)
    except (json.JSONDecodeError, AssertionError):
        raise ValueError(f"invalid --acq_opts")

    print(f"\n Currently using {acq_types} acquisition function (option: {acq_opts}) generating {args.q} candidates...")
    rows, means, sigmas = bo.ask(
        q=args.q,
        acq_types=acq_types,
        acq_options=acq_opts,
    )

    _save_results(rows, means, sigmas, essentials, args.out)

    print(f"\n[Settings]")
    print(f"  - device: {bo.device}, model: GP (Mixed)")
    print(f"  - # of slots: {args.slots}, random seed: {args.seed}")


if __name__ == "__main__":
    main()