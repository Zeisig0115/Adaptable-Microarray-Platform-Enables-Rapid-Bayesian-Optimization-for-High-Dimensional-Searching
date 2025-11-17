from __future__ import annotations
import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch

from fit_model import (
    fit_gp, fit_saasbo, fit_fullyb_gp, fit_saas_gp
)

try:
    from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement, qUpperConfidenceBound, \
        qKnowledgeGradient, ProbabilityOfImprovement, qNoisyExpectedImprovement

    _HAS_QNEI = True
except ImportError:
    from botorch.acquisition.monte_carlo import qLogExpectedImprovement, qLogNoisyExpectedImprovement, \
        qUpperConfidenceBound
    from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
    from botorch.acquisition.analytic import ProbabilityOfImprovement

    _HAS_QNEI = False
    qNoisyExpectedImprovement = qLogNoisyExpectedImprovement

from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

try:
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP, FullyBayesianSingleTaskGP
except ImportError:
    SaasFullyBayesianSingleTaskGP, FullyBayesianSingleTaskGP = SingleTaskGP, SingleTaskGP

torch.set_default_dtype(torch.double)

# ----------------- 常量 -----------------
ESSENTIALS = ["HRP", "TMB", "H2O2"]
DEFAULT_Q = 96


# ----------------- 工具函数 (保持不变) -----------------
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_2d_y(y: np.ndarray | List[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


class EssentialsBO:
    MODEL_FITTERS = {
        "gp": fit_gp, "saasbo": fit_saasbo, "fullyb_gp": fit_fullyb_gp,
        "saas_gp": fit_saas_gp
    }

    def __init__(
            self,
            df: pd.DataFrame,
            essentials: List[str],
            target_col: str,
            surrogate: str = "gp",
            device: str = "auto",
            seed: int = 42,
            nuts_cfg: Dict[str, int] | None = None,
    ):
        self.E = list(essentials)
        self.target_col = target_col
        self.surrogate = surrogate.lower()
        self.nuts_cfg = nuts_cfg or {}
        self.seed = seed


        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        set_seeds(self.seed)


        Z_np = df[self.E].values
        y_np = _ensure_2d_y(df[self.target_col].values)

        self.Z = torch.tensor(Z_np, dtype=torch.double, device=self.device)
        self.y = torch.tensor(y_np, dtype=torch.double, device=self.device)

        fitter = self.MODEL_FITTERS.get(self.surrogate)
        fit_kwargs = {"seed": self.seed}
        if self.surrogate in ("saasbo", "fullyb_gp"):
            fit_kwargs.update(self.nuts_cfg)
        print(f"正在拟合模型 '{self.surrogate}'...")
        self.model = fitter(self.Z, self.y, **fit_kwargs)
        print("模型拟合完成。")

        bounds_min = self.Z.min(dim=0).values
        bounds_max = self.Z.max(dim=0).values
        self.bounds = torch.stack([bounds_min, bounds_max])

    def _make_acqf(self, acq_type: str, **opts: Any):
        t = acq_type.lower()
        if not isinstance(self.model, (SingleTaskGP, SaasFullyBayesianSingleTaskGP, FullyBayesianSingleTaskGP)):
            if t in ("kg", "qkg", "qlognei", "qnei"):
                print(f"[警告] 模型'{self.surrogate}'不支持'{t}'，自动降级为'qei'。")
                t = "qei"
            elif t in ("pi", "poi"):
                print(f"[警告] 模型'{self.surrogate}'不支持'pi'，自动降级为'ucb'。")
                t = "ucb"

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        best_f = self.y.max().item()

        if t in ("qlognei", "lognei"):
            return qLogNoisyExpectedImprovement(self.model, X_baseline=self.Z, sampler=sampler)
        if t in ("qei", "ei"):
            return qLogExpectedImprovement(self.model, best_f=best_f)
        if t in ("ucb", "qucb"):
            return qUpperConfidenceBound(self.model, beta=opts.get("beta", 0.2))
        if t in ("kg", "qkg"):
            return qKnowledgeGradient(self.model, num_fantasies=opts.get("kg_num_fantasies", 64))
        if t in ("pi", "poi"):
            return ProbabilityOfImprovement(self.model, best_f=opts.get("best_f", best_f))
        if t in ("qnei", "nei") and _HAS_QNEI:
            return qNoisyExpectedImprovement(self.model, X_baseline=self.Z, sampler=sampler)
        raise ValueError(f"未知或不支持的采集函数: '{acq_type}'")

    # ----- 【核心修改 1】 -----
    def ask(self, q: int = 8, num_restarts: int = 20, raw_samples: int = 512,
            acq_types: List[str] | None = None, acq_options: Dict | None = None) -> Tuple[
        List[Dict[str, float]], np.ndarray, np.ndarray]:
        """
        生成候选点并返回它们的坐标、预测均值和预测标准差。
        """
        set_seeds(self.seed)
        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}

        acqf = self._make_acqf(acq_types[0], **acq_options)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples
        )

        with torch.no_grad():
            posterior = self.model.posterior(candidates)
            # 获取预测均值
            predicted_means = posterior.mean.squeeze(-1)
            # 获取预测方差并计算标准差
            predicted_variance = posterior.variance.squeeze(-1)
            predicted_stddev = torch.sqrt(predicted_variance)

        # 转换为Numpy数组
        candidates_np = candidates.detach().cpu().numpy()
        predicted_means_np = predicted_means.detach().cpu().numpy()
        predicted_stddev_np = predicted_stddev.detach().cpu().numpy()

        rows = [dict(zip(self.E, c)) for c in candidates_np]

        # 返回三个值：坐标，预测均值，预测标准差
        return rows, predicted_means_np, predicted_stddev_np


def _setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="必需品浓度组合的贝叶斯优化脚本")
    parser.add_argument("--datafile", type=str, default="data.xlsx", help="包含历史数据的CSV或Excel文件路径")
    parser.add_argument("--target", type=str, default="AUC", help="文件中的目标列名")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="计算设备")
    parser.add_argument(
        "--model", type=str, default="gp",
        choices=["gp", "saasbo", "fullyb_gp", "saas_gp"],
        help="选择代理模型"
    )
    parser.add_argument("--q", type=int, default=DEFAULT_Q, help="每次迭代推荐的候选点数量")
    parser.add_argument(
        "--acq", type=str, default="qlognei",
        help="采集函数, 多个用逗号分隔 (例如 'qei,ucb')。常用: qlognei, qei, qucb, qkg, qnei"
    )
    parser.add_argument(
        "--acq_opts", type=str, default="{}",
        help='采集函数选项的JSON字符串, 例如 \'{"beta":0.3}\''
    )
    parser.add_argument("--nuts_warmup", type=int, default=512, help="NUTS预热步数")
    parser.add_argument("--nuts_samples", type=int, default=256, help="NUTS采样数")
    parser.add_argument("--nuts_thinning", type=int, default=16, help="NUTS thinning参数")
    return parser


def _prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path.resolve()}")

    file_suffix = path.suffix.lower()
    if file_suffix == '.csv':
        df = pd.read_csv(path)
    elif file_suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: '{file_suffix}'。请使用 .csv 或 .xlsx 文件。")

    assert all(e in df.columns for e in ESSENTIALS), \
        f"必需组分 {ESSENTIALS} 必须全部存在于数据文件列中。"

    return df, ESSENTIALS


def _save_results(
    rows: List[Dict[str, float]],
    predicted_values: np.ndarray,
    uncertainties: np.ndarray,  # <--- 新增参数
    essentials: List[str],
    out_path: str
):
    """将候选点及其模型预测值和不确定性保存到CSV文件。"""
    if not rows:
        print("[警告] 没有生成任何候选点，不创建输出文件。")
        return

    out_df = pd.DataFrame(rows)
    out_df['predicted_value'] = predicted_values
    out_df['uncertainty_std'] = uncertainties  # <--- 新增列

    # 更新列顺序
    column_order = essentials + ['predicted_value', 'uncertainty_std']
    out_df = out_df[column_order]

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[成功] {len(rows)}个候选点及其预测值和不确定性已写入 -> {Path(out_path).resolve()}")


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    set_seeds(args.seed)

    df, essentials = _prepare_data(args.datafile)
    assert args.target in df.columns, f"目标列 '{args.target}' 在文件中不存在。"

    bo = EssentialsBO(
        df=df,
        essentials=essentials,
        target_col=args.target,
        surrogate=args.model,
        device=args.device,
        seed=args.seed,
        nuts_cfg={"warmup": args.nuts_warmup, "num_samples": args.nuts_samples, "thinning": args.nuts_thinning},
    )

    acq_types = [s.strip() for s in args.acq.split(",") if s.strip()]
    try:
        acq_opts = json.loads(args.acq_opts)
        assert isinstance(acq_opts, dict)
    except (json.JSONDecodeError, AssertionError):
        raise ValueError(f"无效的 --acq_opts。必须是JSON对象字符串。")

    print(f"\n正在使用采集函数 {acq_types} (选项: {acq_opts}) 生成 {args.q} 个候选点...")

    # ----- 【核心修改 3】 -----
    # ask 现在返回三个值
    rows, predicted_values, uncertainties = bo.ask(
        q=args.q,
        acq_types=acq_types,
        acq_options=acq_opts,
    )

    acq_name = acq_types[0]
    output_filename = f"{args.model}_{acq_name}_seed{args.seed}.csv"
    final_output_path = output_filename

    # 将 uncertainties 传递给保存函数
    _save_results(rows, predicted_values, uncertainties, essentials, final_output_path)

    print(f"\n[配置回顾]")
    print(f"  - 设备: {bo.device}, 模型: {args.model}")
    print(f"  - 随机种子: {args.seed}")


if __name__ == "__main__":
    main()