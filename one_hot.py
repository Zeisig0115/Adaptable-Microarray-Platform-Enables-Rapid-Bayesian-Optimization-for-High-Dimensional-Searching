# ===== 0. import & 全局 =====
import logging
from pathlib import Path

import pandas as pd
from ConfigSpace import (
    ConfigurationSpace, CategoricalHyperparameter,
    UniformFloatHyperparameter, EqualsCondition
)
from openbox import Advisor, Observation
from openbox.utils.config_space import Configuration

import random, numpy as np
random.seed(42)
np.random.seed(42)


logging.basicConfig(level=logging.INFO)

ESSENTIALS = ["tmb", "hrp", "h2o2"]     # 3 个必需连续组分
K_MAX = 3                               # ≤ K 个 additive
df = pd.read_csv("data.csv")

# 自动找 22 个 additive 名称（列名格式: "xxx_sw", "xxx_lvl"）
ADDITIVES = [c[:-3] for c in df.columns if c.endswith("_sw")]

# ===== 1. 构造 ConfigSpace（one-hot + 条件金额） =====
cs = ConfigurationSpace(seed=42)

# 1.1 essentials
for e in ESSENTIALS:
    lo, hi = float(df[e].min()), float(df[e].max())
    cs.add_hyperparameter(
        UniformFloatHyperparameter(e, lower=max(lo, 1e-12), upper=hi)
    )

# 1.2 每个 additive：布尔开关 + 条件浓度
for a in ADDITIVES:
    # 开关：0 未选，1 选
    sw = CategoricalHyperparameter(f"{a}_sw", [0, 1])
    cs.add_hyperparameter(sw)

    # 浓度：仅 sw == 1 时激活
    lo, hi = float(df[f"{a}_lvl"].min()), float(df[f"{a}_lvl"].max())
    lvl = UniformFloatHyperparameter(f"{a}_lvl", lower=lo, upper=hi)
    cs.add_hyperparameter(lvl)
    cs.add_condition(EqualsCondition(lvl, sw, 1))

# ===== 2. 辅助函数 =====
def decode(cfg: Configuration):
    """转成人类可读配方 dict；含 essentials 与被选 additive:amount"""
    d = cfg.get_dictionary()
    recipe = {e: d[e] for e in ESSENTIALS}
    for a in ADDITIVES:
        if d[f"{a}_sw"] == 1:
            recipe[a] = d[f"{a}_lvl"]
    return recipe

def n_additives(cfg: Configuration) -> int:
    """统计选中的 additive 个数"""
    d = cfg.get_dictionary()
    return sum(d[f"{a}_sw"] for a in ADDITIVES)

# ===== 3. 初始化 Advisor (树 surrogate + EHVI) =====
ref_point = [
    -df["I_max"].min() * 0.95,
    -df["tail_pct"].min() * 0.95,
    df["t90"].max() * 1.05,
]
advisor = Advisor(
    config_space=cs,
    num_objectives=3,
    surrogate_type="prf",        # Probabilistic Random Forest
    acq_type="ehvi",
    ref_point=ref_point,
    random_state=42,
    task_id="BME_onehot",
)

# ===== 4. 喂入历史数据 =====
for _, row in df.iterrows():
    cfg_dict = {e: float(row[e]) for e in ESSENTIALS}
    for a in ADDITIVES:
        cfg_dict[f"{a}_sw"]  = int(row[f"{a}_sw"])
        if cfg_dict[f"{a}_sw"] == 1:
            cfg_dict[f"{a}_lvl"] = float(row[f"{a}_lvl"])
    advisor.update_observation(
        Observation(
            config=Configuration(cs, values=cfg_dict),
            objectives=[-row.I_max, -row.tail_pct, row.t90],
        )
    )

# ===== 5. 批量获取 28 条候选（带轻量级 reject） =====
BATCH, MAX_TRIES = 28, 10000
cand, tried = [], set()

for _ in range(MAX_TRIES):
    if len(cand) == BATCH:
        break
    cfg = advisor.get_suggestion()
    key = tuple(sorted(
        (a, round(cfg[f"{a}_lvl"], 6))
        for a in ADDITIVES if cfg[f"{a}_sw"] == 1
    ))
    # (a) 避免重复  (b) 基数约束
    if key not in tried and n_additives(cfg) <= K_MAX:
        tried.add(key)
        cand.append(cfg)

else:
    raise RuntimeError(f"Tried {MAX_TRIES} times still not found {BATCH} samples")

rows = []
print(f"\n=== one-hot baseline 下一轮 {BATCH} 条配方 ===")
for idx, cfg in enumerate(cand, 1):
    r = decode(cfg)
    rows.append(r)
    adds = [k for k in r if k not in ESSENTIALS]
    print(f"{idx:02d}: {r}   -> additives used: {adds}")

out_path = Path("onehot.csv")
pd.DataFrame(rows).fillna(0).to_csv(out_path, index=False)
print(f"\nbaseline 候选已写入 {out_path.resolve()}")
