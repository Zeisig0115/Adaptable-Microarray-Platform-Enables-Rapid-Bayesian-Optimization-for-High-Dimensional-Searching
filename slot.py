# ===== 0. import & 全局 =====
import logging
from pathlib import Path

import pandas as pd
from ConfigSpace import (
    ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter,
    EqualsCondition, ForbiddenAndConjunction, ForbiddenEqualsClause
)
from openbox import Advisor, Observation
from openbox.utils.config_space import Configuration

import random, numpy as np
random.seed(42)
np.random.seed(42)


logging.basicConfig(level=logging.INFO)

ESSENTIALS = ["tmb", "hrp", "h2o2"]        # 固定 3 个必需组分
N_SLOT = 3                                 # 最多 3 种 additive
df = pd.read_csv("data.csv")

# 自动找出 22 个 additives 名称
ADDITIVES = [c[:-3] for c in df.columns if c.endswith("_sw")]

# ===== 1. 构造 ConfigSpace（与前版相同） =====
cs = ConfigurationSpace(seed=42)
choices = ["none"] + ADDITIVES

# 1.1 essentials
for e in ESSENTIALS:
    lo, hi = float(df[e].min()), float(df[e].max())
    cs.add_hyperparameter(UniformFloatHyperparameter(e, lower=max(lo, 1e-12), upper=hi))

# 1.2 slots
slot_cat = {}
for i in range(1, N_SLOT + 1):
    cat = CategoricalHyperparameter(f"slot{i}", choices, default_value="none")
    cs.add_hyperparameter(cat)
    slot_cat[i] = cat

    for a in ADDITIVES:
        col = f"{a}_lvl"
        lo, hi = float(df[col].min()), float(df[col].max())
        hp = UniformFloatHyperparameter(f"amt_s{i}_{a}", lower=lo, upper=hi)
        cs.add_hyperparameter(hp)
        cs.add_condition(EqualsCondition(hp, cat, a))

# 1.3 forbid duplicate additive
for a in ADDITIVES:
    for i in range(1, N_SLOT):
        for j in range(i + 1, N_SLOT + 1):
            cs.add_forbidden_clause(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(slot_cat[i], a),
                    ForbiddenEqualsClause(slot_cat[j], a),
                )
            )

# ===== 2. 辅助函数 =====
def _pairs_from_dict(d):
    """从 cfg 字典抽取 (additive, amount) 对并排序"""
    pairs = [
        (d[f"slot{i}"], d[f"amt_s{i}_{d[f'slot{i}']}"])
        for i in range(1, N_SLOT + 1)
        if d[f"slot{i}"] != "none"
    ]
    return sorted(pairs, key=lambda x: x[0])

def canonicalize(cfg: Configuration) -> Configuration:
    """返回 permutation-invariant 的新 Configuration"""
    d_old = cfg.get_dictionary()
    # essentials 保持不动
    d_new = {e: d_old[e] for e in ESSENTIALS}

    # 得到按字典序排好的 pairs，并补 'none'
    pairs = _pairs_from_dict(d_old)
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    # 写回 slot1/2/3
    for i, (a, amt) in enumerate(pairs, 1):
        d_new[f"slot{i}"] = a
        if a != "none":
            d_new[f"amt_s{i}_{a}"] = amt

    return Configuration(cs, values=d_new)

def key_hash(cfg: Configuration):
    """tuple key 用于去重（浮点取 6 位小数）"""
    return tuple(
        (a, round(amt, 6)) for a, amt in _pairs_from_dict(cfg.get_dictionary())
    )

def decode(cfg: Configuration):
    """转成人类可读配方字典"""
    d = cfg.get_dictionary()
    recipe = {e: d[e] for e in ESSENTIALS}
    for i in range(1, N_SLOT + 1):
        a = d[f"slot{i}"]
        if a != "none":
            recipe[a] = d[f"amt_s{i}_{a}"]
    return recipe

# ===== 3. 初始化 Advisor =====
ref_point = [
    -df["I_max"].min() * 0.95,
    -df["tail_pct"].min() * 0.95,
    df["t90"].max() * 1.05,
]
advisor = Advisor(
    config_space=cs,
    num_objectives=3,
    surrogate_type="prf",
    acq_type="ehvi",
    ref_point=ref_point,
    random_state=42,
    task_id="BME_slot",
)

# ===== 4. 喂入历史数据（已 canonical） =====
for _, row in df.iterrows():
    # 收集 (additive, amount)，按 additive 名字排序
    pairs = sorted(
        [(a, row[f"{a}_lvl"]) for a in ADDITIVES if row[f"{a}_sw"] == 1],
        key=lambda x: x[0]
    )[:N_SLOT]
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    cfg_dict = {e: float(row[e]) for e in ESSENTIALS}
    for i, (a, amt) in enumerate(pairs, 1):
        cfg_dict[f"slot{i}"] = a
        if a != "none":
            cfg_dict[f"amt_s{i}_{a}"] = float(amt)

    advisor.update_observation(
        Observation(
            config=Configuration(cs, values=cfg_dict),
            objectives=[-row.I_max, -row.tail_pct, row.t90],
        )
    )

# ===== 5. 批量获取 28 条全新候选 =====
BATCH = 28
cand, tried = [], set()
MAX_TRIES = 10000

for _ in range(MAX_TRIES):
    if len(cand) == BATCH:
        break
    cfg_raw = advisor.get_suggestion()
    cfg = canonicalize(cfg_raw)
    k = key_hash(cfg)
    if k not in tried:
        tried.add(k)
        cand.append(cfg)
else:
    raise RuntimeError(f"Tried {MAX_TRIES} times still not found {BATCH} samples")

# ===== 6. 打印并保存 =====
rows = []
print(f"\n=== 下一轮 {BATCH} 条配方 ===")
for idx, cfg in enumerate(cand, 1):
    r = decode(cfg)
    rows.append(r)
    additives_used = [k for k in r if k not in ESSENTIALS]
    print(f"{idx:02d}: {r}   -> additives used: {additives_used}")

out_path = Path("slot.csv")
pd.DataFrame(rows).fillna(0).to_csv(out_path, index=False)
print(f"\n配方已写入 {out_path.resolve()}")
