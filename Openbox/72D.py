# ==================== 0. import & 全局 ====================
import logging
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from ConfigSpace import (ConfigurationSpace, CategoricalHyperparameter,
                         UniformFloatHyperparameter, EqualsCondition,
                         ForbiddenAndConjunction, ForbiddenEqualsClause)
from openbox import Advisor, Observation
from openbox.utils.config_space import Configuration

# 固定随机种子，方便复现
np.random.seed(42)

logging.basicConfig(level=logging.INFO)

ESSENTIALS = ["tmb", "hrp", "h2o2"]          # 3 个必需组分
N_SLOT      = 3                              # 最多 3 种 additive
df = pd.read_csv("data.csv")

# 自动找出 22 个 additive 名称
ADDITIVES = [c[:-3] for c in df.columns if c.endswith("_sw")]

# ==================== 1. 构造 ConfigSpace (slot 版) ====================
cs = ConfigurationSpace(seed=42)
choices = ["none"] + ADDITIVES

# 1.1 essentials 连续超参
for e in ESSENTIALS:
    lo, hi = float(df[e].min()), float(df[e].max())
    cs.add_hyperparameter(
        UniformFloatHyperparameter(e, lower=max(lo, 1e-12), upper=hi)
    )

# 1.2 三个 slot，每个 slot 选一种 additive 或 "none"
slot_cat = {}
for i in range(1, N_SLOT + 1):
    cat = CategoricalHyperparameter(f"slot{i}", choices)
    cs.add_hyperparameter(cat)
    slot_cat[i] = cat

    # 给每个 slot-additive 组合建立一个 amount 连续变量
    for a in ADDITIVES:
        col = f"{a}_lvl"
        hi = float(df[col].max()) if col in df else 1e-6
        hp = UniformFloatHyperparameter(
            f"amt_s{i}_{a}", lower=0.0, upper=hi)
        cs.add_hyperparameter(hp)
        cs.add_condition(EqualsCondition(hp, cat, a))

# 1.3 禁止同一种 additive 出现在多个 slot
for a in ADDITIVES:
    for i in range(1, N_SLOT):
        for j in range(i + 1, N_SLOT + 1):
            cs.add_forbidden_clause(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(slot_cat[i], a),
                    ForbiddenEqualsClause(slot_cat[j], a),
                )
            )

# ==================== 2. 辅助函数 ====================
def _pairs_from_dict(d):
    """抽取 (additive, amount) 排序后的列表"""
    return sorted(
        [(d[f"slot{i}"], d[f"amt_s{i}_{d[f'slot{i}']}"])
         for i in range(1, N_SLOT + 1) if d[f"slot{i}"] != "none"],
        key=lambda x: x[0]
    )

def canonicalize(cfg: Configuration) -> Configuration:
    """将配置按 additive 名字排序写回 slot1-3，消除排列对称性"""
    d_old = cfg.get_dictionary()
    d_new = {e: d_old[e] for e in ESSENTIALS}

    pairs = _pairs_from_dict(d_old)
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    for i, (a, amt) in enumerate(pairs, 1):
        d_new[f"slot{i}"] = a
        if a != "none":
            d_new[f"amt_s{i}_{a}"] = amt
    return Configuration(cs, values=d_new)

def key_hash(cfg: Configuration):
    """用于去重的 tuple-key"""
    return tuple((a, round(amt, 6)) for a, amt in _pairs_from_dict(cfg.get_dictionary()))

def decode(cfg: Configuration):
    """转人类可读 dict"""
    d = cfg.get_dictionary()
    r = {e: d[e] for e in ESSENTIALS}
    for i in range(1, N_SLOT + 1):
        a = d[f"slot{i}"]
        if a != "none":
            r[a] = d[f"amt_s{i}_{a}"]
    return r

def pick_diverse(pool, k):
    """max-min 贪心：从 pool 里选 k 个互相最远的 cfg"""
    arrs = np.array([c.get_array() for c in pool])
    sel_idx = [0]                       # 先选第一个
    while len(sel_idx) < k and len(sel_idx) < len(pool):
        d_min = np.min(
            np.linalg.norm(arrs[:, None, :] - arrs[sel_idx][None, :, :], axis=-1),
            axis=1
        )  # 每个点到已选集合的最近距离
        d_min[sel_idx] = -1             # 已选点不再考虑
        sel_idx.append(int(d_min.argmax()))
    return [pool[i] for i in sel_idx[:k]]

# ==================== 3. 初始化 Advisor ====================
ref_point = [
    -df["I_max"].min()  * 0.9,
    -df["tail_pct"].min()* 0.9,
     df["t90"].max()    * 1.1,
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

# ——可选：把 acq_optimizer 的采样点数从 5000 降到 3000——
advisor.acq_optimizer.maximize = partial(advisor.acq_optimizer.maximize, num_points=3000)

# ==================== 4. 喂入历史数据（先 canonical） ====================
for _, row in df.iterrows():
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

    cfg_can = canonicalize(Configuration(cs, values=cfg_dict))
    advisor.update_observation(
        Observation(
            config=cfg_can,
            objectives=[-row.I_max, -row.tail_pct, row.t90],
        )
    )

# ==================== 5. 批量获取 28 条候选 ====================
# 5.1 一次性拿 challenger 列表
challengers = advisor.get_suggestion(return_list=True)
print(f"advisor 返回 {len(challengers)} 条 challenger")

# 5.2 统一 canonical + 去重
uniq = {}
for c in challengers:
    c_can = canonicalize(c)
    k = key_hash(c_can)
    if k not in uniq:
        uniq[k] = c_can
pool = list(uniq.values())
assert len(pool) >= 28, "候选池太小，可把 num_points 调大"

# 5.3 max-min 贪心挑 28 个最分散的点
cand = pick_diverse(pool, 28)

# ==================== 6. 打印并保存（按 additive 出现次数排序） ====================
rows = []
# 初始化计数器
counts = {a: 0 for a in ADDITIVES}

print(f"\n=== 下一轮 28 条配方 ===")
for idx, cfg in enumerate(cand, 1):
    # 解码为人类可读的配方字典
    r = decode(cfg)
    rows.append(r)
    # 本次配方中用到的 additive 列表
    adds = [k for k in r if k not in ESSENTIALS]
    # 累加计数
    for a in adds:
        counts[a] += 1
    print(f"{idx:02d}: {r}   -> additives used: {adds}")

# 按出现次数从多到少排序 additive（仅保留出现过的）
sorted_adds = sorted(
    [a for a, cnt in counts.items() if cnt > 0],
    key=lambda x: counts[x],
    reverse=True
)

# 构造最终的列顺序：先是 ESSENTIALS，再是按频次排序的 additives
final_cols = ESSENTIALS + sorted_adds

# 写入 CSV
out_path = Path("72d.csv")
df_out = pd.DataFrame(rows).fillna(0)
# 重新排列列顺序
df_out = df_out[final_cols]
df_out.to_csv(out_path, index=False)

print(f"\n候选配方已写入 {out_path.resolve()} （additives 列已按选中次数从多到少排序）")
