# ==========================================================
# 0. Imports & 全局
# ==========================================================
import logging
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from ConfigSpace import (
    ConfigurationSpace, CategoricalHyperparameter,
    UniformFloatHyperparameter, ForbiddenAndConjunction,
    ForbiddenEqualsClause, InCondition
)
from openbox import Advisor, Observation
from openbox.utils.config_space import Configuration

np.random.seed(42)
logging.basicConfig(level=logging.INFO)

ESSENTIALS = ["tmb", "hrp", "h2o2"]
N_SLOT      = 3
df = pd.read_csv("data.csv")
ADDITIVES = [c[:-3] for c in df.columns if c.endswith("_sw")]

# 每种 additive 的历史 (lo, hi)
RANGE = {a: (float(df[f"{a}_lvl"].min()), float(df[f"{a}_lvl"].max()))
         for a in ADDITIVES}

# ==========================================================
# 1. 构造 ConfigurationSpace（ratio+InCondition）
# ==========================================================
cs = ConfigurationSpace(seed=42)
choices = ["none"] + ADDITIVES

# essentials
for e in ESSENTIALS:
    lo, hi = float(df[e].min()), float(df[e].max())
    cs.add_hyperparameter(UniformFloatHyperparameter(e, lo, hi, log=False))

slot_cat = {}
for i in range(1, N_SLOT + 1):
    cat = CategoricalHyperparameter(f"slot{i}", choices, default_value="none")
    cs.add_hyperparameter(cat)
    slot_cat[i] = cat

    ratio = UniformFloatHyperparameter(f"ratio{i}", 0.0, 1.0, default_value=0.0)
    cs.add_hyperparameter(ratio)

    # ratio 仅在 slot 选到非 'none' 值时活跃
    cs.add_condition(InCondition(ratio, cat, ADDITIVES))

# forbid 同 additive 重复
for a in ADDITIVES:
    for i in range(1, N_SLOT):
        for j in range(i + 1, N_SLOT + 1):
            cs.add_forbidden_clause(
                ForbiddenAndConjunction(
                    ForbiddenEqualsClause(slot_cat[i], a),
                    ForbiddenEqualsClause(slot_cat[j], a),
                )
            )

# ---------- TEST-1: 查看超参 ----------
print("\n=== TEST-1: ConfigurationSpace ===")
for hp in cs.get_hyperparameters():
    if hasattr(hp, 'lower'):
        print(f"  {hp.name:<10}: [{hp.lower}, {hp.upper}]")
    else:
        print(f"  {hp.name:<10}: choices={hp.choices}")

# ==========================================================
# 2. decode & canonicalize
# ==========================================================
def decode(cfg: Configuration):
    d = cfg.get_dictionary()
    r = {e: float(d[e]) for e in ESSENTIALS}
    for i in range(1, N_SLOT + 1):
        a = d[f"slot{i}"]
        if a != "none":
            ratio = d.get(f"ratio{i}", 0.0)
            lo, hi = RANGE[a]
            r[a] = lo + ratio * (hi - lo)
    assert all((d[f"slot{i}"]!='none')== (f"ratio{i}" in d) for i in range(1,N_SLOT+1))
    return r

def canonicalize(cfg: Configuration):
    d_old = cfg.get_dictionary()
    d_new = {e: float(d_old[e]) for e in ESSENTIALS}

    pairs = sorted(
        [(d_old[f"slot{i}"], d_old.get(f"ratio{i}", 0.0))
         for i in range(1, N_SLOT + 1)
         if d_old[f"slot{i}"] != "none"],
        key=lambda x: x[0]
    )
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    for i, (a, ratio) in enumerate(pairs, 1):
        d_new[f"slot{i}"] = a
        if a != "none":                   # 只写活跃 ratio
            d_new[f"ratio{i}"] = ratio

    return Configuration(cs, values=d_new)

# ---------- TEST-2: 随机采样、canonicalize、decode ----------
print("\n=== TEST-2: sampling & decode ===")
for cfg in cs.sample_configuration(3):
    can = canonicalize(cfg)
    print(" raw : ", cfg.get_dictionary())
    print(" canon: ", can.get_dictionary())
    print(" dec  : ", decode(can))
    print("---")

# ==========================================================
# 3. 初始化 Advisor
# ==========================================================
ref_point = [
    -df["I_max"].min() * 0.9,    # 目标1: I_max (越大越好)
    -df["I10"].min() * 0.9,      # 目标2: I10 (越大越好, 替换 t90)
    -df["tail"].min() * 0.9,     # 目标3: tail (越大越好, 替换 tail_pct)
]
advisor = Advisor(
    config_space=cs,
    num_objectives=3,
    surrogate_type="prf",
    acq_type="ehvi",
    ref_point=ref_point,
    random_state=42,
    task_id="BME_ratio",
)
advisor.acq_optimizer.maximize = partial(
    advisor.acq_optimizer.maximize, num_points=3000
)

print("\n=== TEST-3: Advisor init ===")
print(" ref_point:", advisor.ref_point)
print(" init_trials:", advisor.init_num)

# ==========================================================
# 4. 喂入历史数据
# ==========================================================
for _, row in df.iterrows():
    pairs = sorted(
        [
            (
                a,
                (row[f"{a}_lvl"] - RANGE[a][0]) / (RANGE[a][1]-RANGE[a][0])
            )
            for a in ADDITIVES if row[f"{a}_sw"] == 1
        ],
        key=lambda x: x[0]
    )[:N_SLOT]
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    cfg_dict = {e: float(row[e]) for e in ESSENTIALS}
    for i, (a, ratio) in enumerate(pairs, 1):
        cfg_dict[f"slot{i}"] = a
        if a != "none":                  # inactive ratio 不写
            cfg_dict[f"ratio{i}"] = ratio

    cfg_can = canonicalize(Configuration(cs, values=cfg_dict))
    advisor.update_observation(
        Observation(
            config=cfg_can,
            objectives=[-row['I_max'], -row['I10'], -row['tail']]
        )
    )

# ---------- TEST-4: History 检查 ----------
print("\n=== TEST-4: History ===")
print("  len :", len(advisor.history))
print("  first 2 objectives rows:\n",
      advisor.history.get_objectives()[:2])


# ==========================================================
# 5. 生成 28 条候选
# ==========================================================
challengers = advisor.get_suggestion(return_list=True)
uniq = {}
for c in challengers:
    can = canonicalize(c)
    key = tuple(
        (can.get_dictionary()[f"slot{i}"],
         round(can.get_dictionary().get(f"ratio{i}", 0.0), 6))
        for i in range(1, N_SLOT + 1)
    )
    uniq[key] = can
pool = list(uniq.values())

def pick_diverse(pool, k):
    arrs = np.stack([c.get_array() for c in pool])
    sel = [0]
    while len(sel) < k and len(sel) < len(pool):
        dmin = np.min(
            np.linalg.norm(arrs[:, None, :] - arrs[sel][None, :, :], axis=-1),
            axis=1
        )
        dmin[sel] = -1
        sel.append(int(dmin.argmax()))
    return [pool[i] for i in sel[:k]]

cands = pick_diverse(pool, 28)

# ---------- TEST-5: 打印前 5 个候选 ----------
print("\n=== TEST-5: candidates ===")
for i, cfg in enumerate(cands[:5], 1):
    print(f"{i:02d}: ", decode(cfg))

# ==========================================================
# 6. 写 CSV
# ==========================================================
rows = [decode(cfg) for cfg in cands]
counts = {a: 0 for a in ADDITIVES}
for r in rows:
    for a in r:
        if a not in ESSENTIALS:
            counts[a] += 1
sorted_adds = sorted(
    [a for a, cnt in counts.items() if cnt > 0],
    key=lambda x: counts[x], reverse=True
)
df_out = pd.DataFrame(rows).fillna(0)[ESSENTIALS + sorted_adds]
out_path = Path("9d_greedy.csv")
df_out.to_csv(out_path, index=False)
print("\n候选配方已写入 ->", out_path.resolve())

