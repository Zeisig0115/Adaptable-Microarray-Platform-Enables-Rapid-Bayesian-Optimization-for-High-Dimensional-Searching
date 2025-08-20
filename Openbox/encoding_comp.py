# ==========================================================
# 0. imports & 全局
# ==========================================================
import time, logging, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from openbox.core.base import build_surrogate
from openbox.utils.config_space.util import convert_configurations_to_array
from ConfigSpace import (ConfigurationSpace, CategoricalHyperparameter,
                         UniformFloatHyperparameter, EqualsCondition,
                         ForbiddenAndConjunction, ForbiddenEqualsClause,
                         InCondition)
from openbox.utils.config_space import Configuration

logging.basicConfig(level=logging.INFO)
np.random.seed(42)

ESSENTIALS = ["tmb", "hrp", "h2o2"]
N_SLOT     = 3
df         = pd.read_csv("data.csv")
ADDITIVES  = [c[:-3] for c in df.columns if c.endswith("_sw")]
RANGE      = {a: (float(df[f"{a}_lvl"].min()), float(df[f"{a}_lvl"].max()))
              for a in ADDITIVES}
MAX_HI     = max(hi for _, hi in RANGE.values())

# ----------------------------------------------------------
# 1. ConfigSpace 构造函数
# ----------------------------------------------------------
def build_space_amount():
    cs = ConfigurationSpace(seed=42)
    for e in ESSENTIALS:
        lo, hi = float(df[e].min()), float(df[e].max())
        cs.add_hyperparameter(UniformFloatHyperparameter(e, lo, hi, log=True))

    slot_cat = {}
    for i in range(1, N_SLOT + 1):
        cat = CategoricalHyperparameter(f"slot{i}", ["none"] + ADDITIVES)
        cs.add_hyperparameter(cat)
        slot_cat[i] = cat

        for a in ADDITIVES:
            amt = UniformFloatHyperparameter(f"amt_s{i}_{a}", *RANGE[a])
            cs.add_hyperparameter(amt)
            cs.add_condition(EqualsCondition(amt, cat, a))

    for a in ADDITIVES:
        for i in range(1, N_SLOT):
            for j in range(i + 1, N_SLOT + 1):
                cs.add_forbidden_clause(
                    ForbiddenAndConjunction(
                        ForbiddenEqualsClause(slot_cat[i], a),
                        ForbiddenEqualsClause(slot_cat[j], a)))
    return cs

def build_space_ratio():
    cs = ConfigurationSpace(seed=42)
    for e in ESSENTIALS:
        lo, hi = float(df[e].min()), float(df[e].max())
        cs.add_hyperparameter(UniformFloatHyperparameter(e, lo, hi, log=True))

    slot_cat = {}
    for i in range(1, N_SLOT + 1):
        cat = CategoricalHyperparameter(f"slot{i}", ["none"] + ADDITIVES,
                                         default_value="none")
        cs.add_hyperparameter(cat)
        slot_cat[i] = cat
        ratio = UniformFloatHyperparameter(f"ratio{i}", 0.0, 1.0, default_value=0.0)
        cs.add_hyperparameter(ratio)
        cs.add_condition(InCondition(ratio, cat, ADDITIVES))

    for a in ADDITIVES:
        for i in range(1, N_SLOT):
            for j in range(i + 1, N_SLOT + 1):
                cs.add_forbidden_clause(
                    ForbiddenAndConjunction(
                        ForbiddenEqualsClause(slot_cat[i], a),
                        ForbiddenEqualsClause(slot_cat[j], a)))
    return cs

cs72 = build_space_amount()
cs9  = build_space_ratio()

# ----------------------------------------------------------
# 2. 编码函数（72D / 9D）+ canonicalize
# ----------------------------------------------------------
def canonicalize(cfg, cs, use_ratio):
    d_old = cfg.get_dictionary()  # 先拿到原 dict
    d_new = {e: d_old[e] for e in ESSENTIALS}  # 再构造新的子字典

    getter = (lambda i: d_old.get(f"ratio{i}", 0.0)) if use_ratio else \
             (lambda i, a: d_old.get(f"amt_s{i}_{a}", 0.0))

    pairs = []
    for i in range(1, N_SLOT + 1):
        a = d_old[f"slot{i}"]
        if a != "none":
            pairs.append((a, getter(i, a) if not use_ratio else getter(i)))
    pairs.sort()
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    for i, (a, v) in enumerate(pairs, 1):
        d_new[f"slot{i}"] = a
        if a != "none":
            key = f"ratio{i}" if use_ratio else f"amt_s{i}_{a}"
            d_new[key] = v
    return Configuration(cs, values=d_new)

def encode_row_to_cfg(row, cs, use_ratio):
    pairs = [(a, row[f"{a}_lvl"]) for a in ADDITIVES if row[f"{a}_sw"] == 1]
    pairs.sort(); pairs = pairs[:N_SLOT]
    pairs += [("none", 0.0)] * (N_SLOT - len(pairs))

    d = {e: float(row[e]) for e in ESSENTIALS}
    for i, (a, lvl) in enumerate(pairs, 1):
        d[f"slot{i}"] = a
        if a != "none":
            if use_ratio:
                lo, hi = RANGE[a]; ratio = (lvl - lo) / max(hi - lo, 1e-12)
                d[f"ratio{i}"] = ratio
            else:
                d[f"amt_s{i}_{a}"] = lvl
    return canonicalize(Configuration(cs, values=d), cs, use_ratio)

# ----------------------------------------------------------
# 3. 把历史映射成两组配置 & target
# ----------------------------------------------------------
configs72, configs9, Y = [], [], []
for _, row in df.iterrows():
    configs72.append(encode_row_to_cfg(row, cs72, use_ratio=False))
    configs9 .append(encode_row_to_cfg(row, cs9 , use_ratio=True ))
    Y.append([-row['I_max'], -row['I10'], -row['tail']])
Y = np.asarray(Y, float)

# ----------------------------------------------------------
# 4. 5-fold CV 对比
# ----------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
stat = {k: [] for k in ['rmse72','rmse9','r272','r29','t72','t9']}

for fold, (tr, te) in enumerate(kf.split(configs72), 1):
    X72_tr = convert_configurations_to_array([configs72[i] for i in tr])
    X72_te = convert_configurations_to_array([configs72[i] for i in te])
    X9_tr  = convert_configurations_to_array([configs9 [i] for i in tr])
    X9_te  = convert_configurations_to_array([configs9 [i] for i in te])

    pred72, pred9 = np.zeros_like(Y[te]), np.zeros_like(Y[te])

    for j in range(3):
        for tag, cs_, Xtr, Xte, pred in \
            [('72', cs72, X72_tr, X72_te, pred72),
             ('9' , cs9 , X9_tr , X9_te , pred9 )]:
            t0 = time.time()
            model = build_surrogate('prf', config_space=cs_,
                                    rng=np.random.RandomState(fold*100+j))
            model.train(Xtr, Y[tr, j])
            mu, _ = model.predict(Xte)
            pred[:, j] = mu.ravel()
            if j == 0:  # 只记录一次时间
                stat[f't{tag}'].append(time.time()-t0)

    stat['rmse72'].append(np.sqrt(mean_squared_error(
        Y[te], pred72, multioutput='raw_values')))
    stat['rmse9'] .append(np.sqrt(mean_squared_error(
        Y[te], pred9 , multioutput='raw_values')))
    stat['r272'] .append(r2_score(Y[te], pred72, multioutput='raw_values'))
    stat['r29']  .append(r2_score(Y[te], pred9 , multioutput='raw_values'))

    print(f"[Fold {fold}]  RMSE72 {stat['rmse72'][-1]}  | RMSE9 {stat['rmse9'][-1]}")

# ----------------------------------------------------------
# 5. 汇总
# ----------------------------------------------------------
def mean_std(arr): return arr.mean(axis=0), arr.std(axis=0)

rmse72_m, rmse72_s  = mean_std(np.vstack(stat['rmse72']))
rmse9_m , rmse9_s   = mean_std(np.vstack(stat['rmse9']))
r272_m , r272_s     = mean_std(np.vstack(stat['r272']))
r29_m  , r29_s      = mean_std(np.vstack(stat['r29']))
t72, t9             = np.mean(stat['t72']), np.mean(stat['t9'])

print("\n=== 5-Fold CV (mean ± std) ===")
for k,name in enumerate(["-I_max","-I10","tail"]):
    print(f"{name:<9}  RMSE72 {rmse72_m[k]:.4f}±{rmse72_s[k]:.4f} | "
          f"RMSE9 {rmse9_m[k]:.4f}±{rmse9_s[k]:.4f} | "
          f"R²72 {r272_m[k]:.3f}±{r272_s[k]:.3f} | "
          f"R²9 {r29_m[k]:.3f}±{r29_s[k]:.3f}")

print(f"\nAverage training time per objective 72D≈{t72*1000:.1f} ms  |  9D≈{t9*1000:.1f} ms")
