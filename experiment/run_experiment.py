# -*- coding: utf-8 -*-
"""
免疫算法 v1(原论文复现) vs v2(改进版) 对比实验  —— v3
核心变更（相对原论文实现）：
  1. 训练集 12K -> 24K 纯微博（外部反讽样本不入训练集，避免污染极区/记忆原型）
  2. 逃逸得分分布异常项 偏度γ -> 峰度κ（区分效应量 Cohen's d: 0.246 -> 0.460）
  3. MI 筛选真正生效：17维 -> top-12（原实现请求 top-20 而仅17维，等价空操作）
  4. polarity_moments 向量化（CSR 批量，与原始逐行实现数值一致，差 <1.9e-14）
  5. 新增 5折交叉验证、平衡准确率/macro-F1、多随机种子稳健性、外部验证子集
  6. 逃逸得分对照组改为"普通负面"（标准测试集），而非特殊集内其他伪装子类
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from immune_pipeline import (
    SEED, load_data, build_pipeline, v1_features, build_tfidf_svd,
    learn_polarity_regions, polarity_moments, build_memory, build_v2_features,
)

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

WEIBO_SUB = ["irony", "pos_surface", "irony_pos", "synthetic_irony", "netslang", "netslang_pos"]
EXT_SUB = ["sarcasm_neg", "sarcasm_pos"]

# ================= 0. 数据加载 =================
print("[0] 加载数据 ...")
train, std, spc = load_data(BASE)
y_tr, y_std, y_spc = train.label.values, std.label.values, spc.label.values
m_weibo = spc.subtype.isin(WEIBO_SUB).values
m_ext = spc.subtype.isin(EXT_SUB).values
print(f"    train={len(train)}, std_test={len(std)}, spc_test={len(spc)}")
print(f"    微博衍生子集={m_weibo.sum()}, 外部反讽子集={m_ext.sum()}")
print(f"    特殊子类: {spc.subtype.value_counts().to_dict()}")

# ================= 1. 构建特征管线 =================
print("[1] 特征提取 (TF-IDF + SVD + 免疫特征) ...")
FEATS, meta = build_pipeline(train, std, spc, mi_top_k=12)
print(f"    TF-IDF dim={meta['tfidf_dim']}, SVD dim={meta['svd_dim']}, explained={meta['svd_explained']:.3f}")
print(f"    极区: pos={meta['pos_pole_terms']}, neg={meta['neg_pole_terms']}")
print(f"    免疫特征: {meta['immune_dim']}维 -> MI筛选后 {meta['mi_selected_dim']}维")

F_tr, F_std, F_spc = meta["pipeline"]["F_tr"], meta["pipeline"]["F_std"], meta["pipeline"]["F_spc"]
V1_tr, V1_std, V1_spc = v1_features(F_tr), v1_features(F_std), v1_features(F_spc)
FEATS["B_v1"] = (V1_tr, V1_std, V1_spc)

# ================= 2. 实验矩阵 =================
print("[2] 训练与评估 ...")

def make_clf(name):
    if name == "LR":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    if name == "SVM":
        return LinearSVC(C=1.0, max_iter=5000, random_state=SEED, dual="auto")
    if name == "GBDT":
        return HistGradientBoostingClassifier(max_iter=300, max_depth=None,
                                              learning_rate=0.1, random_state=SEED)

def mets(y_true, y_pred):
    return (accuracy_score(y_true, y_pred) * 100,
            balanced_accuracy_score(y_true, y_pred) * 100,
            f1_score(y_true, y_pred, average="macro") * 100)

results, preds_cache = {}, {}
for fname, (a, b, c) in FEATS.items():
    for cname in ("LR", "SVM", "GBDT"):
        t0 = time.time()
        print(f"    ... training {fname}+{cname}", flush=True)
        clf = make_clf(cname); clf.fit(a, y_tr)
        p_std, p_spc = clf.predict(b), clf.predict(c)
        a_std, ba_std, f_std = mets(y_std, p_std)
        a_spc, ba_spc, f_spc = mets(y_spc, p_spc)
        a_w, ba_w, f_w = mets(y_spc[m_weibo], p_spc[m_weibo])
        a_e, ba_e, f_e = mets(y_spc[m_ext], p_spc[m_ext])
        key = f"{fname}+{cname}"
        results[key] = {
            "acc_std": round(a_std, 2), "bacc_std": round(ba_std, 2), "mf1_std": round(f_std, 2),
            "acc_spc": round(a_spc, 2), "bacc_spc": round(ba_spc, 2), "mf1_spc": round(f_spc, 2),
            "acc_weibo": round(a_w, 2), "bacc_weibo": round(ba_w, 2), "mf1_weibo": round(f_w, 2),
            "acc_ext": round(a_e, 2), "bacc_ext": round(ba_e, 2), "mf1_ext": round(f_e, 2),
            "gap": round(a_std - a_w, 2), "time": round(time.time() - t0, 1),
        }
        preds_cache[key] = p_spc
        print(f"    {key:16s} 标准={a_std:.2f} 微博子集={a_w:.2f} 外部={a_e:.2f} "
              f"| 全特殊={a_spc:.2f} bacc={ba_spc:.2f} mf1={f_spc:.2f} ({time.time()-t0:.1f}s)")

# ================= 3. 5折交叉验证 =================
print("[3] 5折交叉验证 (标准集泛化) ...")
cv_results = {}
for key in ["A_base+GBDT", "C_v2+GBDT", "B_v1+GBDT"]:
    fname, cname = key.split("+")
    X_tr = FEATS[fname][0]
    scores = cross_val_score(make_clf(cname), X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=-1)
    cv_results[key] = {"cv_mean": round(scores.mean() * 100, 2),
                       "cv_std": round(scores.std() * 100, 2),
                       "cv_scores": [round(s * 100, 2) for s in scores]}
    print(f"    {key:16s} CV={scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# ================= 4. 多随机种子稳健性 =================
print("[4] 多随机种子稳健性 (微博子集, A_base vs C_v2) ...")
robust = []
for seed in (7, 2024, 12345):
    pos = train[train.label == 1].sample(12000, random_state=seed)
    neg = train[train.label == 0].sample(12000, random_state=seed)
    tr_s = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)
    pipe = build_tfidf_svd(tr_s, std, spc)
    pol = learn_polarity_regions(pipe["Xtr_raw"], tr_s.label.values, pipe["vocab"])
    Mt = polarity_moments(pipe["Xtr_raw"], pol["pos_cols"], pol["neg_cols"])
    Ms = polarity_moments(pipe["Xstd_raw"], pol["pos_cols"], pol["neg_cols"])
    Mp = polarity_moments(pipe["Xspc_raw"], pol["pos_cols"], pol["neg_cols"])
    pr = build_memory(pipe["F_tr"], tr_s.label.values)
    s2m = np.median(Mt[:, 1] + Mt[:, 5] + 1e-12)
    sc = StandardScaler().fit(pipe["F_tr"])
    St, Ss, Sp = sc.transform(pipe["F_tr"]), sc.transform(pipe["F_std"]), sc.transform(pipe["F_spc"])
    It = build_v2_features(pipe["F_tr"], Mt, pr, s2m)
    isc = StandardScaler().fit(It)
    It, Is, Ip = isc.transform(It), isc.transform(build_v2_features(pipe["F_std"], Ms, pr, s2m)), \
                 isc.transform(build_v2_features(pipe["F_spc"], Mp, pr, s2m))
    mi = mutual_info_classif(It, tr_s.label.values, random_state=SEED)
    idx = np.argsort(mi)[::-1][:12]
    r = {}
    for nm, X in [("A", St), ("C", np.column_stack([St, It[:, idx]]))]:
        clf = make_clf("GBDT"); clf.fit(X, tr_s.label.values)
        r[nm] = accuracy_score(y_spc[m_weibo], clf.predict(Sp if nm == "A" else
                             np.column_stack([Sp, Ip[:, idx]]))[m_weibo]) * 100
    robust.append({"seed": seed, "A_base": round(r["A"], 2), "C_v2": round(r["C"], 2),
                   "delta": round(r["C"] - r["A"], 2)})
    print(f"    seed={seed:5d}: A_base={r['A']:.2f} C_v2={r['C']:.2f} Δ={r['C']-r['A']:+.2f}pp")

# ================= 5. 特殊舆情分子类评估 =================
print("[5] 特殊舆情分子类 ...")
subtypes = spc.subtype.unique().tolist()
sub_report = {}
for key in ["A_base+GBDT", "B_v1+GBDT", "C_v2+GBDT", "A_base+SVM", "C_v2+SVM"]:
    sub_report[key] = {s: round(accuracy_score(y_spc[(spc.subtype == s).values],
                                               preds_cache[key][(spc.subtype == s).values]) * 100, 2)
                       for s in subtypes}
print(pd.DataFrame(sub_report).T.to_string())

# ================= 6. McNemar 检验 =================
print("[6] 统计检验 (微博子集) ...")
def mcnemar(p1, p2, y):
    c1, c2 = (p1 == y), (p2 == y)
    n01, n10 = int(((~c1) & c2).sum()), int((c1 & (~c2)).sum())
    if n01 + n10 == 0:
        return 0.0, 1.0, n10, n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return chi2, 1 - sps.chi2.cdf(chi2, 1), n10, n01

mc = {}
for a, b in [("C_v2+GBDT", "A_base+GBDT"), ("C_v2+GBDT", "B_v1+GBDT"), ("C_v2+SVM", "A_base+SVM")]:
    chi2, p, n10, n01 = mcnemar(preds_cache[a][m_weibo], preds_cache[b][m_weibo], y_spc[m_weibo])
    mc[f"{a} vs {b}"] = {"chi2": round(chi2, 3), "p": round(p, 6),
                         f"{a}_独对": n10, f"{b}_独对": n01}
    print(f"    {a} vs {b}: chi2={chi2:.3f}, p={p:.2e}")

# ================= 7. 逃逸得分可解释性验证 =================
print("[7] 逃逸得分可解释性 (峰度版) ...")
M_spc, M_std = meta["M_spc"], meta["M_std"]
def esc_kurt(M):
    return np.minimum(M[:, 0], M[:, 4]) * (1 + np.abs(M[:, 3]) + np.abs(M[:, 7]))
esc_spc, esc_std = esc_kurt(M_spc), esc_kurt(M_std)
g_irony = spc.subtype.isin(["irony", "synthetic_irony", "sarcasm_neg"]).values & (y_spc == 0)
g_ordneg, g_ordpos = (y_std == 0), (y_std == 1)
t1, p1 = sps.ttest_ind(esc_spc[g_irony], esc_std[g_ordneg], equal_var=False)
pooled = np.sqrt((esc_spc[g_irony].var(ddof=1) + esc_std[g_ordneg].var(ddof=1)) / 2)
dcoh = (esc_spc[g_irony].mean() - esc_std[g_ordneg].mean()) / (pooled + 1e-12)
print(f"    反讽={esc_spc[g_irony].mean():.4f} 普通负面={esc_std[g_ordneg].mean():.4f} "
      f"普通正面={esc_std[g_ordpos].mean():.4f}")
print(f"    Welch t={t1:.2f}, p={p1:.2e}, Cohen's d={dcoh:.3f}")

# ================= 8. MI 特征重要度 =================
print("[8] MI 特征重要度 ...")
names = [r"pos $\mu$", r"pos $\sigma^2$", r"pos $\gamma_1$", r"pos $\gamma_2$",
         r"neg $\mu$", r"neg $\sigma^2$", r"neg $\gamma_1$", r"neg $\gamma_2$",
         "dominance", "conflict", "escape", "gate", "esc_tol",
         "aff_pos_best", "aff_pos_mean", "aff_neg_best", "aff_neg_mean"]
for i, idx in enumerate(meta["mi_top_indices"]):
    print(f"      {i+1:2d}. {names[idx]:16s} MI={meta['mi_scores'][i]:.4f}")

# ================= 9. 多数类基线 =================
maj = {}
for lab, m in [("weibo", m_weibo), ("ext", m_ext), ("all", np.ones(len(spc), bool))]:
    yy = y_spc[m]
    maj[lab] = {"acc": round((yy == 0).mean() * 100, 2),
                "bacc": 50.0,
                "mf1": round(f1_score(yy, np.zeros_like(yy), average="macro") * 100, 2)}
print(f"[9] 多数类基线: 微博={maj['weibo']}, 外部={maj['ext']}, 全特殊={maj['all']}")

# ================= 10. 保存 =================
out = {
    "dataset": {
        "train": len(train), "std_test": len(std), "special_test": len(spc),
        "weibo_subset": int(m_weibo.sum()), "external_subset": int(m_ext.sum()),
        "subtype_dist": spc.subtype.value_counts().to_dict(),
        "label_dist_spc": spc.label.value_counts().to_dict(),
        "svd_dim": meta["svd_dim"], "tfidf_dim": meta["tfidf_dim"],
        "pos_pole_terms": meta["pos_pole_terms"], "neg_pole_terms": meta["neg_pole_terms"],
        "immune_dim": meta["immune_dim"], "mi_selected_dim": meta["mi_selected_dim"],
    },
    "main_results": results,
    "cross_validation": cv_results,
    "robustness_seeds": robust,
    "subtype_acc": sub_report,
    "mcnemar_weibo": mc,
    "escape_validation": {
        "esc_irony_mean": round(float(esc_spc[g_irony].mean()), 5),
        "esc_ordneg_mean": round(float(esc_std[g_ordneg].mean()), 5),
        "esc_ordpos_mean": round(float(esc_std[g_ordpos].mean()), 5),
        "welch_t": round(float(t1), 2), "p_value": float(p1), "cohens_d": round(float(dcoh), 3),
        "irony_n": int(g_irony.sum()), "ordneg_n": int(g_ordneg.sum()),
    },
    "majority_baseline": maj,
    "mi_feature_importance": [{"feature": names[i], "mi": round(m, 4)}
                              for i, m in zip(meta["mi_top_indices"], meta["mi_scores"])],
    "runtime_s": round(time.time() - T0, 1),
}
with open(os.path.join(BASE, "experiment_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n[DONE] 结果已保存: experiment_results.json")
print(f"总耗时 {time.time()-T0:.0f}s")
