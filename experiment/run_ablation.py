# -*- coding: utf-8 -*-
"""消融实验（统一管线，24K 纯微博训练，微博衍生特殊子集评估）
与主模型共用 immune_pipeline.build_pipeline，消除原实现"消融与主模型管线不一致"的问题。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from immune_pipeline import (SEED, load_data, build_tfidf_svd, learn_polarity_regions,
                             polarity_moments, build_memory, build_v2_features)

BASE = os.path.dirname(os.path.abspath(__file__))
train, std, spc = load_data(BASE)
y_tr, y_std, y_spc = train.label.values, std.label.values, spc.label.values
WEIBO = ["irony", "pos_surface", "irony_pos", "synthetic_irony", "netslang", "netslang_pos"]
m_w = spc.subtype.isin(WEIBO).values

pipe = build_tfidf_svd(train, std, spc)
pol = learn_polarity_regions(pipe["Xtr_raw"], y_tr, pipe["vocab"])
Mt = polarity_moments(pipe["Xtr_raw"], pol["pos_cols"], pol["neg_cols"])
Ms = polarity_moments(pipe["Xstd_raw"], pol["pos_cols"], pol["neg_cols"])
Mp = polarity_moments(pipe["Xspc_raw"], pol["pos_cols"], pol["neg_cols"])
protos = build_memory(pipe["F_tr"], y_tr)
s2m = np.median(Mt[:, 1] + Mt[:, 5] + 1e-12)
sc = StandardScaler().fit(pipe["F_tr"])
St, Ss, Sp = sc.transform(pipe["F_tr"]), sc.transform(pipe["F_std"]), sc.transform(pipe["F_spc"])
It = build_v2_features(pipe["F_tr"], Mt, protos, s2m)
isc = StandardScaler().fit(It)
It = isc.transform(It)
Is = isc.transform(build_v2_features(pipe["F_std"], Ms, protos, s2m))
Ip = isc.transform(build_v2_features(pipe["F_spc"], Mp, protos, s2m))

MI_TOP_K = 12   # 与 immune_pipeline.build_pipeline 的默认 mi_top_k 一致；
                # 消融必须走同一协议，否则"全特征"行(=17维)与主结果(=MI top-12)
                # 不可比——这正是 83.83/89.26/76.85 与 83.85/89.08/76.83 差 0.18 的原因。
print(f"免疫特征 {It.shape[1]} 维；每个配置内部按 MI 选 top-{MI_TOP_K}\n")

DROPS = {
    "Full immunity (all 17 feats)": set(),
    "w/o polarity-region moments": set(range(0, 8)),
    "w/o escape channel": {10, 12},
    "w/o tolerance gate": {11},
    "w/o memory affinity": {13, 14, 15, 16},
    "w/o dominance & conflict": {8, 9},
}

def ev(y, p):
    return (round(accuracy_score(y, p) * 100, 2),
            round(balanced_accuracy_score(y, p) * 100, 2),
            round(f1_score(y, p, average="macro") * 100, 2))

rows = {}
print(f"{'configuration':38s} {'dim':>4s} | {'weibo acc':>9s} {'bacc':>7s} {'mf1':>7s} {'std acc':>8s}")
print("-" * 82)
for name, drop in DROPS.items():
    keep = [i for i in range(It.shape[1]) if i not in drop]
    # 与主实验同一协议：在候选通道上重算 MI，保留 top-min(K, n)。
    # "全特征"这一行因此与 build_pipeline 选出的通道完全一致，数值应逐位相等。
    mi = mutual_info_classif(It[:, keep], y_tr, random_state=SEED)
    k = min(MI_TOP_K, len(keep))
    sel = [keep[i] for i in np.argsort(mi)[::-1][:k]]
    A = np.column_stack([St, It[:, sel]])
    B = np.column_stack([Ss, Is[:, sel]])
    C = np.column_stack([Sp, Ip[:, sel]])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
    clf.fit(A, y_tr)
    a_w, b_w, f_w = ev(y_spc[m_w], clf.predict(C)[m_w])
    a_s, _, _ = ev(y_std, clf.predict(B))
    rows[name] = {"dim": len(keep), "mi_dim": len(sel), "weibo_acc": a_w,
                  "weibo_bacc": b_w, "weibo_mf1": f_w, "std_acc": a_s}
    print(f"{name:38s} {len(keep):3d}->{len(sel):<2d} | {a_w:9.2f} {b_w:7.2f} {f_w:7.2f} {a_s:8.2f}")

# 基线：纯基础特征
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
clf.fit(St, y_tr)
a_w, b_w, f_w = ev(y_spc[m_w], clf.predict(Sp)[m_w])
a_s, _, _ = ev(y_std, clf.predict(Ss))
rows["w/o immunity (base only)"] = {"dim": 0, "weibo_acc": a_w, "weibo_bacc": b_w,
                                    "weibo_mf1": f_w, "std_acc": a_s}
print(f"{'w/o immunity (base only)':38s} {0:4d} | {a_w:9.2f} {b_w:7.2f} {f_w:7.2f} {a_s:8.2f}")

full = rows["Full immunity (all 17 feats)"]["weibo_acc"]
print(f"\n相对 full 的下降 (weibo acc):")
for k, v in rows.items():
    if k != "Full immunity (all 17 feats)":
        print(f"  {k:38s} {v['weibo_acc']-full:+.2f} pp")

json.dump(rows, open(os.path.join(BASE, "ablation_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nsaved ablation_results.json")
