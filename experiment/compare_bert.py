# -*- coding: utf-8 -*-
"""C_v2+GBDT vs BERT 的 McNemar 检验（微博子集 / 全特殊集 / 标准集）"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from immune_pipeline import SEED, load_data, build_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
train, std, spc = load_data(BASE)
y_std, y_spc = std.label.values, spc.label.values
WEIBO = ["irony", "pos_surface", "irony_pos", "synthetic_irony", "netslang", "netslang_pos"]
m_w = spc.subtype.isin(WEIBO).values

FEATS, meta = build_pipeline(train, std, spc, mi_top_k=12)
A, B, C = FEATS["C_v2"]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
clf.fit(A, train.label.values)
p_std, p_spc = clf.predict(B), clf.predict(C)
np.save(os.path.join(BASE, "immune_v3_pred_std.npy"), p_std)
np.save(os.path.join(BASE, "immune_v3_pred_spc.npy"), p_spc)

bp_std = np.load(os.path.join(BASE, "bert_v3_pred_std.npy"))
bp_spc = np.load(os.path.join(BASE, "bert_v3_pred_spc.npy"))

def mcnemar(p1, p2, y):
    c1, c2 = (p1 == y), (p2 == y)
    n01, n10 = int(((~c1) & c2).sum()), int((c1 & (~c2)).sum())
    if n01 + n10 == 0:
        return 0.0, 1.0, n10, n01
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return chi2, 1 - sps.chi2.cdf(chi2, 1), n10, n01

def mt(y, p):
    return (round(accuracy_score(y, p) * 100, 2),
            round(balanced_accuracy_score(y, p) * 100, 2),
            round(f1_score(y, p, average="macro") * 100, 2))

out = {}
print(f"{'set':16s} {'C_v2 acc':>9s} {'BERT acc':>9s} | {'chi2':>8s} {'p':>10s} {'C独对':>6s} {'B独对':>6s}")
for name, yy, pc, pb in [("weibo(4453)", y_spc[m_w], p_spc[m_w], bp_spc[m_w]),
                         ("all(5774)", y_spc, p_spc, bp_spc),
                         ("std(2000)", y_std, p_std, bp_std)]:
    chi2, p, n10, n01 = mcnemar(pc, pb, yy)
    a_c, b_c, f_c = mt(yy, pc)
    a_b, b_b, f_b = mt(yy, pb)
    out[name] = {"C_v2": {"acc": a_c, "bacc": b_c, "mf1": f_c},
                 "BERT": {"acc": a_b, "bacc": b_b, "mf1": f_b},
                 "mcnemar": {"chi2": round(chi2, 3), "p": p, "C_v2_only": n10, "BERT_only": n01}}
    print(f"{name:16s} {a_c:9.2f} {a_b:9.2f} | {chi2:8.3f} {p:10.2e} {n10:6d} {n01:6d}")
    print(f"{'':16s} bacc {b_c:6.2f} vs {b_b:6.2f} | mf1 {f_c:6.2f} vs {f_b:6.2f}")

json.dump(out, open(os.path.join(BASE, "bert_compare_v3.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nsaved bert_compare_v3.json")
