# -*- coding: utf-8 -*-
"""补充验证: 逃逸得分可解释性 + 耐受门有效性 + 消融实验 (复用主实验流程)"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.RandomState(SEED)
BASE = os.path.dirname(os.path.abspath(__file__))
MYDS = os.path.join(BASE, "my_dataset")
T0 = time.time()

train = pd.read_csv(os.path.join(MYDS, "train_standard.csv"))
std = pd.read_csv(os.path.join(MYDS, "test_standard.csv"))
spc = pd.read_csv(os.path.join(MYDS, "test_special.csv"))
for d in (train, std, spc):
    d["text"] = d["text"].astype(str)

print("[1] 特征提取(与主实验一致) ...")
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=20000, min_df=3, sublinear_tf=True)
Xtr_raw = tfidf.fit_transform(train.text)
Xstd_raw = tfidf.transform(std.text)
Xspc_raw = tfidf.transform(spc.text)
svd = TruncatedSVD(n_components=300, random_state=SEED)
F_tr, F_std, F_spc = svd.fit_transform(Xtr_raw), svd.transform(Xstd_raw), svd.transform(Xspc_raw)
y_tr, y_std, y_spc = train.label.values, std.label.values, spc.label.values
vocab = np.array(tfidf.get_feature_names_out())

print("[2] 极区学习 ...")
pos_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(Xtr_raw, y_tr)
w = pos_lr.coef_[0]
z = (w - w.mean()) / (w.std() + 1e-9)
pos_mask, neg_mask = z > 1.5, z < -1.5
pos_cols, neg_cols = np.where(pos_mask)[0], np.where(neg_mask)[0]

def polarity_moments(X_raw):
    out = []
    for M in (X_raw[:, pos_cols], X_raw[:, neg_cols]):
        Ml = M.tolil()
        rows, data = Ml.rows, Ml.data
        mus, sig2s, gams, kaps = [], [], [], []
        for i in range(M.shape[0]):
            v = np.array(data[i]) if len(data[i]) else np.array([0.0])
            mu, s2 = v.mean(), v.var()
            if s2 > 1e-12:
                st = (v - mu) / np.sqrt(s2)
                gam, kap = (st ** 3).mean(), (st ** 4).mean() - 3.0
            else:
                gam, kap = 0.0, 0.0
            mus.append(mu); sig2s.append(s2); gams.append(gam); kaps.append(kap)
        out += [mus, sig2s, gams, kaps]
    return np.column_stack([np.array(c) for c in out])  # 8列

print("[3] 极区统计矩 ...")
M_tr, M_std, M_spc = polarity_moments(Xtr_raw), polarity_moments(Xstd_raw), polarity_moments(Xspc_raw)

s2_med = np.median(M_tr[:, 1] + M_tr[:, 5] + 1e-12)
def core_feats(M, use_gate=True, use_esc=True):
    p_mu_, n_mu_ = M[:, 0], M[:, 4]
    p_g_, n_g_ = M[:, 2], M[:, 6]
    dominance_ = p_mu_ - n_mu_
    conflict_ = np.minimum(p_mu_, n_mu_)
    esc_ = conflict_ * (1 + np.abs(p_g_) + np.abs(n_g_))
    tot_s2 = M[:, 1] + M[:, 5] + 1e-12
    gate_ = 1.0 / (1.0 + np.exp(-((tot_s2 / s2_med) - 1.0)))
    esc_tol_ = esc_ * gate_ if use_gate else esc_
    if use_esc:
        return np.column_stack([M, dominance_, conflict_, gate_, esc_tol_])
    return np.column_stack([M, dominance_, conflict_, gate_])

def build_memory(F, y, n_per_class=20, beta=0.3):
    protos = {}
    for c in (0, 1):
        Fc = F[y == c]
        mu_c = Fc.mean(axis=0)
        idx = rng.choice(len(Fc), size=min(n_per_class, len(Fc)), replace=False)
        protos[c] = (1 - beta) * Fc[idx] + beta * mu_c
    return protos

def affinity_feats(F, protos):
    feats = np.zeros((F.shape[0], 4))
    for c in (0, 1):
        d = np.linalg.norm(F[:, None, :] - protos[c][None, :, :], axis=2)
        feats[:, 2 * c] = 1.0 / (1.0 + d.min(axis=1))
        feats[:, 2 * c + 1] = 1.0 / (1.0 + d.mean(axis=1))
    return feats

scaler = StandardScaler().fit(F_tr)
S_tr, S_std, S_spc = scaler.transform(F_tr), scaler.transform(F_std), scaler.transform(F_spc)
protos = build_memory(F_tr, y_tr)

def make_immune(M_tr_, M_eval_, with_aff, use_gate=True, use_esc=True, select=True):
    I_tr_ = core_feats(M_tr_, use_gate, use_esc)
    I_ev_ = core_feats(M_eval_, use_gate, use_esc)
    if with_aff:
        I_tr_ = np.column_stack([I_tr_, affinity_feats(F_tr, protos)])
        I_ev_ = np.column_stack([I_ev_, affinity_feats(F_eval_global, protos)])
    sc = StandardScaler().fit(I_tr_)
    I_tr_, I_ev_ = sc.transform(I_tr_), sc.transform(I_ev_)
    if select:
        mi = mutual_info_classif(I_tr_, y_tr, random_state=SEED)
        idx = np.argsort(mi)[::-1][:min(20, I_tr_.shape[1])]
        I_tr_, I_ev_ = I_tr_[:, idx], I_ev_[:, idx]
    return I_tr_, I_ev_

F_eval_global = F_spc  # affinity用于评估时
out = {}

# ============ 验证1: 逃逸得分可解释性 (P3理论验证) ============
print("[4] 逃逸得分可解释性验证 ...")
esc_spc = core_feats(M_spc, use_gate=False)[:, 8 + 0]  # 无gate时esc_在列8
# 重算: core_feats(use_esc=True)列序: M(8) + dom(8) + conflict(9) + gate(10) + esc_tol(11) => esc_tol=11, 但use_gate=False时esc_tol=esc_
esc_spc = core_feats(M_spc, use_gate=False)[:, 11]
esc_tr = core_feats(M_tr, use_gate=False)[:, 11]
irony_mask = (spc.subtype.isin(["irony", "synthetic_irony"])).values & (y_spc == 0)
plain_neg_mask = ((~spc.subtype.isin(["irony", "synthetic_irony"]))).values & (y_spc == 0)
t1, p1 = sps.ttest_ind(esc_spc[irony_mask], esc_spc[plain_neg_mask], equal_var=False)
out["escape_validation"] = {
    "irony_mean": round(float(esc_spc[irony_mask].mean()), 5),
    "plain_neg_mean": round(float(esc_spc[plain_neg_mask].mean()), 5),
    "ratio": round(float(esc_spc[irony_mask].mean() / (esc_spc[plain_neg_mask].mean() + 1e-12)), 2),
    "welch_t": round(float(t1), 2),
    "p_value": float(p1),
    "n_irony": int(irony_mask.sum()),
    "n_plain_neg": int(plain_neg_mask.sum()),
}
print(f"    反讽组esc均值={esc_spc[irony_mask].mean():.4f} 普通负面={esc_spc[plain_neg_mask].mean():.4f} | t={t1:.2f} p={p1:.2e}")

# ============ 验证2: 耐受门有效性 (P4验证: irony_pos误判率) ============
print("[5] 耐受门有效性验证 ...")
def run_gbdt(I_tr_, I_spc_):
    X = np.column_stack([S_tr, I_tr_]); Xe = np.column_stack([S_spc, I_spc_])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED).fit(X, y_tr)
    return clf.predict(Xe)

Itr_g, Isp_g = make_immune(M_tr, M_spc, with_aff=True, use_gate=True)
Itr_ng, Isp_ng = make_immune(M_tr, M_spc, with_aff=True, use_gate=False)
pred_gate, pred_nogate = run_gbdt(Itr_g, Isp_g), run_gbdt(Itr_ng, Isp_ng)
irony_pos_m = (spc.subtype == "irony_pos").values
netslang_pos_m = (spc.subtype == "netslang_pos").values
out["tolerance_gate"] = {
    "irony_pos_acc_with_gate": round(accuracy_score(y_spc[irony_pos_m], pred_gate[irony_pos_m]) * 100, 2),
    "irony_pos_acc_no_gate": round(accuracy_score(y_spc[irony_pos_m], pred_nogate[irony_pos_m]) * 100, 2),
    "netslang_pos_acc_with_gate": round(accuracy_score(y_spc[netslang_pos_m], pred_gate[netslang_pos_m]) * 100, 2),
    "netslang_pos_acc_no_gate": round(accuracy_score(y_spc[netslang_pos_m], pred_nogate[netslang_pos_m]) * 100, 2),
    "overall_spc_with_gate": round(accuracy_score(y_spc, pred_gate) * 100, 2),
    "overall_spc_no_gate": round(accuracy_score(y_spc, pred_nogate) * 100, 2),
}
print("   ", json.dumps(out["tolerance_gate"], ensure_ascii=False))

# ============ 验证3: 消融实验 ============
print("[6] 消融实验 ...")
abl = {}
# Full: 极区矩+极性/冲突/逃逸+耐受门+记忆亲和
Itr_full, Isp_full = make_immune(M_tr, M_spc, with_aff=True)
p_full = run_gbdt(Itr_full, Isp_full)
abl["Full_v2"] = round(accuracy_score(y_spc, p_full) * 100, 2)
# -极区矩 (只留衍生特征+亲和)
Itr_nm = np.column_stack([core_feats(M_tr)[:, 8:], affinity_feats(F_tr, protos)])
Isp_nm = np.column_stack([core_feats(M_spc)[:, 8:], affinity_feats(F_spc, protos)])
abl["no_moments"] = round(accuracy_score(y_spc, run_gbdt(Itr_nm, Isp_nm)) * 100, 2)
# -逃逸通道 (只极区矩+dom/conflict+gate+亲和)
Itr_ne, Isp_ne = make_immune(M_tr, M_spc, with_aff=True, use_esc=False)
abl["no_escape"] = round(accuracy_score(y_spc, run_gbdt(Itr_ne, Isp_ne)) * 100, 2)
# -记忆亲和
Itr_na, Isp_na = make_immune(M_tr, M_spc, with_aff=False)
abl["no_memory"] = round(accuracy_score(y_spc, run_gbdt(Itr_na, Isp_na)) * 100, 2)
# -全部免疫(=基线)
abl["no_immune(=base)"] = round(accuracy_score(y_spc, run_gbdt(np.zeros((len(y_tr), 1)), np.zeros((len(y_spc), 1)))) * 100, 2)
out["ablation"] = abl
print("   ", json.dumps(abl, ensure_ascii=False))

# ============ 验证4: LR系数解读 (免疫特征贡献排序) ============
print("[7] 免疫特征重要性(LR系数+互信息) ...")
Itr_i, Isp_i = make_immune(M_tr, M_spc, with_aff=True)
mi = mutual_info_classif(Itr_i, y_tr, random_state=SEED)
names_base = ["pos_mu", "pos_s2", "pos_gamma", "pos_kappa", "neg_mu", "neg_s2", "neg_gamma", "neg_kappa",
              "dominance", "conflict", "gate", "esc_tol", "aff0_best", "aff0_mean", "aff1_best", "aff1_mean"]
sel = np.argsort(mi)[::-1][:min(20, Itr_i.shape[1])]
imp = [{"feature": names_base[j] if j < len(names_base) else f"dim{j}", "mi": round(float(mi[j]), 4)} for j in sel[:10]]
out["feature_importance_top10"] = imp
for it in imp:
    print(f"    {it['feature']:12s} MI={it['mi']}")

out["runtime_s"] = round(time.time() - T0, 1)
with open(os.path.join(BASE, "verify_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n[DONE] 保存: verify_results.json | 耗时", out["runtime_s"], "s")
