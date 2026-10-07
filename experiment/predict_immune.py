# -*- coding: utf-8 -*-
"""
导出免疫方法（C_v2，GBDT/SVM）在标准/特殊测试集的预测，供与 BERT 的 McNemar 对比。
复用 run_experiment.py 的特征管线（v2 极区统计矩 + 免疫特征 + 互信息筛选）。
"""
import os, json, warnings, numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")
SEED = 42; rng = np.random.RandomState(SEED)
BASE = os.path.dirname(os.path.abspath(__file__)); MYDS = os.path.join(BASE, "my_dataset")
train = pd.read_csv(os.path.join(MYDS, "train_standard.csv"))
std = pd.read_csv(os.path.join(MYDS, "test_standard.csv"))
spc = pd.read_csv(os.path.join(MYDS, "test_special.csv"))
for d in (train, std, spc): d["text"] = d["text"].astype(str)

tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), max_features=20000, min_df=3, sublinear_tf=True)
Xtr = tfidf.fit_transform(train.text); Xstd = tfidf.transform(std.text); Xspc = tfidf.transform(spc.text)
svd = TruncatedSVD(n_components=300, random_state=SEED)
F_tr, F_std, F_spc = svd.fit_transform(Xtr), svd.transform(Xstd), svd.transform(Xspc)
y_tr, y_std, y_spc = train.label.values, std.label.values, spc.label.values

pos_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(Xtr, y_tr)
w = pos_lr.coef_[0]; z = (w - w.mean()) / (w.std() + 1e-9)
pos_mask, neg_mask = z > 1.5, z < -1.5
pos_cols, neg_cols = np.where(pos_mask)[0], np.where(neg_mask)[0]

def polarity_moments(X_raw):
    Xp, Xn = X_raw[:, pos_cols], X_raw[:, neg_cols]
    out = []
    for M in (Xp, Xn):
        Ml = M.tolil(); mus, sig2s, gams, kaps = [], [], [], []
        for i in range(M.shape[0]):
            v = np.array(Ml.data[i]) if len(Ml.data[i]) else np.array([0.0])
            mu = v.mean(); s2 = v.var()
            if s2 > 1e-12:
                st = (v - mu) / np.sqrt(s2); gam = (st ** 3).mean(); kap = (st ** 4).mean() - 3.0
            else: gam, kap = 0.0, 0.0
            mus.append(mu); sig2s.append(s2); gams.append(gam); kaps.append(kap)
        out += [mus, sig2s, gams, kaps]
    return np.column_stack([np.array(c) for c in out])

def build_memory(F, y, n_per_class=20, beta=0.3):
    protos = {}
    for c in (0, 1):
        Fc = F[y == c]; mu_c = Fc.mean(axis=0)
        idx = rng.choice(len(Fc), size=min(n_per_class, len(Fc)), replace=False)
        P = (1 - beta) * Fc[idx] + beta * mu_c
        protos[c] = P
    return protos

def affinity_feats(F, protos):
    feats = np.zeros((F.shape[0], 4))
    for c in (0, 1):
        P = protos[c]
        d = np.linalg.norm(F[:, None, :] - P[None, :, :], axis=2)
        feats[:, 2*c] = 1.0 / (1.0 + d.min(axis=1)); feats[:, 2*c+1] = 1.0 / (1.0 + d.mean(axis=1))
    return feats

M_tr, M_std, M_spc = polarity_moments(Xtr), polarity_moments(Xstd), polarity_moments(Xspc)
protos = build_memory(F_tr, y_tr)
s2_med = np.median(M_tr[:, 1] + M_tr[:, 5] + 1e-12)
def tolerance_gate(M):
    return 1.0 / (1.0 + np.exp(-((M[:, 1] + M[:, 5] + 1e-12) / s2_med - 1.0)))

def build_v2(F, M_raw, protos):
    p_mu_, n_mu_ = M_raw[:, 0], M_raw[:, 4]; p_g_, n_g_ = M_raw[:, 2], M_raw[:, 6]
    dominance_ = p_mu_ - n_mu_; conflict_ = np.minimum(p_mu_, n_mu_)
    esc_ = conflict_ * (1 + np.abs(p_g_) + np.abs(n_g_))
    esc_tol_ = esc_ * tolerance_gate(M_raw)
    return np.column_stack([M_raw, dominance_, conflict_, esc_, tolerance_gate(M_raw), esc_tol_, affinity_feats(F, protos)])

I_tr, I_std, I_spc = build_v2(F_tr, M_tr, protos), build_v2(F_std, M_std, protos), build_v2(F_spc, M_spc, protos)
iscaler = StandardScaler().fit(I_tr)
I_tr, I_std, I_spc = iscaler.transform(I_tr), iscaler.transform(I_std), iscaler.transform(I_spc)
mi = mutual_info_classif(I_tr, y_tr, random_state=SEED); top = np.argsort(mi)[::-1][:20]
I_tr, I_std, I_spc = I_tr[:, top], I_std[:, top], I_spc[:, top]
S_tr, S_std, S_spc = StandardScaler().fit(F_tr).transform(F_tr), StandardScaler().fit(F_tr).transform(F_std), StandardScaler().fit(F_tr).transform(F_spc)
V2_tr, V2_std, V2_spc = np.column_stack([S_tr, I_tr]), np.column_stack([S_std, I_std]), np.column_stack([S_spc, I_spc])

out = {}
for cname in ("GBDT", "SVM"):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED) if cname == "GBDT" else SVC(kernel="rbf", C=2.0, random_state=SEED)
    clf.fit(V2_tr, y_tr)
    p_std, p_spc = clf.predict(V2_std), clf.predict(V2_spc)
    np.save(os.path.join(BASE, f"immune_{cname}_pred_std.npy"), p_std)
    np.save(os.path.join(BASE, f"immune_{cname}_pred_spc.npy"), p_spc)
    out[cname] = {"acc_std": round((p_std == y_std).mean()*100, 2), "acc_spc": round((p_spc == y_spc).mean()*100, 2)}
    print(cname, out[cname], flush=True)
np.save(os.path.join(BASE, "y_std.npy"), y_std); np.save(os.path.join(BASE, "y_spc.npy"), y_spc)
json.dump(out, open(os.path.join(BASE, "immune_pred_summary.json"), "w"), ensure_ascii=False, indent=2)
print("saved", flush=True)