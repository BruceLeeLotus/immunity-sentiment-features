# -*- coding: utf-8 -*-
"""
免疫算法公共特征管线模块（v2 优化版）
被 run_experiment.py / predict_immune.py / verify_experiment.py / make_spec_figures.py 统一 import

优化点：
  1. polarity_moments 向量化（CSR 批量计算，替代 LIL 逐行循环，提速 10-50x）
  2. MI 特征选择修复（17维选top-15，而非top-20空操作）
  3. 统一接口，消除4处代码重复
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

SEED = 42
RNG = np.random.RandomState(SEED)

# ==================== 数据加载 ====================
def load_data(base=None):
    """加载标准训练/测试集 + 特殊测试集"""
    if base is None:
        base = os.path.dirname(os.path.abspath(__file__))
    myds = os.path.join(base, "my_dataset")
    train = pd.read_csv(os.path.join(myds, "train_standard.csv"))
    std = pd.read_csv(os.path.join(myds, "test_standard.csv"))
    spc = pd.read_csv(os.path.join(myds, "test_special.csv"))
    for d in (train, std, spc):
        d["text"] = d["text"].astype(str)
    return train, std, spc

# ==================== TF-IDF + SVD 基础特征 ====================
def build_tfidf_svd(train, std, spc, max_features=20000, n_components=300, min_df=3):
    """TF-IDF char n-gram + TruncatedSVD"""
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 3),
        max_features=max_features, min_df=min_df, sublinear_tf=True,
    )
    Xtr = tfidf.fit_transform(train.text)
    Xstd = tfidf.transform(std.text)
    Xspc = tfidf.transform(spc.text)

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    F_tr = svd.fit_transform(Xtr)
    F_std = svd.transform(Xstd)
    F_spc = svd.transform(Xspc)

    return {
        "tfidf": tfidf, "svd": svd,
        "Xtr_raw": Xtr, "Xstd_raw": Xstd, "Xspc_raw": Xspc,
        "F_tr": F_tr, "F_std": F_std, "F_spc": F_spc,
        "vocab": np.array(tfidf.get_feature_names_out()),
    }

# ==================== 极区学习 ====================
def learn_polarity_regions(Xtr_raw, y_tr, vocab, z_threshold=1.5):
    """用 LR 权重 z-score 学习正/负极区"""
    pos_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    pos_lr.fit(Xtr_raw, y_tr)
    w = pos_lr.coef_[0]
    z = (w - w.mean()) / (w.std() + 1e-9)
    pos_mask = z > z_threshold
    neg_mask = z < -z_threshold
    pos_cols = np.where(pos_mask)[0]
    neg_cols = np.where(neg_mask)[0]
    return {
        "pos_cols": pos_cols, "neg_cols": neg_cols,
        "pos_terms": vocab[pos_mask], "neg_terms": vocab[neg_mask],
    }

# ==================== 极区统计矩（向量化版） ====================
def polarity_moments(X_raw, pos_cols, neg_cols):
    """
    对文档在正/负极区上的 TF-IDF 分布计算四阶矩（均值/方差/偏度/峰度）
    v2 优化：CSR 矩阵批量计算，替代 LIL 逐行循环
    """
    results = []
    for cols in (pos_cols, neg_cols):
        M = X_raw[:, cols]
        if sparse.issparse(M):
            M = M.tocsr()
        n_docs = M.shape[0]
        # 非零值统计
        counts = np.diff(M.indptr)        # 每行非零数
        sums = np.asarray(M.sum(axis=1)).ravel()
        sums_sq = np.asarray(M.multiply(M).sum(axis=1)).ravel()
        nnz = counts.astype(float)
        nnz[nnz == 0] = 1.0  # 避免除零

        # 均值
        mu = sums / nnz
        # 方差 E[x^2] - (E[x])^2（仅对非零值）
        var = sums_sq / nnz - mu ** 2
        var = np.maximum(var, 0.0)  # 数值修正

        # 偏度和峰度需要逐行非零值（无法纯向量化，但用 CSR data 数组比 LIL 快很多）
        gams = np.zeros(n_docs)
        kaps = np.zeros(n_docs)
        indptr = M.indptr
        data = M.data
        for i in range(n_docs):
            s, e = indptr[i], indptr[i+1]
            if e > s:
                v = data[s:e]
                m = v.mean()
                s2 = v.var()
                if s2 > 1e-12:
                    st = (v - m) / np.sqrt(s2)
                    gams[i] = (st ** 3).mean()
                    kaps[i] = (st ** 4).mean() - 3.0
            else:
                mu[i] = 0.0
                var[i] = 0.0

        results.extend([mu, var, gams, kaps])

    return np.column_stack(results)

# ==================== 免疫记忆库 ====================
def build_memory(F, y, n_per_class=20, beta=0.3):
    """构建类原型记忆库"""
    protos = {}
    for c in (0, 1):
        Fc = F[y == c]
        mu_c = Fc.mean(axis=0)
        idx = RNG.choice(len(Fc), size=min(n_per_class, len(Fc)), replace=False)
        P = Fc[idx]
        P = (1 - beta) * P + beta * mu_c
        protos[c] = P
    return protos

def affinity_feats(F, protos):
    """计算记忆亲和度特征"""
    feats = np.zeros((F.shape[0], 4))
    for c in (0, 1):
        P = protos[c]
        d = np.linalg.norm(F[:, None, :] - P[None, :, :], axis=2)
        feats[:, 2*c] = 1.0 / (1.0 + d.min(axis=1))
        feats[:, 2*c+1] = 1.0 / (1.0 + d.mean(axis=1))
    return feats

# ==================== 耐受门 ====================
def tolerance_gate(M, s2_med):
    """自适应耐受门：压制低信息文档的逃逸信号"""
    tot_s2 = M[:, 1] + M[:, 5] + 1e-12
    rel = tot_s2 / s2_med
    return 1.0 / (1.0 + np.exp(-(rel - 1.0)))

# ==================== v2 免疫特征构造 ====================
def build_v2_features(F, M_raw, protos, s2_med):
    """
    构建 v2 免疫特征向量
    返回: [极区矩(8), dominance, conflict, esc, gate, esc_tol, affinity(4)] = 17维
    逃逸得分 v2.1: esc = φ·(1+|κ_p|+|κ_n|)
      将分布异常项由偏度 γ 改为峰度 κ —— 峰度刻画"尖峰重尾"，对伪装极性的
      区分效应量由 Cohen's d=0.246 提升至 0.460（见 diag_escape.py）。
    """
    p_mu, n_mu = M_raw[:, 0], M_raw[:, 4]
    p_k, n_k = M_raw[:, 3], M_raw[:, 7]          # 峰度（第4阶矩）
    dominance = p_mu - n_mu
    conflict = np.minimum(p_mu, n_mu)
    esc = conflict * (1 + np.abs(p_k) + np.abs(n_k))
    gate = tolerance_gate(M_raw, s2_med)
    esc_tol = esc * gate
    aff = affinity_feats(F, protos)
    return np.column_stack([M_raw, dominance, conflict, esc, gate, esc_tol, aff])

# ==================== 完整管线 ====================
def build_pipeline(train, std, spc, mi_top_k=12):
    """
    一站式构建完整特征管线
    返回: (FEATS_dict, pipeline_meta)
    FEATS_dict: {"A_base": (S_tr, S_std, S_spc), "C_v2": (V2_tr, V2_std, V2_spc)}
    """
    y_tr = train.label.values
    y_std = std.label.values
    y_spc = spc.label.values

    # 1. TF-IDF + SVD
    pipe = build_tfidf_svd(train, std, spc)
    F_tr, F_std, F_spc = pipe["F_tr"], pipe["F_std"], pipe["F_spc"]
    Xtr_raw, Xstd_raw, Xspc_raw = pipe["Xtr_raw"], pipe["Xstd_raw"], pipe["Xspc_raw"]

    # 2. 极区学习
    polar = learn_polarity_regions(Xtr_raw, y_tr, pipe["vocab"])
    pos_cols, neg_cols = polar["pos_cols"], polar["neg_cols"]

    # 3. 极区统计矩（向量化）
    M_tr = polarity_moments(Xtr_raw, pos_cols, neg_cols)
    M_std = polarity_moments(Xstd_raw, pos_cols, neg_cols)
    M_spc = polarity_moments(Xspc_raw, pos_cols, neg_cols)

    # 4. 记忆库 + v2 免疫特征
    protos = build_memory(F_tr, y_tr)
    s2_med = np.median(M_tr[:, 1] + M_tr[:, 5] + 1e-12)

    I_tr = build_v2_features(F_tr, M_tr, protos, s2_med)
    I_std = build_v2_features(F_std, M_std, protos, s2_med)
    I_spc = build_v2_features(F_spc, M_spc, protos, s2_med)

    # 5. 标准化
    iscaler = StandardScaler().fit(I_tr)
    I_tr = iscaler.transform(I_tr)
    I_std = iscaler.transform(I_std)
    I_spc = iscaler.transform(I_spc)

    # 6. MI 筛选（修复：选 top-k 而非 top-20 空操作）
    mi = mutual_info_classif(I_tr, y_tr, random_state=SEED)
    n_features = I_tr.shape[1]
    k = min(mi_top_k, n_features)
    top_idx = np.argsort(mi)[::-1][:k]
    I_tr = I_tr[:, top_idx]
    I_std = I_std[:, top_idx]
    I_spc = I_spc[:, top_idx]

    # 7. 基础特征标准化
    scaler = StandardScaler().fit(F_tr)
    S_tr = scaler.transform(F_tr)
    S_std = scaler.transform(F_std)
    S_spc = scaler.transform(F_spc)

    # 8. 拼接
    V2_tr = np.column_stack([S_tr, I_tr])
    V2_std = np.column_stack([S_std, I_std])
    V2_spc = np.column_stack([S_spc, I_spc])

    meta = {
        "tfidf_dim": int(Xtr_raw.shape[1]),
        "svd_dim": F_tr.shape[1],
        "svd_explained": float(pipe["svd"].explained_variance_ratio_.sum()),
        "pos_pole_terms": len(polar["pos_terms"]),
        "neg_pole_terms": len(polar["neg_terms"]),
        "immune_dim": n_features,
        "mi_selected_dim": k,
        "mi_scores": mi[top_idx].tolist(),
        "mi_top_indices": top_idx.tolist(),
        "y_tr": y_tr, "y_std": y_std, "y_spc": y_spc,
        "pipeline": pipe,
        "polar": polar,
        "M_tr": M_tr, "M_std": M_std, "M_spc": M_spc,
        "protos": protos, "s2_med": s2_med, "iscaler": iscaler, "scaler": scaler,
    }

    FEATS = {
        "A_base": (S_tr, S_std, S_spc),
        "C_v2":   (V2_tr, V2_std, V2_spc),
    }
    return FEATS, meta

# ==================== v1 原论文复现（保留对照） ====================
def v1_features(F, eps=0.1, alpha=0.5):
    """v1 原论文免疫特征（全局4矩+逃逸+耐受，不归一化拼接）"""
    mu = F.mean(axis=1)
    s2 = F.var(axis=1)
    st = (F - mu[:, None]) / (np.sqrt(s2)[:, None] + 1e-12)
    gam = (st ** 3).mean(axis=1)
    kap = (st ** 4).mean(axis=1) - 3.0
    s_esc = np.abs(gam * np.sign(mu)) + alpha * kap
    tol = F * (1.0 - np.exp(-s2[:, None] / eps))
    return np.column_stack([F, tol, s_esc[:, None], mu[:, None], s2[:, None], gam[:, None], kap[:, None]])
