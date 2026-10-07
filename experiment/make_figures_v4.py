# -*- coding: utf-8 -*-
"""论文数据图 v4 —— 视觉美化版（图型沿用 v3，只做视觉层精修）。

美化依据
--------
- academic-figure-skill：语义化 6 色板、字号下限、设计在最终印刷宽度、矢量导出嵌入 TrueType
- scipilot-figure-skill：避开 P4(轴截断)/P8(图例遮盖)/P10(chartjunk)/P11(分辨率)/P17(文字裁切)
- 《图表规范_收集表.md》：图型决策表 + 出版质量硬指标

关键改动
--------
1. 尺寸：按论文 \linewidth = 16.6cm = 6.535in 乘以各图实际排版比例作图，字号不再被缩放
   （旧版 10pt 缩到 0.61 倍 ≈ 6.1pt，低于可读下限）。图例一律放进坐标区或收窄到
   比坐标区窄，避免 savefig tight 把画布撑宽导致二次缩字。
2. 配色：语义化——蓝=基准/主色，红=本文方法（单点强调），橙=对照模型，灰=次要/无贡献。
3. fig_dataset：图例移入坐标区右上空白，消除与标题和总数标签的碰撞。
4. fig_ablation：改成对标 full 的发散条形（Δ 精度），正负贡献一眼可辨，横轴天然从 0 起。
5. fig_escape：三组按语义着色，显著性改用括号标注并留出顶部空间，不再压住须线。
6. fig_subtype：BERT 由极淡的 #c9d2dc 换成饱和橙，避免印刷发糊；标签统一放条端。
7. 导出：PDF 嵌入 TrueType（pdf.fonttype=42），PNG 提到 300 dpi。

第二轮精修（字体/字色/标注摆放，针对"实验图的字发虚"）
----------------------------------------------------------
A. 字体选择：mathtext 改为 custom + Arial。旧版公式符号（$\Delta$、$k$、$p$、
   $\beta$）由 matplotlib 默认的 DejaVu 衬线数学字体渲染，与 Arial 正文混排，
   笔画细、风格不一，在 PDF 里最像"发虚"。出图前断言 Arial 真实存在，
   避免 findfont 静默回退（回退是发虚的头号原因）。
B. 字色：**数据标注文字不再使用浅灰**。GREY #999999 对白底对比度仅 2.9:1
   （WCAG AA 门槛 4.5:1），6 pt 灰色印出来就是"看不清"；统一换成
   SLATE #46525E（≈7.6:1）。灰色只保留给线条、条形等非文字元素。
C. 字号：原 6.2–6.8 pt 的注释全部抬到 ≥7.0 pt（由 NOTE_FS 统一管理）。
D. 摆放：图例一律移到坐标区上方（不再压曲线/贴数据）；k=12 的红色说明由
   竖排改横排；base 参考线标签加偏移，不再贴 y 轴；escape 图收窄顶部空白带、
   箱体改为"淡填充 + 实描边"，避免整体发灰。
"""
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = r"c:\Users\user\Desktop\免疫算法\实验工作区"
FIG = r"c:\Users\user\Desktop\免疫算法\manuscript\figures"
PNG = r"c:\Users\user\Desktop\免疫算法\figures"
os.makedirs(FIG, exist_ok=True)
os.makedirs(PNG, exist_ok=True)

# ---------------- 样式基线（academic-figure-skill typography + export baseline） ----------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    # 公式符号随正文无衬线：matplotlib 默认用 DejaVu 衬线的数学字体，与 Arial
    # 正文混排时笔画明显偏细、风格不一，是"图内文字发虚"的常见来源之一
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    # 图内文字统一用墨色（既不用纯黑，更不用浅灰）
    "text.color": "#1A2430",
    "axes.labelcolor": "#1A2430",
    "xtick.color": "#1A2430",
    "ytick.color": "#1A2430",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,      # 嵌入 TrueType，避免 Type 3 被期刊拒收
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

# Arial 必须真实可用：findfont 会**静默回退**到 DejaVu Sans，回退后字形更细、
# 也不符合投稿要求（Elsevier/Nature 均指定 Helvetica/Arial）。宁可报错不静默出图。
from matplotlib import font_manager as _fm  # noqa: E402
if "arial" not in _fm.findfont(_fm.FontProperties(family="Arial")).lower():
    raise SystemExit("Arial 不可用：字体栈会回退到 DejaVu Sans，请先安装 Arial。")

# 语义化色板（academic-figure-skill color-palettes.md，逐字采用）
# 定性色板：Okabe-Ito（见 配色规范.md）—— 色觉异常可辨，Nature Methods 推荐
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, PURPLE, GREY = "#E69F00", "#CC79A7", "#999999"
SKY = "#56B4E9"
RED = VERMILLION          # 单点强调沿用旧名，值改为 Okabe-Ito vermillion
GRID = "#e8ecf0"
INK = "#1A2430"
# 图内"次级文字"色：对白底对比度 ≈7.6:1（旧值 #6b7683 仅 4.6:1，6 pt 下发虚）。
# 灰色 GREY 从此只用于线条 / 条形 / 虚线基准等非文字元素，不再用于数据标注。
MUTED = "#46525E"
NOTE_FS = 7.0            # 图内注释字号下限（旧版 6.2–6.8 pt 在 PDF 里发虚）
DROP = "#C9D2DC"         # 哑铃图连接线用的中性浅灰（骨架记号，不编码模型身份）

# ---------------- 配色收敛（意见："颜色太多、不够正式"） ----------------
# 原则：颜色只编码**一件事**，且原则上只落在条形 / 标记 / 底色上，
# 文字（轴标签、数值、标题）一律用墨色——这是期刊图与"信息图"最大的观感差别。
# 具体：模型类图只用 3 个色相（蓝=本文方法、橙=BERT、灰=其他基线），
# 骨架类连接线用中性浅灰，高亮底色带用中性浅灰而非第四种色相。
YELLOW = "#F0E442"       # 备用高亮色（仅在确需第四色相强调时使用）
ROLE = {"proposed": BLUE, "hero": BLUE, "bert": ORANGE,
        "positive": GREEN, "baseline": GREY, "base": GREY}
DIV_NEG, DIV_POS = "#B2182B", "#4393C3"   # 发散色板 RdBu 两端（配色规范 §3.4）


def shade(col, a):
    """深浅底带：只调同一色相的透明度，不引入新色相。"""
    return to_rgba(col, a)

# 论文版心：A4 21cm - 2 x 2.2cm = 16.6cm
TEXTW_IN = 16.6 / 2.54  # 6.535 in


def save(fig, name):
    """矢量 PDF 存投稿目录，300dpi PNG 存预览目录。"""
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


R = json.load(open(os.path.join(BASE, "experiment_results.json"), encoding="utf-8"))
AB = json.load(open(os.path.join(BASE, "ablation_results.json"), encoding="utf-8"))
BT = json.load(open(os.path.join(BASE, "bert_v3_results.json"), encoding="utf-8"))

LAB = {"irony": "irony (disguised neg.)",
       "pos_surface": "pos-surface (disguised neg.)",
       "sarcasm_neg": "sarcasm_neg (external)",
       "irony_pos": "irony_pos",
       "synthetic_irony": "synthetic irony",
       "sarcasm_pos": "sarcasm_pos (external)",
       "netslang": "netslang (disguised neg.)",
       "netslang_pos": "netslang_pos"}

# ============ 1) fig_dataset —— 排序堆叠水平条形（\linewidth x 0.88） ============
sp = pd.read_csv(os.path.join(BASE, "my_dataset", "test_special.csv"))
sp["subtype"] = sp["subtype"].astype(str)
keys = [k for k in LAB if k in set(sp.subtype)]
rows = []
for k in keys:
    g = sp[sp.subtype == k]
    rows.append((k, int((g.label == 0).sum()), int((g.label == 1).sum())))
rows.sort(key=lambda t: -(t[1] + t[2]))
neg = np.array([r[1] for r in rows], dtype=float)
pos = np.array([r[2] for r in rows], dtype=float)
tot = neg + pos
y = np.arange(len(rows))
XMAX = float(tot.max()) * 1.17

# 每个子类在 CSV 里都是单一极性（不存在双标签混装），因此只画一条长度=子类规模的
# 条形、用颜色编码真实标签即可；不再重复标注一次内部计数（那是同一数字的冗余副本）。
fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.88, 3.05))
# 全图只用两个色相（蓝=负标签、绿=正标签）。家族信息（伪装负类 / 外部人工标注）
# 已经写在 y 轴标签文字里，不再用第三个色相重复编码；同时去掉行底带与彩色数值——
# 同一份信息由"文字标签 + 条形长度 + 条端直接标注"三重承载，颜色不必再叠一遍。
ax.barh(y, tot, color=[GREEN if r[2] > 0 else BLUE for r in rows], height=0.62,
        zorder=2)
for i, t in enumerate(tot):
    ax.text(t + XMAX * 0.012, i, "%d" % t, va="center", ha="left",
            fontsize=7, color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([LAB[r[0]] for r in rows], fontsize=7)
ax.set_ylim(-0.62, len(rows) - 0.38)
ax.set_xlim(0, XMAX)
ax.set_xlabel("Number of posts (special-public-opinion test set, %s)" % format(int(tot.sum()), ","))
# 图例放进坐标区右上空白（该区所有条形都很短，不会压数据，也不会撑宽画布）
ax.legend(handles=[Patch(facecolor=BLUE, label="negative label"),
                   Patch(facecolor=GREEN, label="positive label")],
          loc="upper right", ncol=2, fontsize=NOTE_FS, handlelength=1.1,
          columnspacing=1.4, borderaxespad=0.5)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_dataset")

# ============ 2) fig_overall —— 双面板：(a) 哑铃图  (b) 平衡准确率/macro-F1 ============
# (b) 面板存在的理由：该特情子集负类占 85%，"多数类全猜负"就能拿到 85.09% 准确率
# （高于 BERT 与 base），只看准确率会误判。把平衡准确率与 macro-F1 画出来，正文
# "应从平衡指标读结论"的论点才图文一致。
mr = R["main_results"]
SHORT = {"Base (TF-IDF+SVD) + GBDT": "Base + GBDT",
         "Naive immunity (v1) + GBDT": "Naive v1 + GBDT",
         "Proposed immunity + SVM": "Proposed + SVM",
         "BERT (fine-tuned)": "BERT (fine-tuned)",
         "Proposed immunity + GBDT": "Proposed + GBDT"}
# 跨图一致的三个色相：灰=基础管线、蓝=本文方法（SVM / GBDT 两个配置同色）、
# 橙=BERT 对照。英雄行不再单独给色相，靠"行标签加粗 + 标记放大 + 数值加粗"强调——
# 删掉紫与天蓝两个只为"丰富"而设的槽位，图的正式感来自克制的用色。
# 每项：(全名, 标准集准确率, 特殊子集准确率, 平衡准确率, macro-F1, 颜色)
# 两个面板共用同一次排序 → 行序一致，读者可沿同一行横向对比。排序依据取**平衡准确率**
# （图注明确说结论应从 (b) 的不平衡感知指标读），使 (b) 仍是规范的"排序条形"；
# 原先 (a) 按准确率排、(b) 按平衡准确率各自排，两面板行序对不上。
models = [("Base (TF-IDF+SVD) + GBDT", mr["A_base+GBDT"]["acc_std"], mr["A_base+GBDT"]["acc_weibo"],
           mr["A_base+GBDT"]["bacc_weibo"], mr["A_base+GBDT"]["mf1_weibo"], ROLE["base"]),
          ("Naive immunity (v1) + GBDT", mr["B_v1+GBDT"]["acc_std"], mr["B_v1+GBDT"]["acc_weibo"],
           mr["B_v1+GBDT"]["bacc_weibo"], mr["B_v1+GBDT"]["mf1_weibo"], ROLE["baseline"]),
          ("Proposed immunity + SVM", mr["C_v2+SVM"]["acc_std"], mr["C_v2+SVM"]["acc_weibo"],
           mr["C_v2+SVM"]["bacc_weibo"], mr["C_v2+SVM"]["mf1_weibo"], ROLE["proposed"]),
          ("BERT (fine-tuned)", BT["std"]["acc"], BT["spc_weibo"]["acc"],
           BT["spc_weibo"]["bacc"], BT["spc_weibo"]["mf1"], ROLE["bert"]),
          ("Proposed immunity + GBDT", mr["C_v2+GBDT"]["acc_std"], mr["C_v2+GBDT"]["acc_weibo"],
           mr["C_v2+GBDT"]["bacc_weibo"], mr["C_v2+GBDT"]["mf1_weibo"], ROLE["hero"])]
models.sort(key=lambda t: t[3])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(TEXTW_IN, 3.45),
                               gridspec_kw=dict(width_ratios=[1.15, 1.0]))
try:
    fig.get_layout_engine().set(wspace=0.13)
except Exception:
    pass

# ---- (a) 哑铃图：标准测试集 vs 特情子集 ----
for i, (name, sd, spv, _bacc, _mf1, col) in enumerate(models):
    hero = name.startswith("Proposed immunity + GBDT")
    # 连接线统一中性浅灰：它是"落差"的图形记号，不表示模型身份，不该随行变色
    axA.plot([sd, spv], [i, i], color=DROP, lw=1.8, zorder=1,
             solid_capstyle="round")
    axA.scatter([sd], [i], s=30, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
    axA.scatter([spv], [i], s=46 if hero else 34, facecolor=col, edgecolor="white",
                lw=0.8, zorder=4)
    # 数值一律墨色：颜色只留在标记上，文字保持中性
    axA.text(spv + 1.15, i, "%.2f" % spv, va="center", ha="left",
             fontsize=NOTE_FS, color=INK,
             fontweight="bold" if hero else "normal")
    axA.text(sd - 0.95, i, "%.2f" % sd, va="center", ha="right",
             fontsize=NOTE_FS, color=MUTED)
axA.set_yticks(range(len(models)))
axA.set_yticklabels([SHORT[m[0]] for m in models], fontsize=7)
# 行标签一律墨色（不再一行一色）：本文方法两行加粗，强调靠字重与标记大小，
# 不靠再引入一个色相
for _t, _m in zip(axA.get_yticklabels(), models):
    _t.set_color(INK)
    if _m[0].startswith("Proposed immunity"):
        _t.set_fontweight("bold")
axA.set_ylim(-0.6, len(models) - 0.4)
axA.set_xlim(70.5, 103.5)
axA.set_xlabel("Accuracy (%)")
axA.grid(axis="x", color=GRID, lw=0.6, zorder=0)
axA.set_axisbelow(True)
axA.legend(handles=[Line2D([], [], marker="o", ls="None", mfc="white", mec=MUTED, ms=4.4,
                           label="standard test"),
                    Line2D([], [], marker="o", ls="None", mfc=MUTED, ms=4.4,
                           label="special subset"),
                    Line2D([], [], color=DROP, lw=1.8, label="accuracy drop")],
           loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=NOTE_FS,
           handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
axA.set_title("(a)", fontsize=8, loc="left", pad=18)

# ---- (b) 平衡准确率 / macro-F1（含"多数类"退化基线作参照）----
mb = R["majority_baseline"]["weibo"]
rowsb = [("Majority class (ref.)", mb["bacc"], mb["mf1"])] + \
        [(SHORT[m[0]], m[3], m[4]) for m in models]
yb = np.arange(len(rowsb)) * 1.58
hb = 0.62
axB.barh(yb + hb / 2, [r[1] for r in rowsb], height=hb, color=BLUE,
         label="balanced accuracy")
# 第二条系列用同一色相的浅色调：同族深浅两色足以区分，不必再换一个色相
axB.barh(yb - hb / 2, [r[2] for r in rowsb], height=hb, color=shade(BLUE, 0.45),
         label="macro-F1")
for i, r in enumerate(rowsb):
    axB.text(r[1] + 1.5, yb[i] + hb / 2, "%.2f" % r[1], va="center",
             fontsize=NOTE_FS, color=INK)
    axB.text(r[2] + 1.5, yb[i] - hb / 2, "%.2f" % r[2], va="center",
             fontsize=NOTE_FS, color=MUTED)
# 参照行铺中性浅底（不再用黄）：一眼看出"多数类全猜负"的退化基线在哪
axB.axhspan(yb[0] - 0.92, yb[0] + 0.92, color="#F2F4F6", zorder=0)
axB.set_yticks(yb)
axB.set_yticklabels([r[0] for r in rowsb], fontsize=7)
for _t, _r in zip(axB.get_yticklabels(), rowsb):
    _t.set_color(INK)
    if _r[0] == "Proposed + GBDT":
        _t.set_fontweight("bold")
axB.set_ylim(yb.min() - 1.05, yb.max() + 1.05)
axB.set_xlim(0, 107)
axB.set_xticks([0, 20, 40, 60, 80, 100])
axB.set_xlabel("Score on the Weibo-derived special subset (%)")
axB.grid(axis="x", color=GRID, lw=0.6)
axB.set_axisbelow(True)
axB.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=NOTE_FS,
           handlelength=1.2, columnspacing=1.4, borderaxespad=0.2)
axB.set_title("(b)", fontsize=8, loc="left", pad=18)
save(fig, "fig_overall")

# ============ 3) fig_ablation —— 对标 full 的发散条形（\linewidth x 0.80） ============
FULL = "Full immunity (all 17 feats)"
BASEK = "w/o immunity (base only)"
SHOW = {FULL: r"Full immunity ($17\to12$ feats)",
        "w/o polarity-region moments": "w/o polarity-region moments",
        "w/o dominance & conflict": "w/o dominance & conflict",
        "w/o escape channel": "w/o escape channel",
        "w/o memory affinity": "w/o memory affinity",
        "w/o tolerance gate": "w/o tolerance gate",
        BASEK: "w/o immunity (base)"}
full_v = AB[FULL]["weibo_acc"]
items = []
for k, lab in SHOW.items():
    if k not in AB:
        continue
    v = AB[k]["weibo_acc"]
    items.append((lab, v - full_v, v, k == FULL))
items.sort(key=lambda t: t[1])

fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.80, 2.95))
# Δ 是有正负的连续量 → 按《配色规范.md》§3.4 用发散色板 RdBu（两端对称、中点近白），
# 比"蓝 vs 灰"方向感更强，也把色彩丰富度提上来
for i, (lab, d, v, isf) in enumerate(items):
    if isf:
        ax.axhspan(i - 0.5, i + 0.5, color=shade(YELLOW, 0.30), zorder=0)
        ax.barh(i, 0.0, color=DIV_POS, height=0.58)
    else:
        ax.barh(i, d, color=DIV_NEG if d < 0 else DIV_POS, height=0.58,
                edgecolor="white", lw=0.4)
    ax.text(d + (0.10 if d >= 0 or abs(d) < 0.3 else -0.10), i, "%.2f%%" % v,
            va="center", ha="left" if (d >= 0 or abs(d) < 0.3) else "right",
            fontsize=7, color=DIV_NEG if d < 0 else (INK if isf else MUTED))
ax.axvline(0, color=INK, lw=0.9, zorder=3)
ax.set_yticks(np.arange(len(items)))
ax.set_yticklabels([t[0] for t in items], fontsize=7.5)
ax.set_ylim(-0.62, len(items) - 0.38)
ax.set_xlim(-9.0, 2.3)
ax.set_xlabel(r"$\Delta$ accuracy vs. full model (percentage points)")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
# 图例放在坐标区上方且刻意收短：既不压条形，也不比坐标区宽（避免撑宽画布）
ax.legend(handles=[Patch(facecolor=DIV_NEG, label="removal hurts ($\Delta$ < 0)"),
                   Patch(facecolor=DIV_POS, label="removal does not hurt ($\Delta$ $\geq$ 0)")],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7,
          handlelength=1.2, columnspacing=1.3, borderaxespad=0.2)
save(fig, "fig_ablation")

# ============ 4) fig_feature_importance —— MI 排序水平条形（\linewidth x 0.90） ============
MAP = {r"pos $\mu$": r"pos $\mu$", r"pos $\sigma^2$": r"pos $\sigma^2$",
       r"pos $\gamma_1$": r"pos $\gamma_1$", r"pos $\gamma_2$": r"pos $\kappa$",
       r"neg $\mu$": r"neg $\mu$", r"neg $\sigma^2$": r"neg $\sigma^2$",
       r"neg $\gamma_1$": r"neg $\gamma_1$", r"neg $\gamma_2$": r"neg $\kappa$",
       "dominance": "dominance", "conflict": "conflict", "escape": "escape",
       "esc_tol": r"escape$^\star$", "gate": "gate",
       "aff_pos_best": "affinity (pos, best)", "aff_pos_mean": "affinity (pos, mean)",
       "aff_neg_best": "affinity (neg, best)", "aff_neg_mean": "affinity (neg, mean)"}
fi = sorted(R["mi_feature_importance"], key=lambda d: d["mi"])
labels = [MAP.get(d["feature"], d["feature"]) for d in fi]
vals = [d["mi"] for d in fi]
fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.90, 3.55))
y = np.arange(len(fi))
cols = [BLUE if l.startswith("pos") else (PURPLE if l.startswith("neg") else GREEN)
        for l in labels]
ax.barh(y, vals, color=cols, height=0.64)
xmax = max(vals) * 1.28
for i, v in enumerate(vals):
    ax.text(v + xmax * 0.012, i, "%.4f" % v, va="center", fontsize=7, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_ylim(-0.62, len(fi) - 0.38)
ax.set_xlim(0, xmax)
ax.set_xlabel("Mutual information with the label")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor=BLUE, label="positive-region moments"),
                   Patch(facecolor=PURPLE, label="negative-region moments"),
                   Patch(facecolor=GREEN, label="derived channels")],
          loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=7,
          handlelength=1.2, columnspacing=1.3, borderaxespad=0.2)
save(fig, "fig_feature_importance")

# ============ 5) fig_escape —— 箱线图（\linewidth x 0.62） ============
sys.path.insert(0, BASE)
from immune_pipeline import (load_data, build_tfidf_svd, learn_polarity_regions,
                             polarity_moments)
train, std, spc = load_data(BASE)
pipe = build_tfidf_svd(train, std, spc)
pol = learn_polarity_regions(pipe["Xtr_raw"], train.label.values, pipe["vocab"])
Ms = polarity_moments(pipe["Xstd_raw"], pol["pos_cols"], pol["neg_cols"])
Mp = polarity_moments(pipe["Xspc_raw"], pol["pos_cols"], pol["neg_cols"])
esc = lambda M: np.minimum(M[:, 0], M[:, 4]) * (1 + np.abs(M[:, 3]) + np.abs(M[:, 7]))
e_spc, e_std = esc(Mp), esc(Ms)
y_spc, y_std = np.asarray(spc.label.values), np.asarray(std.label.values)
sub_spc = np.asarray(spc.subtype.astype(str))
g1 = e_spc[np.isin(sub_spc, ["irony", "synthetic_irony", "sarcasm_neg"]) & (y_spc == 0)]
g2 = e_std[y_std == 0]
g3 = e_std[y_std == 1]
ev = R["escape_validation"]

fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.62, 2.85))
# 语义对齐《配色规范.md》§3.2：正向标签用 Bluish Green，橙色专属于 BERT 对照模型
groups = [(g1, BLUE, "Disguised-negative\nirony family"),
          (g2, PURPLE, "Ordinary\nnegative"),
          (g3, GREEN, "Ordinary\npositive")]
_rng = np.random.default_rng(7)
for i, (g, col, _lab) in enumerate(groups, start=1):
    # 散点云：画面有层次，同时让读者看到分布本身（箱线只给四分位）
    ax.scatter(i + _rng.uniform(-0.16, 0.16, size=len(g)), g, s=2.0,
               color=shade(col, 0.5), lw=0, zorder=2)
    ax.boxplot([g], positions=[i], widths=0.46, patch_artist=True, showmeans=True,
               showfliers=False,
               medianprops=dict(color=INK, lw=1.4),
               meanprops=dict(marker="D", markerfacecolor="white",
                              markeredgecolor=INK, ms=3.6, markeredgewidth=0.7),
               # 淡填充 + 不透明描边：旧版整块 alpha=0.42 连轮廓一起变淡，
               # 印出来像"没印实"；现在只让填充半透明，轮廓保持实色。
               boxprops=dict(facecolor=to_rgba(col, 0.34), edgecolor=col, lw=1.0),
               whiskerprops=dict(color=col, lw=1.0),
               capprops=dict(color=col, lw=1.0))
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["%s\n($n$=%s)" % (lb, format(len(g), ","))
                    for g, _c, lb in groups], fontsize=7)
ax.set_xlim(0.45, 3.55)
ax.set_ylabel("Immune-escape score")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)

# 顶部预留空间，显著性括号抬到须线之上；预留量由 1.24 收到 1.16——
# 旧版在括号与统计量行之间空出一条带，整图上部显得松散
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi * 1.16)
brack = hi * 1.05
ax.plot([1, 1, 2, 2], [brack * 0.975, brack, brack, brack * 0.975], color=INK, lw=0.7)
p = ev["p_value"]
ptxt = (r"$p$ = %.1f$\times$10$^{%d}$" % (p / 10 ** int(np.floor(np.log10(p))),
                                          int(np.floor(np.log10(p))))) \
    if p < 1e-3 else (r"$p$ = %.3f" % p)
ax.text(1.5, brack * 1.01, ptxt, ha="center", va="bottom", fontsize=7, color=INK)
ax.text(0.5, 1.012, r"Welch $t$ = %.2f,  Cohen's $d$ = %.3f"
        % (ev["welch_t"], ev["cohens_d"]),
        transform=ax.transAxes, ha="center", va="bottom", fontsize=NOTE_FS,
        color=MUTED)
save(fig, "fig_escape")

# ============ 6) fig_subtype —— 分组水平条形（\linewidth x 0.90） ============
sa_im = R["subtype_acc"]["C_v2+GBDT"]
sa_bt = BT["subtype_acc"]
subs = [k for k in LAB if k in sa_im]
subs.sort(key=lambda k: -sa_im[k])
v_im = np.array([sa_im[k] for k in subs], dtype=float)
v_bt = np.array([sa_bt[k] for k in subs], dtype=float)
yy = np.arange(len(subs)) * 2.0
h = 0.74

fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.90, 3.60))
ax.barh(yy + h / 2, v_bt, height=h, color=ORANGE, label="BERT (fine-tuned)")
ax.barh(yy - h / 2, v_im, height=h, color=BLUE, label="Proposed (immunity + GBDT)")
for i in range(len(subs)):
    ax.text(v_bt[i] + 1.2, yy[i] + h / 2, "%.1f" % v_bt[i], va="center",
            fontsize=7, color=MUTED)
    ax.text(v_im[i] + 1.2, yy[i] - h / 2, "%.1f" % v_im[i], va="center",
            fontsize=7, color=INK)
ax.set_yticks(yy)
ax.set_yticklabels([LAB[k] for k in subs], fontsize=7.5)
# 子类家族已写在 y 轴标签文字里（"(disguised neg.)" / "(external)"），不再给标签
# 上色：本图颜色只承担"BERT vs 本文方法"两条系列的区分，避免与 fig_dataset 冲突
ax.set_ylim(yy.min() - 1.15, yy.max() + 1.15)
ax.set_xlim(0, 112)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Per-subtype accuracy on the special test set (%)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7,
          handlelength=1.4, columnspacing=1.4, borderaxespad=0.2)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_subtype")


# ================================================================
# B 组新增图（数据来自 run_b_group.py -> b_group_results.json）
# ================================================================
BG = json.load(open(os.path.join(BASE, "b_group_results.json"), encoding="utf-8"))
BASE_ACC = BG["base_only"]["acc"]          # 纯基础特征在微博衍生子集上的准确率


def _base_ref(ax, label=True):
    """基础管线参考线：免疫特征必须超过它才有意义。

    标签属于**数据标注**（给出具体数值），所以用深色 MUTED 而不是浅灰 GREY：
    旧版 6.2 pt 浅灰印出来几乎看不清；位置同步右移、上移，不再贴着 y 轴。
    """
    ax.axhline(BASE_ACC, color=GREY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(ax.get_xlim()[0] + 0.30, BASE_ACC + 0.28,
                "base features only (%.2f%%)" % BASE_ACC,
                ha="left", va="bottom", fontsize=NOTE_FS, color=MUTED)


# ============ 7) fig_confusion —— 三个模型的混淆矩阵（微博衍生子集） ============
cn = BG["confusion"]
panels = [("Base + GBDT", cn["base"], MUTED),   # 标题用深灰：浅灰标题发虚
          ("BERT (fine-tuned)", cn["bert"], ORANGE),
          ("Proposed immunity", cn["immunity"], BLUE)]
fig, axes = plt.subplots(1, 3, figsize=(TEXTW_IN, 2.62))
for j, (name, cm, col) in enumerate(panels):
    ax = axes[j]
    M = np.array(cm, dtype=float)
    tot = M.sum()
    acc = (M[0, 0] + M[1, 1]) / tot * 100
    ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max())
    for r in range(2):
        for c in range(2):
            v = int(M[r, c])
            frac = M[r, c] / M[r].sum() * 100 if M[r].sum() else 0
            rgba = plt.get_cmap("viridis")(M[r, c] / M.max())
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "#FFFFFF" if lum < 0.52 else "#1A2430"
            ax.text(c, r - 0.13, "%d" % v, ha="center", va="center", fontsize=8,
                    color=tc, fontweight="bold")
            ax.text(c, r + 0.22, "%.1f%%" % frac, ha="center", va="center",
                    fontsize=NOTE_FS, color=tc)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"], fontsize=NOTE_FS)
    ax.set_yticklabels(["true neg", "true pos"], fontsize=NOTE_FS)
    if j:
        ax.set_yticklabels([])
    ax.set_title("%s\nacc %.2f%%" % (name, acc), fontsize=7.6, pad=4,
                 color=col, fontweight="bold")
    ax.tick_params(length=0)
    ax.grid(False)
    # 彩色外框：与面板标题同色，三块矩阵各自成组，比纯白边框有层次
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(col)
        sp.set_linewidth(1.2)
    # 行归一化百分比写在格内，说明"有多少真实正类被漏掉"
fig.suptitle("Confusion matrices on the Weibo-derived special subset (4,453 records); "
             "row percentages below each count", fontsize=7.9, y=1.05)
save(fig, "fig_confusion")

# ============ 8) fig_topk —— 保留的免疫特征数 k 的选择 ============
tk = BG["topk"]
ks = [d["k"] for d in tk]
fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.82, 3.0))
_lo = min([d["acc"] for d in tk] + [d["bacc"] for d in tk] + [BASE_ACC]) - 1.1
ax.fill_between(ks, [d["acc"] for d in tk], _lo, color=shade(BLUE, 0.10), zorder=0)
ax.plot(ks, [d["acc"] for d in tk], marker="o", ms=3.4, lw=1.4, color=BLUE,
        label="accuracy")
ax.plot(ks, [d["bacc"] for d in tk], marker="s", ms=3.2, lw=1.4, color=SKY,
        label="balanced accuracy")
K = 12
# 上下留白显式给出：右上要放 k=12 说明，底部要放 base 参考线标签
_acc = [d["acc"] for d in tk]
_bac = [d["bacc"] for d in tk]
ax.set_ylim(min(_acc + _bac + [BASE_ACC]) - 1.1, max(_acc + _bac) + 2.8)
ax.axvspan(K - 0.55, K + 0.55, color=shade(YELLOW, 0.38), zorder=0)
ax.axvline(K, color=RED, lw=1.0, ls=(0, (3, 2)), zorder=1)
# 旧版把说明竖排在轴底部（6.4 pt 竖排极难读）；改为横排放在 vline 左上空白处
ax.text(K - 0.45, max(_acc + _bac) + 2.55, "retained in this work ($k$=12)",
        ha="right", va="top", fontsize=NOTE_FS, color=RED, fontweight="bold")
_base_ref(ax)
ax.set_xlabel("Number of immunity features retained by mutual information ($k$)")
ax.set_ylabel("Score on the Weibo-derived subset (%)")
ax.set_xticks(range(1, 18, 2))
ax.set_xlim(0.5, 17.5)
# 图例移到坐标区上方：右上角恰是 bal.-acc 曲线的平坦段，旧位置几乎贴住曲线
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=NOTE_FS,
          handlelength=1.4, columnspacing=1.4, borderaxespad=0.2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_topk")

# ============ 9) fig_sensitivity —— 方法自身超参敏感性 ============
fig, axes = plt.subplots(1, 2, figsize=(TEXTW_IN * 0.86, 2.85))
specs = [("z_threshold", "t", r"Polarity-region threshold $t$", 1.5, axes[0]),
         ("beta", "beta", r"Memory prototype update rate $\beta$", 0.3, axes[1])]
# 两个面板共用同一 y 范围，否则并排显示会被误读成"β 更稳"
_allv = [r["acc"] for r in BG["z_threshold"]] + [r["acc"] for r in BG["beta"]]
# 两个面板共用同一 y 范围，否则并排显示会被误读成"β 更稳"
_allv = [r["acc"] for r in BG["z_threshold"]] + [r["acc"] for r in BG["beta"]]
YLIM = (min(_allv + [BASE_ACC]) - 0.9, max(_allv) + 1.2)
for key, col, title, dflt, ax in specs:
    d = BG[key]
    xs = [r[col] for r in d]
    ys = [r["acc"] for r in d]
    # 甜点区浅绿底带：落在最优值 1.0 个百分点内的设置——底色也在传信息
    _best = max(ys)
    _ok = [x for x, v in zip(xs, ys) if v >= _best - 1.0]
    ax.axvspan(min(_ok) - 0.05, max(_ok) + 0.05, color=shade(GREEN, 0.16), zorder=0)
    ax.plot(xs, ys, marker="o", ms=4.0, lw=1.5, color=BLUE)
    ax.scatter([dflt], [ys[xs.index(dflt)]], s=78, facecolor=RED, edgecolor="white",
               lw=0.9, zorder=5)
    ax.annotate("default", xy=(dflt, ys[xs.index(dflt)]),
                xytext=(7, -10), textcoords="offset points", fontsize=NOTE_FS,
                color=RED, fontweight="bold")
    _base_ref(ax)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(title)
    ax.set_ylabel("Weibo-derived subset\naccuracy (%)" if ax is axes[0] else "")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
fig.suptitle("Sensitivity to the two hyper-parameters of the immunity branch",
             fontsize=8, y=1.02)
save(fig, "fig_sensitivity")

# ============ 10) fig_trainsize —— 训练集规模的影响 ============
tn = BG["trainsize"]
xs = np.arange(len(tn))
b = [d["base_acc"] for d in tn]
m = [d["imm_acc"] for d in tn]
fig, ax = plt.subplots(figsize=(TEXTW_IN * 0.74, 2.95))
# 两曲线之间的浅绿填充 = 免疫增益区域：把最核心的结论直接画成色块
ax.fill_between(xs, b, m, color=shade(GREEN, 0.16), zorder=0)
ax.plot(xs, b, marker="o", ms=4.2, lw=1.5, color=ROLE["base"], label="Base + GBDT")
ax.plot(xs, m, marker="o", ms=4.6, lw=1.6, color=BLUE, label="Proposed immunity")
for i, d in enumerate(tn):
    ax.annotate("", xy=(xs[i], m[i]), xytext=(xs[i], b[i]),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.0, shrinkA=1, shrinkB=1))
    ax.text(xs[i] + 0.12, (b[i] + m[i]) / 2, "$\\Delta$ %+.2f" % d["delta"],
            ha="left", va="center", fontsize=NOTE_FS, color=RED)
ax.set_xticks(xs)
ax.set_xticklabels([format(d["n"], ",") for d in tn])
ax.set_xlim(-0.35, len(tn) - 0.35)
ax.set_xlabel("Training-set size (balanced posts)")
ax.set_ylabel("Weibo-derived subset accuracy (%)")
ax.set_ylim(min(b + m) - 1.6, max(b + m) + 2.4)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=NOTE_FS,
          handlelength=1.4, columnspacing=1.4, borderaxespad=0.2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig_trainsize")

print("ALL DONE")
