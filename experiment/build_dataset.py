# -*- coding: utf-8 -*-
"""
构建专属数据集 v3：标准训练/测试集 + 特殊舆情测试集
改进：
  1. 训练集扩大到 24,000 纯微博（充分利用 119K 语料，且不含外部反讽样本）
  2. 特殊测试集合并 YiMeng-SYSU 真实标注反讽（sarcasm_neg / sarcasm_pos）作外部验证
  3. 保留原有模板合成策略（450 条）作为对照
关键结论（见 diag_repro.py）：外部反讽 Normal 样本若混入训练集会污染极区学习与
  记忆原型，导致免疫特征退化（C_v2: 84.37% -> 79.97%），故训练集保持纯微博。
数据源：
  - weibo_senti_100k（119,988 条真实微博，正负均衡）
  - YiMeng-SYSU/chinese-logic-sentiment-dataset（仅取反讽子类作测试）
"""
import os
import re
import json
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(BASE, "datasets")
OUT = os.path.join(BASE, "my_dataset")
os.makedirs(OUT, exist_ok=True)

# ==================== 加载微博数据 ====================
df = pd.read_csv(os.path.join(DS, "weibo_senti_100k.csv"))
df = df.dropna(subset=["review"]).reset_index(drop=True)
df["review"] = df["review"].astype(str)
df = df[df["review"].str.len() >= 4].reset_index(drop=True)
print("微博语料清洗后总量:", len(df))

# ==================== 加载反讽数据集 ====================
sarcasm_frames = []
for fname in ["sarcasm_train.csv", "sarcasm_val.csv", "sarcasm_test.csv"]:
    fp = os.path.join(DS, fname)
    if os.path.exists(fp):
        s = pd.read_csv(fp)
        s["text"] = s["text"].astype(str)
        s = s.dropna(subset=["text"])
        s = s[s["text"].str.len() >= 4].reset_index(drop=True)
        sarcasm_frames.append(s)
        print(f"  {fname}: {len(s)} 条")

sarcasm_all = pd.concat(sarcasm_frames, ignore_index=True) if sarcasm_frames else pd.DataFrame()
print(f"反讽数据集总量: {len(sarcasm_all)}")

# 按子类型分拆
# Negative_Ironic: 表面正面/中性，实际负面 -> 加入特殊负面
# Positive_Ironic: 表面负面/中性，实际正面 -> 加入特殊正面
# Normal (转折/双重否定/简单句): 加入标准训练集增强
sarcasm_special_neg = sarcasm_all[sarcasm_all["sub_type"] == "Negative_Ironic"].copy() if not sarcasm_all.empty else pd.DataFrame()
sarcasm_special_pos = sarcasm_all[sarcasm_all["sub_type"] == "Positive_Ironic"].copy() if not sarcasm_all.empty else pd.DataFrame()
sarcasm_train_pool = sarcasm_all[sarcasm_all["sub_type"] == "Normal"].copy() if not sarcasm_all.empty else pd.DataFrame()
print(f"  反讽负面(Negative_Ironic): {len(sarcasm_special_neg)}")
print(f"  反讽正面(Positive_Ironic): {len(sarcasm_special_pos)}")
print(f"  常规逻辑(Normal，加入训练): {len(sarcasm_train_pool)}")

# ==================== 特殊舆情标记定义 ====================
IRONIC_MARKS = ["[微笑]", "[呵呵]", "[doge]", "[哼]", "[白眼]", "[挖鼻]", "[允悲]",
                "呵呵", "真是谢谢", "真有你的", "也是服了", "就这", "就这？", "绝了",
                "真是绝了", "好家伙", "我谢谢你", "真棒呢", "真贴心", "可真是"]
NETSLANG_MARKS = ["蚌埠住了", "绷不住了", "离大谱", "破防", "麻了", "真的会谢",
                  "无语住了", "醉了", "服了", "吐了", "裂开", "emo", "emo了",
                  "下头", "逆天", "典中典", "孝死", "赢麻了", "急了", "急了吧"]
POS_SURFACE = ["喜欢", "厉害", "优秀", "完美", "好看", "支持", "点赞", "哈哈",
               "[哈哈]", "[偷笑]", "[赞]", "真好", "太好了", "牛", "大佬"]

def has_any(text, marks):
    return any(m in text for m in marks)

def n_pos_surface(text):
    return sum(text.count(m) for m in POS_SURFACE)

# ==================== 微博特殊舆情采样 ====================
special_neg = []
special_pos = []
used_idx = set()

# (a) 反讽标记 + 负面标签
cand = df[(df.label == 0) & (df.review.apply(lambda t: has_any(t, IRONIC_MARKS)))]
print("\n(a) 反讽标记+负面:", len(cand))
for _, r in cand.iterrows():
    special_neg.append({"text": r.review, "label": 0, "subtype": "irony"})
    used_idx.add(r.name)

# (b) 网络用语 + 负面
cand = df[(df.label == 0) & (df.review.apply(lambda t: has_any(t, NETSLANG_MARKS)))]
cand = cand[~cand.index.isin(used_idx)]
print("(b) 网络用语+负面:", len(cand))
for _, r in cand.iterrows():
    special_neg.append({"text": r.review, "label": 0, "subtype": "netslang"})
    used_idx.add(r.name)

# (c) 表面正面词>=3 + 负面标签
cand = df[(df.label == 0) & (df.review.apply(lambda t: n_pos_surface(t) >= 3))]
cand = cand[~cand.index.isin(used_idx)].sample(frac=1.0, random_state=42)
print("(c) 表面正面堆砌+负面:", len(cand))
for _, r in cand.head(1200).iterrows():
    special_neg.append({"text": r.review, "label": 0, "subtype": "pos_surface"})
    used_idx.add(r.name)

# (d) 网络用语 + 正面标签
cand = df[(df.label == 1) & (df.review.apply(lambda t: has_any(t, NETSLANG_MARKS)))]
cand = cand[~cand.index.isin(used_idx)]
print("(d) 网络用语+正面:", len(cand))
for _, r in cand.head(800).iterrows():
    special_pos.append({"text": r.review, "label": 1, "subtype": "netslang_pos"})
    used_idx.add(r.name)

# (e) 反讽标记 + 正面标签
cand = df[(df.label == 1) & (df.review.apply(lambda t: has_any(t, IRONIC_MARKS)))]
cand = cand[~cand.index.isin(used_idx)]
print("(e) 反讽标记+正面:", len(cand))
for _, r in cand.head(500).iterrows():
    special_pos.append({"text": r.review, "label": 1, "subtype": "irony_pos"})
    used_idx.add(r.name)

# ==================== 合并真实标注反讽到特殊测试集 ====================
# Negative_Ironic -> sarcasm_neg_irony
for _, r in sarcasm_special_neg.iterrows():
    special_neg.append({"text": r["text"], "label": int(r["label"]), "subtype": "sarcasm_neg"})
print(f"(f) 真实反讽负面(sarcasm_neg): {len(sarcasm_special_neg)}")

# Positive_Ironic -> sarcasm_pos_irony
for _, r in sarcasm_special_pos.iterrows():
    special_pos.append({"text": r["text"], "label": int(r["label"]), "subtype": "sarcasm_pos"})
print(f"(g) 真实反讽正面(sarcasm_pos): {len(sarcasm_special_pos)}")

# ==================== 模板合成（反讽句式） ====================
irony_templates = [
    "这服务真是{adj}呢，我真是{emo}了", "这也叫{name}？真是开了眼了",
    "{name}可真是太厉害了，把我整{emo}了", "真是有够{name}的，我也是服了",
    "谢谢您嘞，真是太贴心了，下次别贴心了", "好家伙，这{adj}程度我也是第一次见",
    "真是绝了，这体验感直接拉满，拉到{emo}", "就这？就这点水平也敢出来营业？",
    "我真是谢谢你了，帮倒忙第一名", "这操作，{emo}，真是高手",
    "不愧是{name}，总能整出让我{emo}的花活", "太棒了，棒到我想{emo}了",
    "真是越活越回去了，这波操作满分", "哇塞，好厉害哦，都能把事办砸了诶",
    "真是谢谢你，让我见识到了下限", "这水平，不拿个倒数第一都可惜了",
    "行，你们真是好样的，服气服气", "呵呵，真是精彩，精彩得让人说不出话",
]
tpl_adjs = ["贴心", "周到", "完美", "专业", "厉害", "优秀"]
tpl_names = ["服务", "体验", "产品", "客服", "质量", "设计", "口碑", "速度"]
tpl_emos = ["无语", "破防", "麻了", "蚌埠住", "裂开", "吐血", "emo", "裂开到极点"]

rng = random.Random(42)
synth = []
for t in irony_templates:
    for _ in range(25):
        text = t.replace("{adj}", rng.choice(tpl_adjs)).replace("{name}", rng.choice(tpl_names)).replace("{emo}", rng.choice(tpl_emos))
        synth.append({"text": text, "label": 0, "subtype": "synthetic_irony"})
print(f"(h) 模板合成反讽: {len(synth)}")

special_all = special_neg + special_pos + synth
rng.shuffle(special_all)

# ==================== 标准训练/测试集 ====================
pool = df[~df.index.isin(used_idx)]
pos_pool = pool[pool.label == 1]
neg_pool = pool[pool.label == 0]
print(f"\n微博剩余池: pos={len(pos_pool)}, neg={len(neg_pool)}")

# v3: 训练集扩大到 24,000 纯微博（12K pos + 12K neg）
train_n = 24000
test_n = 2000

train_pos = pos_pool.sample(train_n // 2, random_state=42)
train_neg = neg_pool.sample(train_n // 2, random_state=42)
train = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=42)

remaining_pos = pos_pool.drop(train_pos.index, errors="ignore")
remaining_neg = neg_pool.drop(train_neg.index, errors="ignore")
std_test = pd.concat([remaining_pos.sample(test_n // 2, random_state=43),
                      remaining_neg.sample(test_n // 2, random_state=43)]).sample(frac=1.0, random_state=43)

train_df = train.rename(columns={"review": "text"})[["text", "label"]]
std_test_df = std_test.rename(columns={"review": "text"})[["text", "label"]]
sp_df = pd.DataFrame(special_all)

# 注意：外部反讽 Normal 样本不并入训练集（会污染极区学习与记忆原型，见 diag_repro.py）

# ==================== 保存 ====================
train_df.to_csv(os.path.join(OUT, "train_standard.csv"), index=False, encoding="utf-8-sig")
std_test_df.to_csv(os.path.join(OUT, "test_standard.csv"), index=False, encoding="utf-8-sig")
sp_df.to_csv(os.path.join(OUT, "test_special.csv"), index=False, encoding="utf-8-sig")

print("\n===== 专属数据集构建完成 v2 =====")
print(f"标准训练集: {len(train_df)}")
print(f"标准测试集: {len(std_test_df)}")
print(f"特殊舆情测试集: {len(sp_df)}")
print("\n特殊舆情子类分布:")
print(sp_df["subtype"].value_counts())
print("\n特殊舆情标签分布:")
print(sp_df["label"].value_counts())
print(f"\n正类占比: {sp_df['label'].sum()}/{len(sp_df)} = {sp_df['label'].mean()*100:.1f}%")
print("\n特殊舆情示例(每类2条):")
for st in sp_df["subtype"].unique():
    sub = sp_df[sp_df.subtype == st].head(2)
    for _, r in sub.iterrows():
        print(f"  [{st}][label={r.label}] {r.text[:80]}")
