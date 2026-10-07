# -*- coding: utf-8 -*-
"""
BERT 基线（v3）：在新数据集上重跑，保证与免疫方法同数据同评估口径
  - 训练：train_standard.csv（24,000 纯微博）
  - 评估：标准测试集(2000) / 特殊测试集(5774) 及微博衍生(4453)、外部反讽(1321)子集
  - 配置与原文一致：bert-base-chinese, max_length=64, 3 epochs, lr=2e-5
模型权重：本地 .hf/bert-base-chinese（离线运行）
"""
import os, json, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import warnings; warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE), ".hf", "bert-base-chinese")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXL, BS, LR, EPOCHS, WD = 64, 32, 2e-5, 3, 0.01
WEIBO_SUB = ["irony", "pos_surface", "irony_pos", "synthetic_irony", "netslang", "netslang_pos"]
EXT_SUB = ["sarcasm_neg", "sarcasm_pos"]


class TD(Dataset):
    def __init__(self, texts, labels, tk):
        self.texts, self.labels, self.tk = texts, labels, tk
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e = self.tk(self.texts[i], max_length=MAXL, truncation=True, padding="max_length")
        return {k: torch.tensor(v) for k, v in e.items()}, torch.tensor(self.labels[i])


def load():
    tr = pd.read_csv(os.path.join(BASE, "my_dataset", "train_standard.csv"))
    sd = pd.read_csv(os.path.join(BASE, "my_dataset", "test_standard.csv"))
    sp = pd.read_csv(os.path.join(BASE, "my_dataset", "test_special.csv"))
    for d in (tr, sd, sp): d["text"] = d["text"].astype(str)
    return tr, sd, sp


def run():
    print(f"device={DEVICE} torch={torch.__version__}", flush=True)
    train, std, spc = load()
    print(f"train={len(train)} std={len(std)} spc={len(spc)}", flush=True)
    tk = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=2, local_files_only=True).to(DEVICE)

    dl_tr = DataLoader(TD(train.text.tolist(), train.label.values.tolist(), tk),
                       batch_size=BS, shuffle=True, drop_last=True)
    dl_sd = DataLoader(TD(std.text.tolist(), std.label.values.tolist(), tk), batch_size=128)
    dl_sp = DataLoader(TD(spc.text.tolist(), spc.label.values.tolist(), tk), batch_size=128)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    wu = max(1, len(dl_tr) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / wu, 1.0))

    model.train(); T0 = time.time()
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for e, y in dl_tr:
            e = {k: v.to(DEVICE) for k, v in e.items()}; y = y.to(DEVICE)
            out = model(**e, labels=y)
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += out.loss.item(); nb += 1
        print(f"[ep{ep}] loss={tot/nb:.4f} {time.time()-T0:.0f}s", flush=True)

    model.eval()
    def infer(dl):
        ps = []
        with torch.no_grad():
            for e, _ in dl:
                e = {k: v.to(DEVICE) for k, v in e.items()}
                ps.extend(model(**e).logits.argmax(-1).cpu().tolist())
        return np.array(ps)

    p_sd, p_sp = infer(dl_sd), infer(dl_sp)
    y_sd, y_sp = std.label.values, spc.label.values
    m_w = spc.subtype.isin(WEIBO_SUB).values
    m_e = spc.subtype.isin(EXT_SUB).values

    def mt(y, p):
        return {"acc": round(accuracy_score(y, p) * 100, 2),
                "bacc": round(balanced_accuracy_score(y, p) * 100, 2),
                "mf1": round(f1_score(y, p, average="macro") * 100, 2)}

    out = {"config": {"max_length": MAXL, "epochs": EPOCHS, "lr": LR, "batch": BS, "device": DEVICE},
           "std": mt(y_sd, p_sd), "spc_all": mt(y_sp, p_sp),
           "spc_weibo": mt(y_sp[m_w], p_sp[m_w]), "spc_ext": mt(y_sp[m_e], p_sp[m_e]),
           "runtime_s": round(time.time() - T0, 1)}
    sub = {}
    for s in spc.subtype.unique():
        mm = (spc.subtype == s).values
        sub[s] = round(accuracy_score(y_sp[mm], p_sp[mm]) * 100, 2)
    out["subtype_acc"] = sub
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    np.save(os.path.join(BASE, "bert_v3_pred_std.npy"), p_sd)
    np.save(os.path.join(BASE, "bert_v3_pred_spc.npy"), p_sp)
    json.dump(out, open(os.path.join(BASE, "bert_v3_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved bert_v3_results.json", flush=True)


if __name__ == "__main__":
    run()
