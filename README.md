# Immunity-Inspired Feature Framework for Chinese Sentiment Classification

Reproduction code for the paper **"An Immunity-Inspired Feature Framework for Challenging
Chinese Sentiment Classification"**.

The method builds four interpretable feature channels on the sentiment-polarity distribution —
polarity-region statistical moments, a conflict (escape) score, a variance-based tolerance gate,
and class-prototype memory affinities — reduces them by mutual information, and concatenates the
selected channels with a 300-dimensional SVD compression of TF-IDF features. The fused vector is
fed to a linear SVM or a gradient-boosted classifier; no new neural architecture is introduced.

Reported headline result on the Weibo-derived special subset (4,453 records):
**83.85% accuracy / 89.08% balanced accuracy / 76.83% macro-F1**, against 76.82 / 85.14 / 69.92
for the base TF-IDF+SVD pipeline and 77.81 / 82.86 / 70.03 for the evaluated fine-tuned
`bert-base-chinese` configuration.

---

## 1. Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

Python 3.11. Versions in `requirements.txt` are those that produced every number and figure in
the paper. `torch` / `transformers` are needed only for the BERT baseline; install them for your
own CUDA version.

## 2. Data preparation

**The raw corpora are not redistributed here** (platform terms, user-generated content and
licensing). Place the following files yourself:

| File | Description | Location |
|---|---|---|
| `weibo_senti_100k.csv` | Chinese Weibo sentiment corpus (binary labels) | `experiment/datasets/` |
| `sarcasm_train.csv`, `sarcasm_val.csv`, `sarcasm_test.csv` | externally annotated irony data | `experiment/datasets/` |

Then build the study dataset used in the paper:

```bash
cd experiment
python build_dataset.py
```

This writes `experiment/my_dataset/`:

| File | Size | Purpose |
|---|---|---|
| `train_standard.csv` | 24,000 | Weibo-only training set (external irony items are kept out, so that the polarity regions and memory prototypes are not contaminated) |
| `test_standard.csv` | 2,000 | in-distribution standard test set |
| `test_special.csv` | 5,774 | challenge set = 4,453 corpus-derived Weibo records + 1,321 externally annotated irony records |

## 3. Reproducing the experiments

```bash
cd experiment

python run_experiment.py    # main results        -> experiment_results.json   (~5 min, CPU only)
python run_ablation.py      # channel ablation    -> ablation_results.json
python verify_experiment.py # escape-score / tolerance-gate validation
python run_bert_v3.py       # BERT baseline       -> bert_v3_results.json       (needs torch)
python predict_immune.py    # per-record predictions of the proposed method
python compare_bert.py      # McNemar test, proposed method vs. BERT
```

`immune_pipeline.py` is the shared feature constructor. The main experiment and the ablation both
go through it, and each ablation configuration re-runs the mutual-information step on its own
candidate pool and keeps the top twelve channels — this is what makes the "full immunity" row of
the ablation table numerically identical to the main model.

## 4. Reproducing the figures

```bash
python experiment/make_figures_v4.py     # data figures -> manuscript/figures/*.pdf (vector) + figures/*.png (300 dpi)
```

Figure scripts read the JSON result files directly, so re-running the experiments updates every
plot. `paper_figure_kit.py` holds the shared drawing primitives (figure class, tensor glyphs,
layout audit helpers).

## 5. Data and claim boundaries

- The Weibo-derived special subclasses are **rule-constructed proxies**, not independently
  human-annotated data, and the external irony set comes from a **single annotated source**;
  the two are not independent double annotation.
- The escape score is an **interpretability quantity**, not a standalone detector, and its effect
  size is small (Cohen's *d* = 0.252).
- Zero-shot disguised positives fail: on surface-negative posts whose true label is positive the
  method reaches only 5.30%. This limitation is reported as such in the paper.
- Every number in the paper is produced by the scripts in this repository; none is fabricated.

## 6. License

MIT — see [LICENSE](LICENSE).
