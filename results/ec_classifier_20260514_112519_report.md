# EC Class Classification Report

**Date:** 2026-05-14  
**Run ID:** `ec_classifier_20260514_112519`

This report compares reaction SMARTS featurisation strategies on the task of predicting enzyme class (EC number) from reaction templates. It demonstrates whether transformer embeddings learned via masked language modelling (MLM) encode more reaction-type information than TF-IDF baselines and randomly initialised (unlearned) embeddings.

---

## 1. Experimental Environment

### Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU Memory | 39.4 GB |
| GPU Count | 1 |
| CUDA Version | 12.4 |
| Driver Version | 550.90.07 |
| CPU | AMD EPYC 7742 64-Core Processor |
| CPU Cores | 128 |
| RAM | 503.7 GB |
| OS | Linux-4.18.0-348.el8.0.2.x86_64-x86_64-with-glibc2.28 |

### Software

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| NumPy | 2.4.3 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| SentencePiece | 0.2.1 |
| RDKit | 2025.9.6 |

---

## 2. Experimental Setup

### Dataset

| Parameter | Value |
|-----------|-------|
| Source | RetroRules v3.0 |
| EC label depth | 2 |
| EC classes | EC 1 (Oxidoreductases), EC 1.1 (?), EC 1.10 (?), EC 1.11 (?), EC 1.12 (?), EC 1.13 (?), EC 1.14 (?), EC 1.16 (?), EC 1.17 (?), EC 1.18 (?), EC 1.2 (?), EC 1.20 (?), EC 1.21 (?), EC 1.23 (?), EC 1.3 (?), EC 1.4 (?), EC 1.5 (?), EC 1.6 (?), EC 1.7 (?), EC 1.8 (?), EC 1.97 (?), EC 2 (Transferases), EC 2.1 (?), EC 2.10 (?), EC 2.2 (?), EC 2.3 (?), EC 2.4 (?), EC 2.5 (?), EC 2.6 (?), EC 2.7 (?), EC 2.8 (?), EC 3 (Hydrolases), EC 3.1 (?), EC 3.10 (?), EC 3.11 (?), EC 3.13 (?), EC 3.2 (?), EC 3.3 (?), EC 3.4 (?), EC 3.5 (?), EC 3.6 (?), EC 3.7 (?), EC 3.8 (?), EC 3.9 (?), EC 4.1 (?), EC 4.2 (?), EC 4.3 (?), EC 4.4 (?), EC 4.5 (?), EC 4.6 (?), EC 4.7 (?), EC 4.8 (?), EC 4.98 (?), EC 4.99 (?), EC 5 (Isomerases), EC 5.1 (?), EC 5.2 (?), EC 5.3 (?), EC 5.4 (?), EC 5.5 (?), EC 5.6 (?), EC 5.99 (?), EC 6 (Ligases), EC 6.1 (?), EC 6.2 (?), EC 6.3 (?), EC 6.4 (?), EC 6.5 (?), EC 6.6 (?), EC 6.7 (?), EC 7.2 (?), EC 7.6 (?) |
| Samples (requested) | 50,000 |
| Samples (actual) | 49,962 |
| Train / test split | 80/20 stratified |
| Cross-validation folds | 10 |
| Random seed | 42 |
| Total experiment time | 23m 51s |

### Pretrained Transformer (embedding experiments)

| Hyper-parameter | Value |
|-----------------|-------|
| Hidden dimension (`d_model`) | 512 |
| Attention heads | 16 |
| Encoder layers | 8 |
| Feed-forward dimension | 2048 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Pooling strategy | mean |
| Weights file | `smarts_transformer_20260513_093030.pt` |
| Embedding extraction (pretrained) | 33.5 s (49962 sequences) |
| Embedding extraction (random-init) | 36.3 s (49962 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,962** labelled SMARTS 
across **72** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.3979 ± 0.0059 | 0.2373 ± 0.0167 | 0.4216 ± 0.0070 | 28.2 | 37.7 |
| SentencePieceTokenizer | 0.4617 ± 0.0083 | 0.2918 ± 0.0233 | 0.4867 ± 0.0087 | 35.8 | 45.8 |
| Pretrained+logreg | 0.7647 ± 0.0050 | 0.6167 ± 0.0183 | 0.7702 ± 0.0051 | 104.4 | 253.8 |
| Random+logreg | 0.3956 ± 0.0054 | 0.2279 ± 0.0114 | 0.4193 ± 0.0050 | 107.6 | 492.8 |
| Pretrained+mlp | **0.8261 ± 0.0068** | **0.6180 ± 0.0343** | **0.8238 ± 0.0068** | 91.0 | 27.0 |
| Random+mlp | 0.5539 ± 0.0090 | 0.2789 ± 0.0174 | 0.5450 ± 0.0099 | 71.6 | 32.3 |

![EC classification comparison](results/ec_classifier_20260514_112519_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 559 | 0.526 | 0.181 | 0.269 | 0.581 | 0.276 | 0.374 | 0.808 | 0.707 | 0.754 | 0.540 | 0.216 | 0.309 | 0.878 | 0.801 | 0.838 | 0.454 | 0.499 | 0.475 |
| EC 2 ★ | Transferases | 35 | 0.039 | 0.229 | 0.066 | 0.076 | 0.371 | 0.127 | 0.339 | 0.629 | 0.440 | 0.040 | 0.171 | 0.065 | 0.528 | 0.543 | 0.535 | 0.132 | 0.143 | 0.137 |
| EC 3 | Hydrolases | 64 | 0.081 | 0.188 | 0.113 | 0.140 | 0.281 | 0.186 | 0.325 | 0.625 | 0.428 | 0.098 | 0.250 | 0.141 | 0.760 | 0.594 | 0.667 | 0.219 | 0.250 | 0.234 |
| EC 5 | Isomerases | 206 | 0.392 | 0.359 | 0.375 | 0.453 | 0.519 | 0.484 | 0.662 | 0.714 | 0.687 | 0.409 | 0.413 | 0.411 | 0.865 | 0.714 | 0.782 | 0.579 | 0.461 | 0.513 |
| EC 6 | Ligases | 980 | 0.680 | 0.089 | 0.157 | 0.732 | 0.201 | 0.316 | 0.876 | 0.608 | 0.718 | 0.634 | 0.164 | 0.261 | 0.759 | 0.866 | 0.809 | 0.485 | 0.590 | 0.532 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.8261 ± 0.0068, F1 macro 0.6180 ± 0.0343).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.7647 vs random-init 0.3956 (Δ acc = +0.3691; Δ F1 macro = +0.3888). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.8261 vs random-init 0.5539 (Δ acc = +0.2722; Δ F1 macro = +0.3391). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.8261) vs best TF-IDF baseline (SentencePieceTokenizer, 0.4617) (Δ acc = +0.3644).
