# EC Class Classification Report

**Date:** 2026-04-22  
**Run ID:** `ec_classifier_20260422_154808`

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
| Total experiment time | 25m 46s |

### Pretrained Transformer (embedding experiments)

| Hyper-parameter | Value |
|-----------------|-------|
| Hidden dimension (`d_model`) | 256 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Feed-forward dimension | 1024 |
| Max sequence length | 256 |
| Vocabulary size | 4,466 |
| Pooling strategy | mean |
| Weights file | `smarts_transformer_20260410_111830.pt` |
| Embedding extraction (pretrained) | 15.6 s (49962 sequences) |
| Embedding extraction (random-init) | 14.5 s (49962 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,962** labelled SMARTS 
across **72** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.3979 ± 0.0059 | 0.2373 ± 0.0167 | 0.4216 ± 0.0070 | 15.8 | 25.8 |
| SentencePieceTokenizer | 0.4617 ± 0.0083 | 0.2918 ± 0.0233 | 0.4867 ± 0.0087 | 16.6 | 31.0 |
| Pretrained+logreg | 0.6091 ± 0.0109 | 0.4302 ± 0.0197 | 0.6257 ± 0.0109 | 35.2 | 307.9 |
| Random+logreg | 0.3391 ± 0.0067 | 0.1797 ± 0.0097 | 0.3641 ± 0.0077 | 65.5 | 832.8 |
| Pretrained+mlp | **0.7492 ± 0.0097** | **0.5097 ± 0.0220** | **0.7461 ± 0.0095** | 84.1 | 33.7 |
| Random+mlp | 0.5396 ± 0.0068 | 0.2630 ± 0.0206 | 0.5280 ± 0.0057 | 26.0 | 33.2 |

![EC classification comparison](results/ec_classifier_20260422_154808_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 559 | 0.526 | 0.181 | 0.269 | 0.581 | 0.276 | 0.374 | 0.766 | 0.462 | 0.576 | 0.568 | 0.179 | 0.272 | 0.765 | 0.726 | 0.745 | 0.516 | 0.411 | 0.458 |
| EC 2 ★ | Transferases | 35 | 0.039 | 0.229 | 0.066 | 0.076 | 0.371 | 0.127 | 0.123 | 0.400 | 0.188 | 0.050 | 0.257 | 0.084 | 0.371 | 0.371 | 0.371 | 0.278 | 0.143 | 0.189 |
| EC 3 | Hydrolases | 64 | 0.081 | 0.188 | 0.113 | 0.140 | 0.281 | 0.186 | 0.192 | 0.469 | 0.273 | 0.090 | 0.250 | 0.132 | 0.469 | 0.469 | 0.469 | 0.256 | 0.172 | 0.206 |
| EC 5 | Isomerases | 206 | 0.392 | 0.359 | 0.375 | 0.453 | 0.519 | 0.484 | 0.525 | 0.558 | 0.541 | 0.382 | 0.379 | 0.381 | 0.729 | 0.665 | 0.695 | 0.502 | 0.485 | 0.494 |
| EC 6 | Ligases | 980 | 0.680 | 0.089 | 0.157 | 0.732 | 0.201 | 0.316 | 0.797 | 0.385 | 0.519 | 0.588 | 0.102 | 0.174 | 0.707 | 0.774 | 0.739 | 0.454 | 0.594 | 0.515 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.7492 ± 0.0097, F1 macro 0.5097 ± 0.0220).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.6091 vs random-init 0.3391 (Δ acc = +0.2700; Δ F1 macro = +0.2505). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.7492 vs random-init 0.5396 (Δ acc = +0.2096; Δ F1 macro = +0.2467). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.7492) vs best TF-IDF baseline (SentencePieceTokenizer, 0.4617) (Δ acc = +0.2875).
