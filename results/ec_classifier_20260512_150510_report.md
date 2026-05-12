# EC Class Classification Report

**Date:** 2026-05-12  
**Run ID:** `ec_classifier_20260512_150510`

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
| Total experiment time | 25m 8s |

### Pretrained Transformer (embedding experiments)

| Hyper-parameter | Value |
|-----------------|-------|
| Hidden dimension (`d_model`) | 256 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Feed-forward dimension | 1024 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Pooling strategy | mean |
| Weights file | `smarts_transformer_20260428_160310.pt` |
| Embedding extraction (pretrained) | 19.9 s (49962 sequences) |
| Embedding extraction (random-init) | 16.1 s (49962 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,962** labelled SMARTS 
across **72** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.3979 ± 0.0059 | 0.2373 ± 0.0167 | 0.4216 ± 0.0070 | 15.7 | 29.2 |
| SentencePieceTokenizer | 0.4617 ± 0.0083 | 0.2918 ± 0.0233 | 0.4867 ± 0.0087 | 16.8 | 37.0 |
| Pretrained+logreg | 0.6182 ± 0.0090 | 0.4371 ± 0.0235 | 0.6353 ± 0.0079 | 87.4 | 313.4 |
| Random+logreg | 0.3374 ± 0.0052 | 0.1823 ± 0.0136 | 0.3627 ± 0.0052 | 105.4 | 627.9 |
| Pretrained+mlp | **0.7542 ± 0.0063** | **0.5208 ± 0.0286** | **0.7515 ± 0.0058** | 100.2 | 19.3 |
| Random+mlp | 0.5410 ± 0.0063 | 0.2567 ± 0.0215 | 0.5302 ± 0.0071 | 69.6 | 42.8 |

![EC classification comparison](results/ec_classifier_20260512_150510_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 559 | 0.526 | 0.181 | 0.269 | 0.581 | 0.276 | 0.374 | 0.715 | 0.449 | 0.552 | 0.455 | 0.163 | 0.240 | 0.693 | 0.760 | 0.725 | 0.459 | 0.504 | 0.480 |
| EC 2 ★ | Transferases | 35 | 0.039 | 0.229 | 0.066 | 0.076 | 0.371 | 0.127 | 0.135 | 0.429 | 0.205 | 0.041 | 0.200 | 0.068 | 0.486 | 0.486 | 0.486 | 0.312 | 0.143 | 0.196 |
| EC 3 | Hydrolases | 64 | 0.081 | 0.188 | 0.113 | 0.140 | 0.281 | 0.186 | 0.207 | 0.516 | 0.296 | 0.087 | 0.203 | 0.122 | 0.473 | 0.406 | 0.437 | 0.357 | 0.234 | 0.283 |
| EC 5 | Isomerases | 206 | 0.392 | 0.359 | 0.375 | 0.453 | 0.519 | 0.484 | 0.554 | 0.568 | 0.561 | 0.401 | 0.393 | 0.397 | 0.757 | 0.665 | 0.708 | 0.533 | 0.427 | 0.474 |
| EC 6 | Ligases | 980 | 0.680 | 0.089 | 0.157 | 0.732 | 0.201 | 0.316 | 0.831 | 0.412 | 0.551 | 0.520 | 0.065 | 0.116 | 0.712 | 0.761 | 0.736 | 0.451 | 0.540 | 0.492 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.7542 ± 0.0063, F1 macro 0.5208 ± 0.0286).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.6182 vs random-init 0.3374 (Δ acc = +0.2808; Δ F1 macro = +0.2548). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.7542 vs random-init 0.5410 (Δ acc = +0.2132; Δ F1 macro = +0.2641). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.7542) vs best TF-IDF baseline (SentencePieceTokenizer, 0.4617) (Δ acc = +0.2925).
