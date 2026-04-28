# EC Class Classification Report

**Date:** 2026-04-28  
**Run ID:** `ec_classifier_20260428_165633`

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
| EC label depth | 3 |
| EC classes | EC 1 (Oxidoreductases), EC 1.1 (?), EC 1.1.1 (?), EC 1.1.2 (?), EC 1.1.3 (?), EC 1.1.4 (?), EC 1.1.5 (?), EC 1.1.7 (?), EC 1.1.98 (?), EC 1.1.99 (?), EC 1.10.3 (?), EC 1.10.5 (?), EC 1.10.99 (?), EC 1.11 (?), EC 1.11.1 (?), EC 1.11.2 (?), EC 1.12.2 (?), EC 1.12.98 (?), EC 1.12.99 (?), EC 1.13 (?), EC 1.13.11 (?), EC 1.13.12 (?), EC 1.13.99 (?), EC 1.14 (?), EC 1.14.11 (?), EC 1.14.12 (?), EC 1.14.13 (?), EC 1.14.14 (?), EC 1.14.15 (?), EC 1.14.16 (?), EC 1.14.17 (?), EC 1.14.18 (?), EC 1.14.19 (?), EC 1.14.20 (?), EC 1.14.21 (?), EC 1.14.99 (?), EC 1.16.1 (?), EC 1.17.1 (?), EC 1.17.2 (?), EC 1.17.3 (?), EC 1.17.4 (?), EC 1.17.5 (?), EC 1.17.7 (?), EC 1.17.8 (?), EC 1.17.9 (?), EC 1.17.99 (?), EC 1.18 (?), EC 1.18.1 (?), EC 1.2 (?), EC 1.2.1 (?), EC 1.2.2 (?), EC 1.2.3 (?), EC 1.2.4 (?), EC 1.2.5 (?), EC 1.2.7 (?), EC 1.2.98 (?), EC 1.2.99 (?), EC 1.20.1 (?), EC 1.20.4 (?), EC 1.21 (?), EC 1.21.1 (?), EC 1.21.3 (?), EC 1.21.4 (?), EC 1.21.98 (?), EC 1.23.1 (?), EC 1.23.5 (?), EC 1.3 (?), EC 1.3.1 (?), EC 1.3.3 (?), EC 1.3.4 (?), EC 1.3.5 (?), EC 1.3.7 (?), EC 1.3.8 (?), EC 1.3.98 (?), EC 1.3.99 (?), EC 1.4 (?), EC 1.4.1 (?), EC 1.4.3 (?), EC 1.4.4 (?), EC 1.4.99 (?), EC 1.5 (?), EC 1.5.1 (?), EC 1.5.3 (?), EC 1.5.4 (?), EC 1.5.98 (?), EC 1.5.99 (?), EC 1.6.2 (?), EC 1.6.3 (?), EC 1.6.5 (?), EC 1.6.6 (?), EC 1.6.99 (?), EC 1.7 (?), EC 1.7.1 (?), EC 1.7.2 (?), EC 1.7.3 (?), EC 1.8.1 (?), EC 1.8.3 (?), EC 1.8.4 (?), EC 1.8.5 (?), EC 1.97.1 (?), EC 2 (Transferases), EC 2.1.1 (?), EC 2.1.2 (?), EC 2.1.3 (?), EC 2.1.4 (?), EC 2.1.5 (?), EC 2.10.1 (?), EC 2.2.1 (?), EC 2.3 (?), EC 2.3.1 (?), EC 2.3.2 (?), EC 2.3.3 (?), EC 2.4 (?), EC 2.4.1 (?), EC 2.4.2 (?), EC 2.4.3 (?), EC 2.4.99 (?), EC 2.5.1 (?), EC 2.6.1 (?), EC 2.6.3 (?), EC 2.6.99 (?), EC 2.7.1 (?), EC 2.7.11 (?), EC 2.7.2 (?), EC 2.7.3 (?), EC 2.7.4 (?), EC 2.7.6 (?), EC 2.7.7 (?), EC 2.7.8 (?), EC 2.7.9 (?), EC 2.8.1 (?), EC 2.8.2 (?), EC 2.8.3 (?), EC 2.8.4 (?), EC 2.8.5 (?), EC 2.8.7 (?), EC 3 (Hydrolases), EC 3.1 (?), EC 3.1.1 (?), EC 3.1.2 (?), EC 3.1.27 (?), EC 3.1.3 (?), EC 3.1.4 (?), EC 3.1.5 (?), EC 3.1.6 (?), EC 3.1.7 (?), EC 3.1.8 (?), EC 3.10.1 (?), EC 3.11.1 (?), EC 3.13.1 (?), EC 3.13.2 (?), EC 3.2 (?), EC 3.2.1 (?), EC 3.2.2 (?), EC 3.3.1 (?), EC 3.3.2 (?), EC 3.4 (?), EC 3.4.11 (?), EC 3.4.13 (?), EC 3.4.14 (?), EC 3.4.15 (?), EC 3.4.16 (?), EC 3.4.17 (?), EC 3.4.19 (?), EC 3.4.21 (?), EC 3.4.22 (?), EC 3.4.24 (?), EC 3.5 (?), EC 3.5.1 (?), EC 3.5.2 (?), EC 3.5.3 (?), EC 3.5.4 (?), EC 3.5.5 (?), EC 3.5.99 (?), EC 3.6.1 (?), EC 3.6.2 (?), EC 3.6.3 (?), EC 3.6.4 (?), EC 3.7.1 (?), EC 3.8.1 (?), EC 3.9.1 (?), EC 4.1.1 (?), EC 4.1.2 (?), EC 4.1.3 (?), EC 4.1.99 (?), EC 4.2.1 (?), EC 4.2.2 (?), EC 4.2.3 (?), EC 4.2.99 (?), EC 4.3.1 (?), EC 4.3.2 (?), EC 4.3.3 (?), EC 4.3.99 (?), EC 4.4.1 (?), EC 4.5.1 (?), EC 4.6.1 (?), EC 4.7.1 (?), EC 4.8.1 (?), EC 4.98.1 (?), EC 4.99.1 (?), EC 5 (Isomerases), EC 5.1.1 (?), EC 5.1.2 (?), EC 5.1.3 (?), EC 5.2.1 (?), EC 5.3 (?), EC 5.3.1 (?), EC 5.3.2 (?), EC 5.3.3 (?), EC 5.3.99 (?), EC 5.4.1 (?), EC 5.4.2 (?), EC 5.4.3 (?), EC 5.4.4 (?), EC 5.4.99 (?), EC 5.5.1 (?), EC 5.6.1 (?), EC 5.99.1 (?), EC 6 (Ligases), EC 6.1 (?), EC 6.1.1 (?), EC 6.1.2 (?), EC 6.1.3 (?), EC 6.2.1 (?), EC 6.3 (?), EC 6.3.1 (?), EC 6.3.2 (?), EC 6.3.3 (?), EC 6.3.4 (?), EC 6.3.5 (?), EC 6.4.1 (?), EC 6.5.1 (?), EC 6.6.1 (?), EC 6.7.1 (?), EC 7.2.1 (?), EC 7.2.4 (?), EC 7.6.2 (?) |
| Samples (requested) | 50,000 |
| Samples (actual) | 49,884 |
| Train / test split | 80/20 stratified |
| Cross-validation folds | 10 |
| Random seed | 42 |
| Total experiment time | 15m 19s |

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
| Embedding extraction (pretrained) | 15.4 s (49884 sequences) |
| Embedding extraction (random-init) | 14.4 s (49884 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,884** labelled SMARTS 
across **237** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.2493 ± 0.0079 | 0.1481 ± 0.0106 | 0.2709 ± 0.0075 | 34.5 | 23.3 |
| SentencePieceTokenizer | 0.3169 ± 0.0064 | 0.1875 ± 0.0105 | 0.3348 ± 0.0055 | 26.9 | 27.3 |
| Pretrained+logreg | 0.5588 ± 0.0059 | 0.3856 ± 0.0190 | 0.5730 ± 0.0059 | 74.9 | 134.5 |
| Random+logreg | 0.2421 ± 0.0068 | 0.1375 ± 0.0072 | 0.2649 ± 0.0060 | 135.2 | 196.1 |
| Pretrained+mlp | **0.7010 ± 0.0058** | **0.4425 ± 0.0247** | **0.6936 ± 0.0053** | 80.2 | 20.2 |
| Random+mlp | 0.4729 ± 0.0088 | 0.1897 ± 0.0163 | 0.4538 ± 0.0112 | 82.0 | 24.6 |

![EC classification comparison](results/ec_classifier_20260428_165633_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 ★ | Oxidoreductases | 14 | 0.042 | 0.143 | 0.065 | 0.038 | 0.143 | 0.061 | 0.113 | 0.429 | 0.179 | 0.047 | 0.214 | 0.077 | 0.444 | 0.286 | 0.348 | 0.000 | 0.000 | 0.000 |
| EC 2 | Transferases | 478 | 0.480 | 0.025 | 0.048 | 0.500 | 0.034 | 0.063 | 0.746 | 0.276 | 0.403 | 0.560 | 0.059 | 0.106 | 0.686 | 0.682 | 0.684 | 0.451 | 0.412 | 0.431 |
| EC 3 ★ | Hydrolases | 2 | 0.067 | 1.000 | 0.125 | 0.118 | 1.000 | 0.210 | 0.500 | 1.000 | 0.667 | 0.100 | 0.500 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| EC 5 ★ | Isomerases | 4 | 0.042 | 0.250 | 0.071 | 1.000 | 0.250 | 0.400 | 0.400 | 0.500 | 0.444 | 0.200 | 0.250 | 0.222 | 0.500 | 0.250 | 0.333 | 0.000 | 0.000 | 0.000 |
| EC 6 ★ | Ligases | 4 | 0.075 | 0.750 | 0.136 | 0.143 | 0.750 | 0.240 | 0.250 | 0.750 | 0.375 | 0.083 | 0.500 | 0.143 | 0.333 | 0.750 | 0.462 | 0.200 | 0.250 | 0.222 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.7010 ± 0.0058, F1 macro 0.4425 ± 0.0247).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.5588 vs random-init 0.2421 (Δ acc = +0.3167; Δ F1 macro = +0.2481). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.7010 vs random-init 0.4729 (Δ acc = +0.2281; Δ F1 macro = +0.2528). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.7010) vs best TF-IDF baseline (SentencePieceTokenizer, 0.3169) (Δ acc = +0.3841).
