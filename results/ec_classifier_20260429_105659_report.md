# EC Class Classification Report

**Date:** 2026-04-29  
**Run ID:** `ec_classifier_20260429_105659`

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
| EC label depth | 1 |
| EC classes | EC 1 (Oxidoreductases), EC 2 (Transferases), EC 3 (Hydrolases), EC 4 (Lyases), EC 5 (Isomerases), EC 6 (Ligases), EC 7 (Translocases) |
| Samples (requested) | 50,000 |
| Samples (actual) | 49,996 |
| Train / test split | 80/20 stratified |
| Cross-validation folds | 10 |
| Random seed | 42 |
| Total experiment time | 50m 20s |

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
| Embedding extraction (pretrained) | 17.8 s (49996 sequences) |
| Embedding extraction (random-init) | 17.3 s (49996 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.5768 ± 0.0069 | 0.4201 ± 0.0111 | 0.6083 ± 0.0064 | 10.4 | 70.2 |
| SentencePieceTokenizer | 0.6232 ± 0.0102 | 0.4699 ± 0.0264 | 0.6474 ± 0.0088 | 11.8 | 10.4 |
| Pretrained+logreg | 0.6796 ± 0.0049 | 0.5294 ± 0.0316 | 0.6990 ± 0.0045 | 16.5 | 477.9 |
| Random+logreg | 0.4883 ± 0.0052 | 0.3479 ± 0.0123 | 0.5267 ± 0.0049 | 32.2 | 1339.5 |
| Pretrained+mlp | **0.8380 ± 0.0040** | **0.6836 ± 0.0527** | **0.8368 ± 0.0046** | 76.9 | 755.8 |
| Random+mlp | 0.6972 ± 0.0041 | 0.4639 ± 0.0111 | 0.6882 ± 0.0049 | 55.4 | 120.4 |

![EC classification comparison](results/ec_classifier_20260429_105659_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 4,314 | 0.878 | 0.635 | 0.737 | 0.877 | 0.673 | 0.762 | 0.906 | 0.732 | 0.810 | 0.816 | 0.561 | 0.665 | 0.915 | 0.919 | 0.917 | 0.807 | 0.852 | 0.829 |
| EC 2 | Transferases | 1,578 | 0.527 | 0.538 | 0.532 | 0.546 | 0.561 | 0.553 | 0.577 | 0.631 | 0.603 | 0.375 | 0.378 | 0.377 | 0.761 | 0.793 | 0.776 | 0.561 | 0.500 | 0.529 |
| EC 3 | Hydrolases | 854 | 0.388 | 0.506 | 0.439 | 0.408 | 0.557 | 0.471 | 0.459 | 0.640 | 0.534 | 0.282 | 0.389 | 0.327 | 0.707 | 0.672 | 0.689 | 0.551 | 0.373 | 0.445 |
| EC 4 | Lyases | 224 | 0.188 | 0.763 | 0.301 | 0.247 | 0.830 | 0.380 | 0.315 | 0.839 | 0.459 | 0.140 | 0.692 | 0.233 | 0.770 | 0.732 | 0.751 | 0.449 | 0.272 | 0.339 |
| EC 5 | Isomerases | 384 | 0.227 | 0.716 | 0.345 | 0.277 | 0.742 | 0.403 | 0.310 | 0.768 | 0.442 | 0.171 | 0.620 | 0.268 | 0.753 | 0.651 | 0.698 | 0.594 | 0.404 | 0.481 |
| EC 6 ★ | Ligases | 3 | 0.028 | 0.333 | 0.051 | 0.105 | 0.667 | 0.182 | 0.077 | 0.333 | 0.125 | 0.050 | 0.333 | 0.087 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.8380 ± 0.0040, F1 macro 0.6836 ± 0.0527).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.6796 vs random-init 0.4883 (Δ acc = +0.1913; Δ F1 macro = +0.1815). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.8380 vs random-init 0.6972 (Δ acc = +0.1408; Δ F1 macro = +0.2197). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.8380) vs best TF-IDF baseline (SentencePieceTokenizer, 0.6232) (Δ acc = +0.2148).
