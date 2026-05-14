# EC Class Classification Report

**Date:** 2026-05-14  
**Run ID:** `ec_classifier_20260514_105358`

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
| Total experiment time | 20m 50s |

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
| Embedding extraction (pretrained) | 33.6 s (49996 sequences) |
| Embedding extraction (random-init) | 33.2 s (49996 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.5768 ± 0.0069 | 0.4201 ± 0.0111 | 0.6083 ± 0.0064 | 24.1 | 52.6 |
| SentencePieceTokenizer | 0.6232 ± 0.0102 | 0.4699 ± 0.0264 | 0.6474 ± 0.0088 | 29.6 | 10.4 |
| Pretrained+logreg | 0.7923 ± 0.0066 | 0.6610 ± 0.0470 | 0.8013 ± 0.0062 | 79.6 | 245.8 |
| Random+logreg | 0.5266 ± 0.0076 | 0.3865 ± 0.0258 | 0.5601 ± 0.0069 | 69.0 | 455.8 |
| Pretrained+mlp | **0.8911 ± 0.0059** | **0.7714 ± 0.0612** | **0.8903 ± 0.0061** | 87.9 | 10.8 |
| Random+mlp | 0.7053 ± 0.0044 | 0.4872 ± 0.0467 | 0.6951 ± 0.0059 | 60.7 | 29.7 |

![EC classification comparison](results/ec_classifier_20260514_105358_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 4,314 | 0.878 | 0.635 | 0.737 | 0.877 | 0.673 | 0.762 | 0.937 | 0.822 | 0.876 | 0.826 | 0.601 | 0.696 | 0.940 | 0.937 | 0.938 | 0.788 | 0.899 | 0.840 |
| EC 2 | Transferases | 1,578 | 0.527 | 0.538 | 0.532 | 0.546 | 0.561 | 0.553 | 0.695 | 0.755 | 0.724 | 0.431 | 0.442 | 0.436 | 0.845 | 0.830 | 0.837 | 0.613 | 0.489 | 0.544 |
| EC 3 | Hydrolases | 854 | 0.388 | 0.506 | 0.439 | 0.408 | 0.557 | 0.471 | 0.578 | 0.786 | 0.666 | 0.297 | 0.424 | 0.349 | 0.799 | 0.814 | 0.806 | 0.540 | 0.411 | 0.467 |
| EC 4 | Lyases | 224 | 0.188 | 0.763 | 0.301 | 0.247 | 0.830 | 0.380 | 0.503 | 0.853 | 0.632 | 0.170 | 0.683 | 0.273 | 0.879 | 0.777 | 0.825 | 0.469 | 0.344 | 0.397 |
| EC 5 | Isomerases | 384 | 0.227 | 0.716 | 0.345 | 0.277 | 0.742 | 0.403 | 0.448 | 0.818 | 0.579 | 0.196 | 0.635 | 0.299 | 0.811 | 0.740 | 0.774 | 0.576 | 0.396 | 0.469 |
| EC 6 ★ | Ligases | 3 | 0.028 | 0.333 | 0.051 | 0.105 | 0.667 | 0.182 | 0.250 | 0.333 | 0.286 | 0.071 | 0.333 | 0.118 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.8911 ± 0.0059, F1 macro 0.7714 ± 0.0612).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.7923 vs random-init 0.5266 (Δ acc = +0.2657; Δ F1 macro = +0.2745). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.8911 vs random-init 0.7053 (Δ acc = +0.1858; Δ F1 macro = +0.2842). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.8911) vs best TF-IDF baseline (SentencePieceTokenizer, 0.6232) (Δ acc = +0.2679).
