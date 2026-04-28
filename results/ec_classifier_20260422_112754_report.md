# EC Class Classification Report

**Date:** 2026-04-22  
**Run ID:** `ec_classifier_20260422_112754`

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
| Total experiment time | 13m 21s |

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
| Embedding extraction (pretrained) | 15.3 s (49996 sequences) |
| Embedding extraction (random-init) | 14.3 s (49996 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.5768 ± 0.0069 | 0.4201 ± 0.0111 | 0.6083 ± 0.0064 | 9.9 | 54.0 |
| SentencePieceTokenizer | 0.6232 ± 0.0102 | 0.4699 ± 0.0264 | 0.6474 ± 0.0088 | 11.4 | 9.9 |
| Pretrained+logreg | 0.6757 ± 0.0076 | 0.5249 ± 0.0238 | 0.6942 ± 0.0070 | 24.3 | 150.9 |
| Random+logreg | 0.4862 ± 0.0068 | 0.3445 ± 0.0194 | 0.5250 ± 0.0063 | 26.4 | 371.4 |
| Pretrained+mlp | **0.8379 ± 0.0050** | **0.6657 ± 0.0426** | **0.8367 ± 0.0054** | 26.7 | 30.6 |
| Random+mlp | 0.6918 ± 0.0078 | 0.4639 ± 0.0457 | 0.6785 ± 0.0083 | 20.2 | 29.2 |

![EC classification comparison](results/ec_classifier_20260422_112754_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 2,643 | 0.672 | 0.508 | 0.578 | 0.731 | 0.580 | 0.646 | 0.791 | 0.597 | 0.680 | 0.637 | 0.424 | 0.509 | 0.814 | 0.888 | 0.850 | 0.689 | 0.751 | 0.718 |
| EC 2 | Transferases | 4,314 | 0.878 | 0.635 | 0.737 | 0.877 | 0.673 | 0.762 | 0.902 | 0.725 | 0.804 | 0.815 | 0.564 | 0.667 | 0.923 | 0.901 | 0.912 | 0.813 | 0.838 | 0.825 |
| EC 3 | Hydrolases | 1,578 | 0.527 | 0.538 | 0.532 | 0.546 | 0.561 | 0.553 | 0.565 | 0.620 | 0.591 | 0.373 | 0.374 | 0.374 | 0.798 | 0.767 | 0.783 | 0.553 | 0.523 | 0.537 |
| EC 4 | Lyases | 854 | 0.388 | 0.506 | 0.439 | 0.408 | 0.557 | 0.471 | 0.461 | 0.642 | 0.536 | 0.262 | 0.350 | 0.299 | 0.669 | 0.673 | 0.671 | 0.478 | 0.416 | 0.445 |
| EC 5 | Isomerases | 224 | 0.188 | 0.763 | 0.301 | 0.247 | 0.830 | 0.380 | 0.325 | 0.871 | 0.473 | 0.132 | 0.656 | 0.220 | 0.783 | 0.692 | 0.735 | 0.459 | 0.326 | 0.381 |
| EC 6 | Ligases | 384 | 0.227 | 0.716 | 0.345 | 0.277 | 0.742 | 0.403 | 0.314 | 0.812 | 0.453 | 0.173 | 0.633 | 0.271 | 0.789 | 0.672 | 0.726 | 0.577 | 0.419 | 0.486 |
| EC 7 ★ | Translocases | 3 | 0.028 | 0.333 | 0.051 | 0.105 | 0.667 | 0.182 | 0.050 | 0.333 | 0.087 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.8379 ± 0.0050, F1 macro 0.6657 ± 0.0426).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.6757 vs random-init 0.4862 (Δ acc = +0.1895; Δ F1 macro = +0.1804). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.8379 vs random-init 0.6918 (Δ acc = +0.1461; Δ F1 macro = +0.2018). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.8379) vs best TF-IDF baseline (SentencePieceTokenizer, 0.6232) (Δ acc = +0.2147).
