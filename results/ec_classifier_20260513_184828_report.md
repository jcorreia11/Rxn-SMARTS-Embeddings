# EC Class Classification Report

**Date:** 2026-05-13  
**Run ID:** `ec_classifier_20260513_184828`

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
| Total experiment time | 8m 17s |

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
| Embedding extraction (pretrained) | 17.9 s (49996 sequences) |
| Embedding extraction (random-init) | 17.2 s (49996 sequences) |

The random-embedding baseline uses the **same architecture with randomly initialised weights** (no pretraining), isolating the contribution of the MLM pretraining objective.

---

## 3. Results

Evaluated on **49,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric. Scores are mean ± std over folds (CV set); final test accuracy is on the held-out 20 % split.

| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |
|--------|----------|----------|-------------|-------------|--------------|
| SmartsTokenizer | 0.5768 ± 0.0069 | 0.4201 ± 0.0111 | 0.6083 ± 0.0064 | 10.1 | 36.1 |
| SentencePieceTokenizer | 0.6232 ± 0.0102 | 0.4699 ± 0.0264 | 0.6474 ± 0.0088 | 11.5 | 10.0 |
| Pretrained+logreg | 0.6796 ± 0.0049 | 0.5294 ± 0.0316 | 0.6990 ± 0.0045 | 16.9 | 68.8 |
| Random+logreg | 0.4987 ± 0.0066 | 0.3523 ± 0.0167 | 0.5359 ± 0.0061 | 26.6 | 206.5 |
| Pretrained+mlp | **0.8380 ± 0.0040** | **0.6836 ± 0.0527** | **0.8368 ± 0.0046** | 22.8 | 13.8 |
| Random+mlp | 0.6938 ± 0.0048 | 0.4746 ± 0.0437 | 0.6843 ± 0.0048 | 21.2 | 10.8 |

![EC classification comparison](results/ec_classifier_20260513_184828_report.pdf)

*Figure: Accuracy and F1 macro for all methods on EC class prediction. Error bars show ± 1 standard deviation across CV folds.*

---

## 4. Per-class Breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) | P (Pretrained+logreg) | R (Pretrained+logreg) | F1 (Pretrained+logreg) | P (Random+logreg) | R (Random+logreg) | F1 (Random+logreg) | P (Pretrained+mlp) | R (Pretrained+mlp) | F1 (Pretrained+mlp) | P (Random+mlp) | R (Random+mlp) | F1 (Random+mlp) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 4,314 | 0.878 | 0.635 | 0.737 | 0.877 | 0.673 | 0.762 | 0.906 | 0.732 | 0.810 | 0.823 | 0.588 | 0.686 | 0.915 | 0.919 | 0.917 | 0.800 | 0.848 | 0.824 |
| EC 2 | Transferases | 1,578 | 0.527 | 0.538 | 0.532 | 0.546 | 0.561 | 0.553 | 0.577 | 0.631 | 0.603 | 0.382 | 0.387 | 0.384 | 0.761 | 0.793 | 0.776 | 0.560 | 0.527 | 0.543 |
| EC 3 | Hydrolases | 854 | 0.388 | 0.506 | 0.439 | 0.408 | 0.557 | 0.471 | 0.459 | 0.640 | 0.534 | 0.250 | 0.330 | 0.285 | 0.707 | 0.672 | 0.689 | 0.529 | 0.321 | 0.399 |
| EC 4 | Lyases | 224 | 0.188 | 0.763 | 0.301 | 0.247 | 0.830 | 0.380 | 0.315 | 0.839 | 0.459 | 0.149 | 0.710 | 0.247 | 0.770 | 0.732 | 0.751 | 0.395 | 0.286 | 0.332 |
| EC 5 | Isomerases | 384 | 0.227 | 0.716 | 0.345 | 0.277 | 0.742 | 0.403 | 0.310 | 0.768 | 0.442 | 0.172 | 0.591 | 0.267 | 0.753 | 0.651 | 0.698 | 0.611 | 0.380 | 0.469 |
| EC 6 ★ | Ligases | 3 | 0.028 | 0.333 | 0.051 | 0.105 | 0.667 | 0.182 | 0.077 | 0.333 | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

> **★ Rare class** — fewer than 1 % of the per-class sample mean or < 50 examples. Low scores reflect data scarcity, not model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 5. Summary

- **Best method**: Pretrained+mlp (accuracy 0.8380 ± 0.0040, F1 macro 0.6836 ± 0.0527).
- **Pretraining gain (logreg)**: pretrained embeddings achieve accuracy 0.6796 vs random-init 0.4987 (Δ acc = +0.1809; Δ F1 macro = +0.1771). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Pretraining gain (mlp)**: pretrained embeddings achieve accuracy 0.8380 vs random-init 0.6938 (Δ acc = +0.1442; Δ F1 macro = +0.2090). Positive gain confirms that MLM pretraining encodes reaction-type information beyond random projection.
- **Embeddings vs TF-IDF**: best pretrained embedding method (Pretrained+mlp, 0.8380) vs best TF-IDF baseline (SentencePieceTokenizer, 0.6232) (Δ acc = +0.2148).
