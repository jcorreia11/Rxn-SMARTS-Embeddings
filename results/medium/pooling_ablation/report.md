# Pooling Ablation Report: CLS vs Mean

**Date:** 2026-05-13  
**Run ID:** `results`

This ablation compares **CLS** and **mean** pooling strategies for collapsing the transformer encoder output into a fixed-size reaction embedding. Both strategies use the same pretrained weights; the only difference is how the per-token representations are aggregated. Downstream performance is measured on EC class prediction.

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
| RDKit | 2025.9.6 |

---

## 2. Experimental Setup

### Dataset

| Parameter | Value |
|-----------|-------|
| Source | RetroRules v3.0 |
| EC label depth | 1 |
| EC classes | EC 1 (Oxidoreductases), EC 2 (Transferases), EC 3 (Hydrolases), EC 4 (Lyases), EC 5 (Isomerases), EC 6 (Ligases), EC 7 (Translocases) |
| Samples (requested) | 20,000 |
| Samples (actual) | 19,996 |
| Train / test split | 80/20 stratified |
| CV folds | 10 |
| Random seed | 42 |

### Pretrained Transformer

| Hyper-parameter | Value |
|-----------------|-------|
| Hidden dimension (`d_model`) | 256 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Feed-forward dimension | 1024 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Weights file | `smarts_transformer_20260428_160310.pt` |
| Embedding extraction (both poolings) | 17.1 s |
| Total experiment time | 1h 18m 24s |

Both pooling strategies are extracted from the **same model in a single forward pass**, so the only variable between conditions is how the encoder's output sequence is collapsed to a fixed-size vector:

- **CLS pooling** — uses the representation of the `[BOS]` token.
- **Mean pooling** — averages all non-padding token representations.

---

## 3. Results

Evaluated on **19,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric.

| Method | Pooling | Head | Accuracy | F1 macro | F1 weighted | CV time (s) |
|--------|---------|------|----------|----------|-------------|-------------|
| CLS+logreg | CLS | logreg | 0.6035 ± 0.0097 | 0.5091 ± 0.0586 | 0.6281 ± 0.0095 | 1785.91 |
| CLS+mlp | CLS | mlp | 0.7416 ± 0.0060 | 0.5820 ± 0.0437 | 0.7378 ± 0.0057 | 369.43 |
| Mean+logreg | MEAN | logreg | 0.6746 ± 0.0152 | 0.5693 ± 0.0520 | 0.6927 ± 0.0144 | 1872.02 |
| Mean+mlp | MEAN | mlp | **0.7959 ± 0.0123** | **0.6599 ± 0.0652** | **0.7932 ± 0.0126** | 188.24 |

![Pooling ablation](results/medium/pooling_ablation/report.pdf)

*Figure: Accuracy and F1 macro for CLS and mean pooling across both classifier heads. Error bars show ± 1 standard deviation over CV folds.*

---

## 4. Per-class Breakdown (F1)

| EC Class | Name | Support | F1 (CLS+logreg) | F1 (CLS+mlp) | F1 (Mean+logreg) | F1 (Mean+mlp) |
|----------|------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 1,057 | 0.616 | 0.762 | 0.688 | 0.810 |
| EC 2 | Transferases | 1,726 | 0.760 | 0.856 | 0.812 | 0.883 |
| EC 3 | Hydrolases | 631 | 0.526 | 0.629 | 0.599 | 0.667 |
| EC 4 | Lyases | 342 | 0.466 | 0.577 | 0.517 | 0.622 |
| EC 5 | Isomerases | 89 | 0.360 | 0.470 | 0.456 | 0.637 |
| EC 6 | Ligases | 154 | 0.352 | 0.498 | 0.441 | 0.528 |
| EC 7 | Translocases | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## 5. Summary

- **LOGREG head**: Mean pooling wins. CLS accuracy 0.6035 vs mean 0.6746 (Δ acc = -0.0711; Δ F1 macro = -0.0602).
- **MLP head**: Mean pooling wins. CLS accuracy 0.7416 vs mean 0.7959 (Δ acc = -0.0543; Δ F1 macro = -0.0779).

**Overall best**: Mean+mlp (accuracy 0.7959 ± 0.0123, F1 macro 0.6599 ± 0.0652).
