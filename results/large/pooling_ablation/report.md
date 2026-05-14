# Pooling Ablation Report: CLS vs Mean

**Date:** 2026-05-14  
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
| Hidden dimension (`d_model`) | 512 |
| Attention heads | 16 |
| Encoder layers | 8 |
| Feed-forward dimension | 2048 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Weights file | `smarts_transformer_20260513_093030.pt` |
| Embedding extraction (both poolings) | 27.2 s |
| Total experiment time | 53m 18s |

Both pooling strategies are extracted from the **same model in a single forward pass**, so the only variable between conditions is how the encoder's output sequence is collapsed to a fixed-size vector:

- **CLS pooling** — uses the representation of the `[BOS]` token.
- **Mean pooling** — averages all non-padding token representations.

---

## 3. Results

Evaluated on **19,996** labelled SMARTS 
across **7** EC classes with **10**-fold stratified cross-validation. Bold values indicate the best result per metric.

| Method | Pooling | Head | Accuracy | F1 macro | F1 weighted | CV time (s) |
|--------|---------|------|----------|----------|-------------|-------------|
| CLS+logreg | CLS | logreg | 0.7159 ± 0.0070 | 0.6255 ± 0.0480 | 0.7288 ± 0.0064 | 1321.63 |
| CLS+mlp | CLS | mlp | 0.8009 ± 0.0105 | 0.6917 ± 0.0569 | 0.7988 ± 0.0103 | 229.97 |
| Mean+logreg | MEAN | logreg | 0.7818 ± 0.0097 | 0.6806 ± 0.0558 | 0.7905 ± 0.0080 | 1182.97 |
| Mean+mlp | MEAN | mlp | **0.8561 ± 0.0094** | **0.7343 ± 0.0611** | **0.8548 ± 0.0096** | 104.65 |

![Pooling ablation](results/large/pooling_ablation/report.pdf)

*Figure: Accuracy and F1 macro for CLS and mean pooling across both classifier heads. Error bars show ± 1 standard deviation over CV folds.*

---

## 4. Per-class Breakdown (F1)

| EC Class | Name | Support | F1 (CLS+logreg) | F1 (CLS+mlp) | F1 (Mean+logreg) | F1 (Mean+mlp) |
|----------|------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 1,057 | 0.744 | 0.836 | 0.805 | 0.879 |
| EC 2 | Transferases | 1,726 | 0.827 | 0.886 | 0.874 | 0.917 |
| EC 3 | Hydrolases | 631 | 0.619 | 0.714 | 0.712 | 0.786 |
| EC 4 | Lyases | 342 | 0.558 | 0.698 | 0.638 | 0.755 |
| EC 5 | Isomerases | 89 | 0.617 | 0.565 | 0.635 | 0.762 |
| EC 6 | Ligases | 154 | 0.440 | 0.593 | 0.552 | 0.652 |
| EC 7 | Translocases | 1 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## 5. Summary

- **LOGREG head**: Mean pooling wins. CLS accuracy 0.7159 vs mean 0.7818 (Δ acc = -0.0659; Δ F1 macro = -0.0551).
- **MLP head**: Mean pooling wins. CLS accuracy 0.8009 vs mean 0.8561 (Δ acc = -0.0552; Δ F1 macro = -0.0426).

**Overall best**: Mean+mlp (accuracy 0.8561 ± 0.0094, F1 macro 0.7343 ± 0.0611).
