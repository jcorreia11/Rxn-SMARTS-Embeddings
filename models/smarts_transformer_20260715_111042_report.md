# MLM Pre-training Report

**Date:** 2026-07-16  
**Run ID:** `smarts_transformer_20260715_111042`

---

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU Memory | 39.5 GB |
| GPU Count | 1 |
| CUDA Version | 12.4 |
| Driver Version | 580.167.08 |
| CPU | AMD EPYC 7742 64-Core Processor |
| CPU Cores | 128 |
| RAM | 503.7 GB |
| OS | Linux-4.18.0-348.el8.0.2.x86_64-x86_64-with-glibc2.28 |

---

## Software

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

## Model Architecture

Transformer encoder pre-trained with a masked language modelling (MLM) objective (BERT-style, 80/10/10 masking strategy).

| Hyper-parameter | Value |
|-----------------|-------|
| Hidden dimension (`d_model`) | 512 |
| Attention heads (`nhead`) | 16 |
| Encoder layers | 8 |
| Feed-forward dimension | 2048 |
| Dropout | 0.1 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Total parameters | 30,322,546 |
| Masking probability | 25% |

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| LR schedule | Linear warmup for 2000 steps, then cosine decay to 0 |
| Batch size | 32 |
| Epochs | 100 |
| Gradient clipping | Max norm 1.0 |
| Validation split | 10% held out |
| DataLoader workers | 4 |
| Training time | 19h 47m 19s |

---

## Results

| Metric | Value |
|--------|-------|
| Initial train loss | 1.8011 |
| Final train loss | 0.0437 |
| Best train loss | 0.0436 (epoch 99) |
| Final val loss | 0.0519 |
| Best val loss | 0.0512 (epoch 91) |
| Final top-1 accuracy | 98.69% |
| Best top-1 accuracy | 98.72% (epoch 98) |
| Final top-5 accuracy | 99.87% |
| Best top-5 accuracy | 99.88% (epoch 91) |

![Training loss curve](models/smarts_transformer_20260715_111042_training_loss.pdf)

*Figure: MLM training loss over epochs.*
