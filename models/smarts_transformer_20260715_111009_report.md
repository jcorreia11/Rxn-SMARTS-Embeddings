# MLM Pre-training Report

**Date:** 2026-07-15  
**Run ID:** `smarts_transformer_20260715_111009`

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
| Hidden dimension (`d_model`) | 256 |
| Attention heads (`nhead`) | 8 |
| Encoder layers | 6 |
| Feed-forward dimension | 1024 |
| Dropout | 0.1 |
| Max sequence length | 512 |
| Vocabulary size | 4,466 |
| Total parameters | 7,226,994 |
| Masking probability | 25% |

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| LR schedule | Linear warmup for 2000 steps, then cosine decay to 0 |
| Batch size | 64 |
| Epochs | 100 |
| Gradient clipping | Max norm 1.0 |
| Validation split | 10% held out |
| DataLoader workers | 4 |
| Training time | 6h 17m 5s |

---

## Results

| Metric | Value |
|--------|-------|
| Initial train loss | 3.0741 |
| Final train loss | 0.1602 |
| Best train loss | 0.1599 (epoch 93) |
| Final val loss | 0.1266 |
| Best val loss | 0.1266 (epoch 100) |
| Final top-1 accuracy | 96.58% |
| Best top-1 accuracy | 96.58% (epoch 100) |
| Final top-5 accuracy | 99.55% |
| Best top-5 accuracy | 99.55% (epoch 93) |

![Training loss curve](models/smarts_transformer_20260715_111009_training_loss.pdf)

*Figure: MLM training loss over epochs.*
