# MLM Pre-training Report

**Date:** 2026-04-10  
**Run ID:** `smarts_transformer_20260409_143643`

---

## Hardware

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
| Max sequence length | 256 |
| Vocabulary size | 4,466 |
| Total parameters | 7,161,458 |
| Masking probability | 15% |

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
| Training time | 11h 22m 47s |

---

## Results

| Metric | Value |
|--------|-------|
| Initial train loss | 2.7582 |
| Final train loss | 0.0463 |
| Best train loss | 0.0459 (epoch 99) |
| Final val loss | 0.0328 |
| Best val loss | 0.0312 (epoch 94) |
| Final top-1 accuracy | 99.08% |
| Best top-1 accuracy | 99.11% (epoch 99) |
| Final top-5 accuracy | 99.89% |
| Best top-5 accuracy | 99.90% (epoch 80) |

![Training loss curve](models/smarts_transformer_20260409_143643_training_loss.pdf)

*Figure: MLM training loss over epochs.*
