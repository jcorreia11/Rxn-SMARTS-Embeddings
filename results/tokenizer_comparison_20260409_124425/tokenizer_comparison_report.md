# Tokenizer Comparison Report

**Date:** 2026-04-09  
**Tokenizers:** SmartsTokenizer vs SentencePieceTokenizer

---

## 0. Experimental Environment

**Hardware**

| Component | Details |
|-----------|---------|
| CPU | AMD EPYC 7742 64-Core Processor |
| CPU Cores | 128 |
| RAM | 251.7 GB |
| OS | Linux-4.18.0-348.el8.0.2.x86_64-x86_64-with-glibc2.28 |

**Software**

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

## 1. Intrinsic Metrics

Properties measured directly from the tokenization of a shared SMARTS sample.
Bold values indicate the better result where applicable.

| Metric | SmartsTokenizer | SentencePieceTokenizer |
|--------|--------|--------|
| Vocabulary size | rule-based | 1,000 |
| Samples evaluated | 10,000 | 10,000 |
| Error rate | **0.00%** | 0.00% |
| Throughput (SMARTS/s) | **9,153** | 2,845 |
| Throughput (chars/s) | **3,905,033** | 1,213,834 |
| Seq length — mean | **118.98** | 189.71 |
| Seq length — median | **100.00** | 160.00 |
| Seq length — std | **80.80** | 126.28 |
| Seq length — min | 3 | 12 |
| Seq length — max | 659 | 1,204 |
| Fertility ratio (tokens/char) | **0.2751** | 0.4476 |
| Unique tokens used | 1,882 | 982 |
| Vocabulary utilisation | N/A | 98.20% |
| Round-trip fidelity | **100.00%** | 100.00% |

![token_length_distribution](results/tokenizer_comparison_20260409_124425/figures/token_length_distribution.png)


![throughput_fertility](results/tokenizer_comparison_20260409_124425/figures/throughput_fertility.png)


---

## 2. Downstream Task — EC Class Classification

Each tokenizer's sequences are featurised with TF-IDF and fed into a logistic regression classifier predicting the top-level EC enzyme class (oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase, translocase). Higher scores indicate the tokenizer encodes more reaction-type information.

Evaluated on **9996** labelled SMARTS, **7** EC classes, **5**-fold stratified cross-validation.

| Metric | SmartsTokenizer | SentencePieceTokenizer |
|--------|--------|--------|
| Accuracy | 0.5400 ± 0.0066 | **0.5858 ± 0.0115** |
| F1 macro | 0.3688 ± 0.0076 | **0.4217 ± 0.0309** |
| F1 weighted | 0.5734 ± 0.0053 | **0.6109 ± 0.0118** |

![ec_classifier](results/tokenizer_comparison_20260409_124425/figures/ec_classifier.png)


### Per-class breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 529 | 0.614 | 0.442 | 0.514 | 0.654 | 0.550 | 0.598 |
| EC 2 | Transferases | 863 | 0.843 | 0.629 | 0.721 | 0.850 | 0.665 | 0.746 |
| EC 3 | Hydrolases | 315 | 0.468 | 0.467 | 0.467 | 0.450 | 0.454 | 0.452 |
| EC 4 | Lyases | 171 | 0.333 | 0.444 | 0.381 | 0.359 | 0.509 | 0.421 |
| EC 5 ★ | Isomerases | 45 | 0.122 | 0.511 | 0.197 | 0.172 | 0.489 | 0.254 |
| EC 6 | Ligases | 77 | 0.189 | 0.597 | 0.287 | 0.267 | 0.662 | 0.381 |
| EC 7 ★ | Translocases | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

> **★ Rare class** — fewer samples than 1 % of class mean or < 50 examples. Low scores for these classes reflect data scarcity, not a model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 3. Summary

- **SmartsTokenizer** is **3.2×** faster than SentencePieceTokenizer (9,153 vs 2,845 SMARTS/s).
- **SmartsTokenizer** produces shorter sequences (fertility 0.2751 vs 0.4476 tokens/char).
- On the EC classification task, **SentencePieceTokenizer** achieves the highest accuracy (0.5858), vs runner-up SmartsTokenizer (0.5400).
