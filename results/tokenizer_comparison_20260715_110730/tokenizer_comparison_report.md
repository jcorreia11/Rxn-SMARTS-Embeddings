# Tokenizer Comparison Report

**Date:** 2026-07-15  
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
| Samples evaluated | 361,751 | 361,751 |
| Error rate | **0.00%** | 0.00% |
| Throughput (SMARTS/s) | **9,018** | 2,965 |
| Throughput (chars/s) | **3,862,916** | 1,270,049 |
| Seq length — mean | **119.76** | 190.41 |
| Seq length — median | **101** | 161 |
| Seq length — std | **80.88** | 126.32 |
| Seq length — min | 3 | 10 |
| Seq length — max | 763 | 1,433 |
| Fertility ratio (tokens/char) | **0.2758** | 0.4473 |
| Unique tokens used | 4,461 | 988 |
| Vocabulary utilisation | N/A | 98.80% |
| Round-trip fidelity | **100.00%** | 100.00% |

![token_length_distribution](results/tokenizer_comparison_20260715_110730/figures/token_length_distribution.png)


![throughput_fertility](results/tokenizer_comparison_20260715_110730/figures/throughput_fertility.png)


---

## 2. Downstream Task — EC Class Classification

Each tokenizer's sequences are featurised with TF-IDF and fed into a logistic regression classifier predicting the top-level EC enzyme class (oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase, translocase). Higher scores indicate the tokenizer encodes more reaction-type information.

Evaluated on **214007** labelled SMARTS, **7** EC classes, **5**-fold stratified cross-validation.

| Metric | SmartsTokenizer | SentencePieceTokenizer |
|--------|--------|--------|
| Accuracy | 0.5902 ± 0.0042 | **0.6322 ± 0.0019** |
| F1 macro | 0.4273 ± 0.0042 | **0.4627 ± 0.0025** |
| F1 weighted | 0.6197 ± 0.0034 | **0.6560 ± 0.0022** |

![ec_classifier](results/tokenizer_comparison_20260715_110730/figures/ec_classifier.png)


### Per-class breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 18,463 | 0.880 | 0.652 | 0.749 | 0.885 | 0.696 | 0.779 |
| EC 2 | Transferases | 6,753 | 0.563 | 0.549 | 0.556 | 0.554 | 0.570 | 0.562 |
| EC 3 | Hydrolases | 3,655 | 0.405 | 0.521 | 0.456 | 0.439 | 0.606 | 0.509 |
| EC 4 | Lyases | 958 | 0.203 | 0.831 | 0.326 | 0.242 | 0.837 | 0.376 |
| EC 5 | Isomerases | 1,645 | 0.236 | 0.737 | 0.358 | 0.282 | 0.762 | 0.412 |
| EC 6 ★ | Ligases | 13 | 0.013 | 0.231 | 0.024 | 0.040 | 0.231 | 0.068 |

> **★ Rare class** — fewer samples than 1 % of class mean or < 50 examples. Low scores for these classes reflect data scarcity, not a model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 3. Summary

- **SmartsTokenizer** is **3.0×** faster than SentencePieceTokenizer (9,018 vs 2,965 SMARTS/s).
- **SmartsTokenizer** produces shorter sequences (fertility 0.2758 vs 0.4473 tokens/char).
- On the EC classification task, **SentencePieceTokenizer** achieves the highest accuracy (0.6322), vs runner-up SmartsTokenizer (0.5902).
