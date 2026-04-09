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
| Samples evaluated | 361,751 | 361,751 |
| Error rate | **0.00%** | 0.00% |
| Throughput (SMARTS/s) | **9,002** | 2,940 |
| Throughput (chars/s) | **3,856,029** | 1,259,283 |
| Seq length — mean | **119.76** | 190.41 |
| Seq length — median | **101** | 161 |
| Seq length — std | **80.88** | 126.32 |
| Seq length — min | 3 | 10 |
| Seq length — max | 763 | 1,433 |
| Fertility ratio (tokens/char) | **0.2758** | 0.4473 |
| Unique tokens used | 4,461 | 988 |
| Vocabulary utilisation | N/A | 98.80% |
| Round-trip fidelity | **100.00%** | 100.00% |

![token_length_distribution](results/tokenizer_comparison_20260409_131636/figures/token_length_distribution.png)


![throughput_fertility](results/tokenizer_comparison_20260409_131636/figures/throughput_fertility.png)


---

## 2. Downstream Task — EC Class Classification

Each tokenizer's sequences are featurised with TF-IDF and fed into a logistic regression classifier predicting the top-level EC enzyme class (oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase, translocase). Higher scores indicate the tokenizer encodes more reaction-type information.

Evaluated on **214007** labelled SMARTS, **7** EC classes, **5**-fold stratified cross-validation.

| Metric | SmartsTokenizer | SentencePieceTokenizer |
|--------|--------|--------|
| Accuracy | 0.5979 ± 0.0009 | **0.6428 ± 0.0026** |
| F1 macro | 0.4479 ± 0.0025 | **0.5083 ± 0.0038** |
| F1 weighted | 0.6270 ± 0.0012 | **0.6658 ± 0.0019** |

![ec_classifier](results/tokenizer_comparison_20260409_131636/figures/ec_classifier.png)


### Per-class breakdown

| EC Class | Name | Support | P (SmartsTokenizer) | R (SmartsTokenizer) | F1 (SmartsTokenizer) | P (SentencePieceTokenizer) | R (SentencePieceTokenizer) | F1 (SentencePieceTokenizer) |
|----------|------|---------|---------|---------|---------|---------|---------|---------|
| EC 1 | Oxidoreductases | 11,314 | 0.676 | 0.517 | 0.586 | 0.739 | 0.569 | 0.643 |
| EC 2 | Transferases | 18,463 | 0.886 | 0.655 | 0.753 | 0.896 | 0.694 | 0.782 |
| EC 3 | Hydrolases | 6,753 | 0.561 | 0.556 | 0.559 | 0.562 | 0.578 | 0.570 |
| EC 4 | Lyases | 3,655 | 0.405 | 0.516 | 0.454 | 0.435 | 0.591 | 0.501 |
| EC 5 | Isomerases | 958 | 0.214 | 0.863 | 0.343 | 0.256 | 0.898 | 0.399 |
| EC 6 | Ligases | 1,645 | 0.253 | 0.771 | 0.381 | 0.296 | 0.795 | 0.431 |
| EC 7 ★ | Translocases | 14 | 0.046 | 0.929 | 0.087 | 0.140 | 1.000 | 0.246 |

> **★ Rare class** — fewer samples than 1 % of class mean or < 50 examples. Low scores for these classes reflect data scarcity, not a model failure. EC 7 (Translocases) is heavily under-represented in RetroRules.

---

## 3. Summary

- **SmartsTokenizer** is **3.1×** faster than SentencePieceTokenizer (9,002 vs 2,940 SMARTS/s).
- **SmartsTokenizer** produces shorter sequences (fertility 0.2758 vs 0.4473 tokens/char).
- On the EC classification task, **SentencePieceTokenizer** achieves the highest accuracy (0.6428), vs runner-up SmartsTokenizer (0.5979).
