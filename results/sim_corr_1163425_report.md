# Embedding vs Structural Similarity Report

**Date:** 2026-04-22  
**Embeddings:** `reaction_embeddings.npy`  
**Reactions sampled:** 3000  
**Pairs evaluated:** 4,498,500

This report assesses whether the learned reaction embeddings capture chemical similarity beyond token-level syntax. For a random sample of reactions, pairwise cosine similarity in embedding space is compared against pairwise Tanimoto similarity over structural reaction fingerprints (RDKit, 4096 bits). A strong positive correlation indicates the model encodes chemistry, not just syntax.

---

## 1. Experimental Environment

### Hardware

| Component | Details |
|-----------|---------|
| CPU | AMD EPYC 7742 64-Core Processor |
| CPU Cores | 128 |
| RAM | 251.7 GB |
| OS | Linux-4.18.0-348.el8.0.2.x86_64-x86_64-with-glibc2.28 |

### Software

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| NumPy | 2.4.3 |
| pandas | 3.0.2 |
| RDKit | 2025.9.6 |
| scikit-learn | 1.8.0 |

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Embeddings file | `reaction_embeddings.npy` |
| SMARTS file | `reaction_smarts.txt` |
| Total reactions in index | 361,751 |
| Reactions sampled | 3,000 |
| Pairs evaluated | 4,498,500 |
| Random seed | 42 |
| Structural fingerprint | RDKit structural reaction fingerprint (4096 bits) |
| Total runtime | 49.4 s |

All pairwise combinations of the sampled reactions are evaluated (upper triangle, excluding self-pairs). Pairs where RDKit fails to parse the reaction SMARTS are dropped.

**Structural similarity** is measured by Tanimoto coefficient over RDKit structural reaction fingerprints, which encode the atom environments present in both reactant and product templates.

**Embedding similarity** is measured by cosine similarity in the learned representation space.

---

## 3. Results

### Correlation

| Metric | Value | p-value |
|--------|-------|---------|
| Pearson $r$ | 0.6065 | < 1e-300 |
| Spearman $\rho$ | 0.5862 | < 1e-300 |
| Pairs | 4,498,500 | — |

### Distribution

| Statistic | Cosine similarity | Tanimoto similarity |
|-----------|-------------------|---------------------|
| Mean | 0.7410 | 0.3384 |
| Std | 0.0961 | 0.1599 |

![Embedding vs structural similarity](results/figures/sim_corr_1163425.pdf)

*Figure: Hexbin density plot of pairwise cosine similarity (embedding space) against Tanimoto similarity (structural reaction fingerprint). Each bin is coloured by the number of reaction pairs it contains. The dashed line shows the linear fit.*

---

## 4. Summary

Across **4,498,500** reaction pairs, the Pearson correlation between embedding cosine similarity and structural Tanimoto similarity is **r = 0.606** (Spearman ρ = 0.586), indicating a **strong** positive relationship.

The embedding space reliably reflects structural reaction similarity: chemically similar reactions are placed close together regardless of superficial syntactic differences.
