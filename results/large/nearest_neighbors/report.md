# Nearest-Neighbor Analysis Report

**Date:** 2026-05-14  
**Embeddings:** `embeddings.npy`  
**Query index:** 335243  
**K:** 10  
**Metric:** cosine

This report presents the top-K nearest neighbors retrieved from the reaction SMARTS embedding space for a given query reaction. EC class labels from RetroRules v3.0 are used to assess whether neighboring reactions are chemically related.

---

## 1. Experimental Environment

### Hardware

| Component | Details |
|-----------|---------|
| CPU | AMD EPYC 7742 64-Core Processor |
| CPU Cores | 128 |
| RAM | 251.7 GB |
| OS | Linux-4.18.0-348.el8.0.2.x86_64-x86_64-with-glibc2.28 |
| GPU | None (CPU-only run) |

### Software

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| NumPy | 2.4.3 |
| pandas | 3.0.2 |
| PyTorch | 2.6.0+cu124 |
| scikit-learn | 1.8.0 |
| RDKit | 2025.9.6 |

---

## 2. Experimental Setup

### Embedding Index

| Parameter | Value |
|-----------|-------|
| Source | RetroRules v3.0 |
| Embeddings file | `embeddings.npy` |
| SMARTS file | `smarts.txt` |
| Total reactions indexed | 361,751 |
| Embedding dimension | 512 |
| Data load time | 983.0 ms |

### Search Parameters

| Parameter | Value |
|-----------|-------|
| Distance metric | cosine |
| Score interpretation | Cosine similarity (higher = closer) |
| Neighbors retrieved (K) | 10 |
| Search time | 343.0 ms |

### Query Reaction

| Field | Value |
|-------|-------|
| Index | 335243 |
| EC class | EC ? — Unknown |
| Full ECS | `N/A` |

**Query SMARTS:** `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C;H0](-[C;H3])=[C;H1]-[C;H2]-[C;H2]-[C;H0](-[C;H3])=[C;H1]-[C;H2]-[C;H2]-[C;H0](-[C;H3])=[C;H1]-[C;H2]-[C;H2]-[C;H0](-[C;H3])=[C;H1]-[C;H2]-[C;H2:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]`

---

## 3. Results

Query reaction is EC ? (Unknown). Neighbors are ranked by cosine sim..

| Rank | Cosine Sim. | EC Class | EC Name | Same EC? | Index | SMARTS |
|------|------------|----------|---------|----------|-------|--------|
| 1 | +0.9974 | EC 1 | Oxidoreductases | No | 163998 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 2 | +0.9969 | EC 2 | Transferases | No | 154421 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 3 | +0.9956 | EC 1 | Oxidoreductases | No | 162691 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]>>...` |
| 4 | +0.9937 | EC 2 | Transferases | No | 98405 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 5 | +0.9927 | EC 2 | Transferases | No | 104160 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]>>...` |
| 6 | +0.9925 | EC ? | Unknown | No | 338220 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 7 | +0.9925 | EC ? | Unknown | No | 335051 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]>>[C;H3]-[C;H0](-[C;...` |
| 8 | +0.9923 | EC 1 | Oxidoreductases | No | 164002 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 9 | +0.9921 | EC 1 | Oxidoreductases | No | 162695 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]-[...` |
| 10 | +0.9920 | EC ? | Unknown | No | 338324 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]>>...` |

---

## 4. Summary

The query reaction has no EC label in the RetroRules database. EC class agreement cannot be assessed.
