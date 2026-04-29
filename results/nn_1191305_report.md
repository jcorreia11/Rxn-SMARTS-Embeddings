# Nearest-Neighbor Analysis Report

**Date:** 2026-04-29  
**Embeddings:** `reaction_embeddings_512.npy`  
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
| Embeddings file | `reaction_embeddings_512.npy` |
| SMARTS file | `reaction_smarts_512.txt` |
| Total reactions indexed | 361,751 |
| Embedding dimension | 256 |
| Data load time | 601.0 ms |

### Search Parameters

| Parameter | Value |
|-----------|-------|
| Distance metric | cosine |
| Score interpretation | Cosine similarity (higher = closer) |
| Neighbors retrieved (K) | 10 |
| Search time | 184.0 ms |

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
| 1 | +0.9985 | EC 1 | Oxidoreductases | No | 163998 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 2 | +0.9983 | EC 2 | Transferases | No | 98405 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 3 | +0.9971 | EC 1 | Oxidoreductases | No | 162691 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]>>...` |
| 4 | +0.9970 | EC 2 | Transferases | No | 154421 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 5 | +0.9968 | EC 1 | Oxidoreductases | No | 164002 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]>>[C;H3]-[C...` |
| 6 | +0.9963 | EC ? | Unknown | No | 335051 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]>>[C;H3]-[C;H0](-[C;...` |
| 7 | +0.9958 | EC 2 | Transferases | No | 76731 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[C;H2:6]>>...` |
| 8 | +0.9958 | EC ? | Unknown | No | 335389 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[O;H0:6]>>...` |
| 9 | +0.9957 | EC 2 | Transferases | No | 336557 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]-[C;H2:5]-[O;H0:6]>>...` |
| 10 | +0.9954 | EC 1 | Oxidoreductases | No | 163977 | `[C;H3:1]-[C;H0:2](-[C;H3:3])=[C;H1:4]>>[C;H3]-[C;H0](-[C;...` |

---

## 4. Summary

The query reaction has no EC label in the RetroRules database. EC class agreement cannot be assessed.
