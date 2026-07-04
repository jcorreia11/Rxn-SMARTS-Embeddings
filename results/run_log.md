# Run Log — SmartRxnEmbeddings

Maps every generated artifact to the script, job ID, date, and key parameters that produced it.

> ## ⚠️ Train/test split fix — 2026-07-04
>
> RetroRules generates several templates per underlying reaction at different
> context radii (`RADIUS_MIN`/`RADIUS_MAX`). On the full corpus, 361,751
> templates collapse to ~45,600 distinct reactions (mean ~8 templates each,
> 98% with a sibling), and every template in a group shares one EC label.
> **All runs dated before 2026-07-04 split train/test/CV folds at the
> template level with plain (stratified) randomness**, so a template's
> near-duplicate siblings routinely leaked across the split — inflating
> MLM validation accuracy, EC-classification/pooling-ablation metrics, and
> making nearest-neighbor retrieval trivially "succeed" on siblings.
>
> `train_ec_classifier.py`, `ablation_pooling.py`, `similarity_correlation.py`,
> and `nearest_neighbors.py` now support group-aware splitting (see
> `--split-strategy group`, the new default) that keeps each `reaction_group`
> on one side of any split. **Entries below marked `Split strategy: random
> (pre-fix — superseded)` should not be cited in the paper** until replaced
> by a group-split re-run. Use `scripts/generate_split_comparison_report.py`
> to quantify the before/after delta once both runs exist, and log the new
> run in the "Group-split re-runs" section at the bottom of this file.

---

## Phase 0 — Dataset Statistics & Tokenizer Comparison (CPU)

### plot_dataset_stats
| Field | Value |
|---|---|
| Script | `scripts/plot_dataset_stats.sbatch` |
| Date | 2026-04-09 |

**Artifacts**
| File | Description |
|---|---|
| `results/figures/dataset_overview.pdf` | Combined dataset overview figure |
| `results/figures/panel_a_smarts_length.*` | SMARTS length distribution |
| `results/figures/panel_b_ec_distribution.*` | EC class distribution |
| `results/figures/panel_c_train_val_split.*` | Train/val split breakdown |

### compare_tokenizers
| Field | Value |
|---|---|
| Script | `scripts/compare_tokenizers.sbatch` |
| Run ID | `20260409_131636` |
| Date | 2026-04-09 |

**Artifacts**
| File | Description |
|---|---|
| `results/tokenizer_comparison_20260409_131636/` | Full tokenizer comparison results |

---

## Phase 1 — Train MLM (GPU)

### train_mlm — run 20260410_111830 ✅ (used downstream)
| Field | Value |
|---|---|
| Script | `scripts/train_mlm.sbatch` |
| Run ID | `20260410_111830` |
| Date | 2026-04-10 |
| Node | `gnx510` |
| Duration | 12210 s (~3.4 h) |
| Epochs | 100 |
| d_model | 256 |
| nhead | 8 |
| Layers | 6 |
| FFN dim | 1024 |
| Batch size | 64 |
| LR | 1e-4 |
| Warmup steps | 2000 |
| Mask prob | 0.25 |
| Mean span | 3.0 / Max span 10 |
| Val split | 0.1 |

**Final metrics:** val loss 0.1216 · top-1 96.35% · top-5 99.54% · content top-1 95.95%

**Artifacts**
| File | Description |
|---|---|
| `models/smarts_transformer_20260410_111830.pt` | Model weights (DVC-tracked) |
| `models/smarts_transformer_20260410_111830.json` | Model config (DVC-tracked) |
| `models/smarts_transformer_20260410_111830_env.json` | Environment info |
| `models/smarts_transformer_20260410_111830_report.md` | Training report |
| `models/smarts_transformer_20260410_111830_training_loss.pdf` | Loss curve |

### train_mlm — run 20260409_143643 (superseded)
| Field | Value |
|---|---|
| Script | `scripts/train_mlm.sbatch` |
| Run ID | `20260409_143643` |
| Date | 2026-04-09 |

**Artifacts**
| File | Description |
|---|---|
| `models/smarts_transformer_20260409_143643.pt` | Model weights (DVC-tracked) |
| `models/smarts_transformer_20260409_143643.json` | Model config (DVC-tracked) |
| `models/smarts_transformer_20260409_143643_env.json` | Environment info |
| `models/smarts_transformer_20260409_143643_report.md` | Training report |
| `models/smarts_transformer_20260409_143643_training_loss.pdf` | Loss curve |

---

## Phase 2 — Extract Embeddings + Pooling Ablation (GPU)

### extract_embeddings (mean pooling) ✅ (used downstream)
| Field | Value |
|---|---|
| Script | `scripts/extract_embeddings.sbatch` |
| Date | 2026-04-22 |
| Weights | `models/smarts_transformer_20260410_111830.pt` |
| Pooling | `mean` (winner from ablation) |
| Batch size | 256 |

**Artifacts**
| File | Description |
|---|---|
| `data/embeddings/reaction_embeddings.npy` | Reaction embeddings — 361751 × 256 (DVC-tracked) |
| `data/embeddings/reaction_smarts.txt` | Corresponding SMARTS list (DVC-tracked) |

### ablation_pooling — job 1163359 ⚠️ (superseded — see split fix note above)
| Field | Value |
|---|---|
| Script | `scripts/ablation_pooling.sbatch` |
| Job ID | `1163359` |
| Date | 2026-04-22 |
| Node | `gnx506` |
| Weights | `models/smarts_transformer_20260410_111830.pt` |
| N samples | 20000 |
| EC depth | 1 |
| Folds | 10 |
| Embed batch size | 128 |
| Split strategy | random (pre-fix — superseded, re-run with `--split-strategy group`) |

**Results summary**

| Strategy | Accuracy | F1 macro |
|---|---|---|
| CLS + logreg | 0.592 | 0.474 |
| CLS + mlp | 0.737 | 0.584 |
| Mean + logreg | 0.671 | 0.556 |
| **Mean + mlp** | **0.796** | **0.646** |

**Winner: mean pooling**

**Artifacts**
| File | Description |
|---|---|
| `results/pooling_ablation_1163359.json` | Full results |
| `results/pooling_ablation_1163359_env.json` | Environment info |
| `results/pooling_ablation_1163359_report.md` | Report |
| `results/pooling_ablation_1163359_report.pdf` | Figure |

*(Jobs 1163317 and 1163332 failed — torch install error and pandas groupby bug respectively)*

---

## Phase 3 — Evaluation (CPU)

### plot_umap — job 1163482
| Field | Value |
|---|---|
| Script | `scripts/plot_umap.sbatch` |
| Job ID | `1163482` |
| Date | 2026-04-22 |
| Node | `gnx504` |
| Embeddings | `data/embeddings/reaction_embeddings.npy` |
| n_neighbors | 15 |
| min_dist | 0.1 |
| metric | cosine |
| random_state | 42 |
| Duration | 689 s |

**Artifacts**
| File | Description |
|---|---|
| `results/figures/umap_embedding_space.pdf` | UMAP plot (361751 points, 7 EC classes) |
| `results/figures/reaction_embeddings_umap_coords.npy` | Cached 2D coordinates |

### similarity_correlation — jobs 1163425 / 1163426
| Field | Value |
|---|---|
| Script | `scripts/similarity_correlation.sbatch` |
| Job IDs | `1163425`, `1163426` |
| Date | 2026-04-22 |
| Node | `cnx215` |
| Embeddings | `data/embeddings/reaction_embeddings.npy` |
| N reactions | 3000 (~4.5M pairs) |
| Seed | 42 |

**Results:** Pearson r = 0.6065 · Spearman ρ = 0.5862 · N pairs = 4,498,500

**Artifacts**
| File | Description |
|---|---|
| `results/sim_corr_1163425.json` / `sim_corr_1163426.json` | Statistics |
| `results/sim_corr_1163425_report.md` / `sim_corr_1163426_report.md` | Reports |
| `results/figures/sim_corr_1163425.pdf` / `sim_corr_1163426.pdf` | Figures |

### train_ec_classifier (EC depth=1) — run 20260422_112754 ⚠️ (superseded — see split fix note above)
| Field | Value |
|---|---|
| Script | `scripts/train_ec_classifier.sbatch` |
| Run ID | `20260422_112754` |
| Date | 2026-04-22 |
| Node | `gnx504` |
| Weights | `models/smarts_transformer_20260410_111830.pt` |
| Pooling | `mean` |
| N samples | 50000 |
| EC depth | 1 |
| Folds | 10 |
| Split strategy | random (pre-fix — superseded, re-run with `--split-strategy group`) |

**Results summary**

| Method | Accuracy | F1 macro |
|---|---|---|
| Random + logreg | 0.486 | 0.345 |
| SmartsTokenizer + logreg | 0.577 | 0.420 |
| SentencePiece + logreg | 0.623 | 0.470 |
| Pretrained + logreg | 0.676 | 0.525 |
| Random + mlp | 0.692 | 0.464 |
| **Pretrained + mlp** | **0.838** | **0.666** |

**Artifacts**
| File | Description |
|---|---|
| `results/ec_classifier_20260422_112754.json` | Full results |
| `results/ec_classifier_20260422_112754_env.json` | Environment info |
| `results/ec_classifier_20260422_112754_report.md` | Report |
| `results/ec_classifier_20260422_112754_report.pdf` | Figure |

---

## Phase 4 — Nearest Neighbors (CPU)

### nearest_neighbors — job 1163543
| Field | Value |
|---|---|
| Script | `scripts/nearest_neighbors.sbatch` |
| Job ID | `1163543` |
| Date | 2026-04-22 |
| Node | `cnx198` |
| Embeddings | `data/embeddings/reaction_embeddings.npy` |
| Top K | 10 |
| Metric | cosine |
| Query mode | random |
| Seed | 42 |

**Artifacts**
| File | Description |
|---|---|
| `results/nn_1163543.json` | Results |
| `results/nn_1163543_env.json` | Environment info |
| `results/nn_1163543_report.md` | Report |

---

## Phase 3 (extra) — EC Classifier depth=2

### train_ec_classifier (EC depth=2) — pending
| Field | Value |
|---|---|
| Script | `scripts/train_ec_classifier.sbatch` |
| Weights | `models/smarts_transformer_20260410_111830.pt` |
| Pooling | `mean` |
| N samples | 50000 |
| EC depth | 2 |
| Folds | 10 |

**Submit command (from `scripts/`):**
```bash
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,POOLING=mean,N_SAMPLES=50000,EC_DEPTH=2,FOLDS=10,EMBED_BATCH_SIZE=256 train_ec_classifier.sbatch
```

---

## Group-split re-runs (leakage fix) — pending

Preprocessing already re-ran locally with the fix (`dvc repro load_data
validate_smarts`, 2026-07-04): `validated_smarts.csv` now has a
`reaction_group` column (45,596 groups over 361,751 templates, 97.7% with a
sibling). **`dvc push` this to the HPC remote (or re-run `dvc repro` there)
before submitting the jobs below**, since they read `validated_smarts.csv`.

`train_ec_classifier.sbatch` and `ablation_pooling.sbatch` now accept
`SPLIT_STRATEGY` (default `group` — no need to pass it explicitly for the
new correct behavior). To reproduce the superseded runs above for a
before/after comparison, submit both strategies with matching config:

```bash
# EC classifier depth=1 — group (correct) and random (comparison-only)
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,POOLING=mean,N_SAMPLES=50000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=256,SPLIT_STRATEGY=group  train_ec_classifier.sbatch
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,POOLING=mean,N_SAMPLES=50000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=256,SPLIT_STRATEGY=random train_ec_classifier.sbatch

# Pooling ablation — group (correct) and random (comparison-only)
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,SPLIT_STRATEGY=group  ablation_pooling.sbatch
sbatch --export=ALL,WEIGHTS=models/smarts_transformer_20260410_111830.pt,CONFIG=models/smarts_transformer_20260410_111830.json,N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,SPLIT_STRATEGY=random ablation_pooling.sbatch
```

Once both `group` and `random` runs for a given experiment have completed:

```bash
python scripts/generate_split_comparison_report.py \
    --random results/ec_classifier_<RANDOM_RUN_ID>.json \
    --group  results/ec_classifier_<GROUP_RUN_ID>.json \
    --label  "EC classifier (depth=1)" \
    --output results/ec_classifier_split_comparison.md
```

Log the new `group`-strategy run under Phase 2/3 above (replacing the
superseded entries) once complete, and keep the `random` run only as the
comparison artifact — it should not be cited as a paper result.

**Note:** `results/run_log.md` itself is missing entries for a number of
later runs already present under `results/` (multiple `ec_classifier_*` and
`models/smarts_transformer_*` runs from 2026-04-28 through 2026-05-14) —
that gap predates this fix and is a separate cleanup.