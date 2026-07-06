# HPC Runbook — Reproducing All Results

Exact `sbatch` commands to regenerate every result in the paper from scratch,
in dependency order. Replaces `runbook.md` and `experiments-dependency-graph.md`
(both retired — this is the single source of truth).

Every phase — including preprocessing and the final paper-statistics
aggregation — is a real `sbatch` script; nothing here needs to be run by
hand on a login node. All commands assume you're in the repo root on the
HPC with a working `../smartrxn_env` virtualenv. `--export=ALL,VAR=value,...`
overrides an `sbatch` script's defaults; any variable not listed keeps the
script's built-in default.

**Two defaults worth knowing before you start, so you don't need to pass
them on every command:**
- `SPLIT_STRATEGY=group` (default in `train_mlm`, `ablation_pooling`,
  `train_ec_classifier`) keeps RetroRules radius-siblings — near-duplicate
  templates at different context radii, ~98% of the corpus has one — out of
  both sides of every train/test and CV split. Only pass `SPLIT_STRATEGY=random`
  if you deliberately want the old leaky split for a before/after comparison
  (see `results/run_log.md`).
- `POOLING=mean` is the established winner (Section 3.3: mean beats CLS at
  every model size) — `extract_embeddings` below uses it directly rather
  than re-deciding per run. The pooling-ablation phase still runs, to keep
  the evidence for that conclusion reproducible per model size.
- `SEED=42` (default in `train_mlm`) controls weight init and data-loader
  shuffling/masking — independent of `SPLIT_SEED`, which only controls the
  train/val split. The commands below submit one seed per size, matching
  the paper's headline numbers. To estimate pretraining run-to-run variance
  (a gap the paper currently only quantifies for the *random-init* baseline,
  via `MODEL_SEED` in Phase 4b — see Section 3.4), resubmit Phase 1 for a
  given `MODEL_SIZE` with additional `SEED` values (e.g. 1, 2, 3) and carry
  each resulting `WEIGHTS`/`CONFIG` through Phases 2–5 independently.

## Model configurations

| Size | d_model | nhead | Layers | FFN | max_seq_len | Batch size |
|---|--:|--:|--:|--:|--:|--:|
| small | 256 | 8 | 6 | 1024 | 256 | 64 |
| medium | 256 | 8 | 6 | 1024 | 512 | 64 |
| large | 512 | 16 | 8 | 2048 | 512 | 32 |

## Dependency graph

```
[preprocess.sbatch]  (CPU, Phase 0, run once)
      |
      +─────────────────────────────┐
      |                             |
[plot_dataset_stats]         [train_mlm] × 3 sizes           <- Phase 0b (CPU)   Phase 1 (GPU, parallel)
[compare_tokenizers]                |
   (Phase 0b, CPU,                  | (per size, independent — only needs that
    parallel, no model needed)      |  size's WEIGHTS/CONFIG, nothing else)
                                     |
              ┌──────────────────────┼──────────────────────┐
              |                      |                      |
     [extract_embeddings]   [ablation_pooling]    [train_ec_classifier]
        × 3 sizes              × 3 sizes            × 3 sizes × 3 depths
     Phase 2 (GPU)           Phase 3a (GPU)           Phase 3b (GPU)
     all parallel,           all parallel,          all 9 parallel,
     parallel w/ 3a & 3b     parallel w/ 2 & 3b     parallel w/ 2 & 3a
              |
              |  (needs Phase 2's embeddings.npy/smarts.txt)
   ┌──────────┼──────────────────┬───────────────────┐
   |          |                  |                    |
[sim_corr] [sim_corr_random]  [nearest_neighbors]  [plot_umap depth=1]
 × 3 sizes   1 job, needs ALL     × 3 sizes            × 3 sizes
Phase 4a    3 sizes' Phase 2    Phase 4c             Phase 4d
            done (4b)                                   |
                                                          | (reuses cached coords)
                                                [plot_umap depth=2,3]
                                                    × 6 (3 sizes × 2 depths)
                                                       Phase 5
                                                          |
                                                [paper_stats.sbatch]
                                            Phase 6 (CPU, after everything above)
```

**Key correction vs. the old docs**: `ablation_pooling` and `train_ec_classifier`
extract their own embeddings directly from `--weights`/`--config` — they do
**not** read `data/embeddings/`. So Phase 2, 3a, and 3b only depend on
Phase 1 (same size) and can all be submitted **together**, immediately after
each model finishes training — no need to wait for embeddings extraction.

`[plot_dataset_stats]` above also covers `paper/plot_length_distributions.py`
and `paper/plot_tokenizer_example.py` — both run as extra steps in the same
job/script (see Phase 0b) rather than getting their own graph node.

---

## Phase 0 — Preprocessing (once, CPU, ~2 min)

Runs `load_data.py -> validate_smarts.py -> build_vocab.py -> sentencepiece_tokenizer.py`
in sequence (no pipeline tool — each step is cheap and deterministic).

```bash
sbatch scripts/preprocess.sbatch
```

Outputs (`data/raw/*`, `data/processed/*`) are local, gitignored, not
auto-synced between machines — copy with rsync/scp, or just re-run this job
on each machine you need them on.

## Phase 0b — Dataset stats & tokenizer comparison (parallel, CPU)

Independent of everything except Phase 0. Submit anytime after it.

`plot_dataset_stats.sbatch` now also runs `paper/plot_length_distributions.py`
(Methods §2.1 sequence-length figures — needs Phase 0's `validated_smarts.csv`
/ `vocab.json` / `sp_tokenizer.model`) and `paper/plot_tokenizer_example.py`
(Methods §2.2.1 tokenizer example figure — no data dependency at all) as
extra steps in the same job, so nothing under `paper/` needs a manual,
undocumented re-run after Phase 6. Both this job and `compare_tokenizers.sbatch`
now also mirror into `results/final/0-dataset-stats/` (the latter under a
`tokenizer_comparison/` subfolder) — same browsing convention as every later
phase, though (like all `results/final/` mirrors) it's a gitignored, local-only
convenience copy, not the git-tracked record.

```bash
sbatch scripts/plot_dataset_stats.sbatch

# N_SAMPLES=0 means "all" (see train_ec_classifier.py --n-samples help) — the
# extrinsic EC-classifier table in the paper (Section 3.1, 214,007 templates,
# 5-fold CV) was produced on the full annotated pool, not a subsample. Passing
# a positive N_SAMPLES here (e.g. 50000) will not reproduce those numbers.
sbatch --export=ALL,N_SAMPLES=0,EC_DEPTH=1,SP_VOCAB_SIZE=1000 \
    scripts/compare_tokenizers.sbatch
```

---

## Phase 1 — Train MLM (GPU, parallel across sizes)

All three are independent jobs — submit together.

```bash
# small
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=256,\
D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,DROPOUT=0.1,MASK_PROB=0.25,\
MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,SEED=42,\
MODEL_SIZE=small \
    scripts/train_mlm.sbatch

# medium
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=64,LR=1e-4,MAX_LENGTH=512,\
D_MODEL=256,NHEAD=8,NUM_LAYERS=6,DIM_FEEDFORWARD=1024,DROPOUT=0.1,MASK_PROB=0.25,\
MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,SEED=42,\
MODEL_SIZE=medium \
    scripts/train_mlm.sbatch

# large (note BATCH_SIZE=32, not 64 — reduced to fit GPU memory)
sbatch --export=ALL,EPOCHS=100,WARMUP_STEPS=2000,BATCH_SIZE=32,LR=1e-4,MAX_LENGTH=512,\
D_MODEL=512,NHEAD=16,NUM_LAYERS=8,DIM_FEEDFORWARD=2048,DROPOUT=0.1,MASK_PROB=0.25,\
MEAN_SPAN=3.0,MAX_SPAN=10,VAL_SPLIT=0.1,MAX_GRAD_NORM=1.0,NUM_WORKERS=4,SEED=42,\
MODEL_SIZE=large \
    scripts/train_mlm.sbatch
```

`SEED` (new — see "Two defaults" above) seeds weight init plus data-loader
shuffling/masking; `SPLIT_SEED` (default 42, not overridden above) separately
seeds the group-aware train/val split. To add a replicate run for variance
estimation, resubmit with a different `SEED` (keep `SPLIT_SEED` fixed so the
train/val partition itself stays identical across replicates) — e.g.
`SEED=1,MODEL_SIZE=small`.

Expected wall time: ~3.4h (small) / ~6.2h (medium) / ~19.5h (large).

Each job prints its `Run ID: YYYYMMDD_HHMMSS` near the top of the log. Once
all three finish, capture the paths (used by every phase below):

```bash
export SMALL_WEIGHTS=models/smarts_transformer_<SMALL_RUN_ID>.pt
export SMALL_CONFIG=models/smarts_transformer_<SMALL_RUN_ID>.json
export MEDIUM_WEIGHTS=models/smarts_transformer_<MEDIUM_RUN_ID>.pt
export MEDIUM_CONFIG=models/smarts_transformer_<MEDIUM_RUN_ID>.json
export LARGE_WEIGHTS=models/smarts_transformer_<LARGE_RUN_ID>.pt
export LARGE_CONFIG=models/smarts_transformer_<LARGE_RUN_ID>.json
```

---

## Phase 2 + 3a + 3b — Embeddings, Pooling Ablation, EC Classifier (GPU, all parallel)

All 15 jobs below depend only on Phase 1 (matching size) — submit them all
together, right after Phase 1 finishes. None of them wait on each other.

### Phase 2 — Extract embeddings (×3, ~3 min each)

```bash
sbatch --export=ALL,WEIGHTS=${SMALL_WEIGHTS},CONFIG=${SMALL_CONFIG},POOLING=mean,\
BATCH_SIZE=256,MODEL_SIZE=small \
    scripts/extract_embeddings.sbatch

sbatch --export=ALL,WEIGHTS=${MEDIUM_WEIGHTS},CONFIG=${MEDIUM_CONFIG},POOLING=mean,\
BATCH_SIZE=256,MODEL_SIZE=medium \
    scripts/extract_embeddings.sbatch

sbatch --export=ALL,WEIGHTS=${LARGE_WEIGHTS},CONFIG=${LARGE_CONFIG},POOLING=mean,\
BATCH_SIZE=256,MODEL_SIZE=large \
    scripts/extract_embeddings.sbatch
```

### Phase 3a — Pooling ablation (×3, ~47 min each)

Run at EC depth=1 only — pooling strategy is a property of the embeddings
themselves, not of the downstream task.

```bash
sbatch --export=ALL,WEIGHTS=${SMALL_WEIGHTS},CONFIG=${SMALL_CONFIG},\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,MODEL_SIZE=small \
    scripts/ablation_pooling.sbatch

sbatch --export=ALL,WEIGHTS=${MEDIUM_WEIGHTS},CONFIG=${MEDIUM_CONFIG},\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,MODEL_SIZE=medium \
    scripts/ablation_pooling.sbatch

sbatch --export=ALL,WEIGHTS=${LARGE_WEIGHTS},CONFIG=${LARGE_CONFIG},\
N_SAMPLES=20000,EC_DEPTH=1,FOLDS=10,EMBED_BATCH_SIZE=128,MODEL_SIZE=large \
    scripts/ablation_pooling.sbatch
```

### Phase 3b — EC classifier (×9 = 3 sizes × 3 depths, ~13–50 min each)

```bash
for SIZE in small medium large; do
  W="${SIZE^^}_WEIGHTS"; C="${SIZE^^}_CONFIG"
  for DEPTH in 1 2 3; do
    sbatch --export=ALL,WEIGHTS=${!W},CONFIG=${!C},POOLING=mean,\
EMBED_BATCH_SIZE=256,N_SAMPLES=50000,FOLDS=10,EC_DEPTH=${DEPTH},MODEL_SIZE=${SIZE} \
        scripts/train_ec_classifier.sbatch
  done
done
```

(Spelled out per-job if your shell doesn't like the indirect-expansion loop:
just substitute `${SMALL_WEIGHTS}`/`${SMALL_CONFIG}`/`MODEL_SIZE=small`, etc.,
for each of the 9 combinations.)

---

## Phase 4 — Similarity Correlation, Random Baseline, Nearest Neighbors, UMAP depth=1

All depend on Phase 2 (embeddings extracted). Submit together once **all
three** Phase 2 jobs have finished (Phase 4b needs all three).

### Phase 4a — Similarity correlation (×3, ~1 min each)

```bash
sbatch --export=ALL,MODEL_SIZE=small,N_REACTIONS=3000,SEED=42,FORMAT=pdf,DPI=300,GRIDSIZE=50 \
    scripts/similarity_correlation.sbatch

sbatch --export=ALL,MODEL_SIZE=medium,N_REACTIONS=3000,SEED=42,FORMAT=pdf,DPI=300,GRIDSIZE=50 \
    scripts/similarity_correlation.sbatch

sbatch --export=ALL,MODEL_SIZE=large,N_REACTIONS=3000,SEED=42,FORMAT=pdf,DPI=300,GRIDSIZE=50 \
    scripts/similarity_correlation.sbatch
```

### Phase 4b — Random-init baseline, all 3 sizes in one job (~5 min)

Needs all three Phase 2 jobs done (reads `data/embeddings/{small,medium,large}/smarts.txt`).
Pre-samples the same indices/seed as Phase 4a so Tanimoto pairs match exactly.

Report in the paper uses 5 model seeds — submit once per seed. `SMALL_CONFIG`/
`MEDIUM_CONFIG`/`LARGE_CONFIG` below are the same variables captured after
Phase 1 (architecture only — random weights are generated fresh from
`MODEL_SEED`, no checkpoint is loaded) — pass them explicitly rather than
relying on the script's defaults, which point at a specific prior run's
config files and won't track a fresh Phase 1 rerun:

```bash
for MSEED in 42 1 2 3 4; do
  sbatch --export=ALL,N_REACTIONS=3000,SEED=42,MODEL_SEED=${MSEED},FORMAT=pdf,DPI=300,GRIDSIZE=50,\
SMALL_CONFIG=${SMALL_CONFIG},MEDIUM_CONFIG=${MEDIUM_CONFIG},LARGE_CONFIG=${LARGE_CONFIG} \
      scripts/similarity_correlation_random_baseline.sbatch
done
```

(`SEED` is the reaction-sampling seed — keep it fixed at 42 to match 4a;
`MODEL_SEED` is what varies across the 5 runs; the loop variable is named
`MSEED`, deliberately not `SEED`, so it can't be confused with the
reaction-sampling seed above it.)

### Phase 4c — Nearest neighbors (×3, <1 min each)

```bash
sbatch --export=ALL,MODEL_SIZE=small,TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42 \
    scripts/nearest_neighbors.sbatch

sbatch --export=ALL,MODEL_SIZE=medium,TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42 \
    scripts/nearest_neighbors.sbatch

sbatch --export=ALL,MODEL_SIZE=large,TOP_K=10,METRIC=cosine,QUERY_MODE=random,SEED=42 \
    scripts/nearest_neighbors.sbatch
```

For specific illustrative examples for the paper's qualitative section
(rather than a random query), add `QUERY_MODE=index,QUERY_INDEX=<N>` (or
`QUERY_MODE=smarts,QUERY_SMARTS=<SMARTS>`) and `QUERY_LABEL=<name>` so it's
mirrored alongside the main run instead of overwriting it, e.g.:
`QUERY_MODE=index,QUERY_INDEX=12345,QUERY_LABEL=ec1_query`.

### Phase 4d — UMAP depth=1 (×3, ~13 min each — fits UMAP from scratch)

Also writes the coordinate cache Phase 5 reuses.

```bash
sbatch --export=ALL,MODEL_SIZE=small,EC_DEPTH=1,FORMAT=pdf,DPI=300 \
    scripts/plot_umap.sbatch

sbatch --export=ALL,MODEL_SIZE=medium,EC_DEPTH=1,FORMAT=pdf,DPI=300 \
    scripts/plot_umap.sbatch

sbatch --export=ALL,MODEL_SIZE=large,EC_DEPTH=1,FORMAT=pdf,DPI=300 \
    scripts/plot_umap.sbatch
```

---

## Phase 5 — UMAP depth=2, depth=3 (×6, ~30 sec each — reads cached coords)

Wait for the matching Phase 4d job (same size) to finish — depth=2/3 reuse
that size's coordinate cache instead of re-fitting UMAP. All 6 can then run
in parallel.

```bash
for SIZE in small medium large; do
  for DEPTH in 2 3; do
    sbatch --export=ALL,MODEL_SIZE=${SIZE},EC_DEPTH=${DEPTH},FORMAT=pdf,DPI=300 \
        scripts/plot_umap.sbatch
  done
done
```

---

## Phase 6 — Paper statistics (CPU, ~3 min, after everything above finishes)

Runs all 9 `paper/*_stats.py` scripts in dependency order (`tokenizer_stats.py`
before `dataset_tokenizer_results.py`, which reads its output; the rest are
independent of each other), commits `paper/*.json` to git, then mirrors
everything into `results/final/4-paper/` via `paper/sync_to_results_final.py`.
Needs every other phase's `results/final/` mirror to already exist on this
machine — this all runs on the HPC, same filesystem as everything else, so
no manual copying is needed between phases.

```bash
sbatch scripts/paper_stats.sbatch
```

---

## Reference: publication-quality argument choices vs. script defaults

| Script | Arg | Script default | Used here | Reason |
|---|---|---|---|---|
| `train_mlm` | `EPOCHS` | 40 | **100** | Ensure convergence |
| `train_mlm` | `WARMUP_STEPS` | 500 | **2000** | Large dataset needs longer warmup |
| `train_mlm` | `BATCH_SIZE` | 128 | **64** (32 for large) | Better gradient signal; large reduced for GPU memory |
| `ablation_pooling` | `N_SAMPLES` | 10000 | **20000** | More reliable CV estimate |
| `ablation_pooling` | `FOLDS` | 5 | **10** | Tighter confidence intervals |
| `train_ec_classifier` | `N_SAMPLES` | 10000 | **50000** | Use as much annotated data as practical (of 214,007 available) |
| `train_ec_classifier` | `FOLDS` | 5 | **10** | Publication standard |
| `similarity_correlation` | `N_REACTIONS` | 1000 | **3000** | ~500k pairs (default) is too small; 3000 gives ~4.5M |
| `compare_tokenizers` (EC step) | `N_SAMPLES` | 10000 | **0 (all)** | Published tokenizer-comparison table uses the full 214,007-template pool, not a subsample |
| `compare_tokenizers` (EC step) | `FOLDS` | 5 | **5 (unchanged)** | Deliberately lighter than Phase 3b's 10 — this is a tokenizer probe, not the headline EC result |

## Estimated wall time

| Phase | Jobs | Time |
|---|---|---|
| 0 — Preprocessing | 1 job (CPU) | ~2 min |
| 0b — Dataset stats + tokenizer comparison | 2 parallel (CPU) | ~4h (tokenizer comparison, dominated by EC classifier step) |
| 1 — Train MLM | 3 parallel (GPU) | 3.4h / 6.2h / 19.5h (small/medium/large) |
| 2 — Extract embeddings | 3 parallel (GPU) | ~3 min |
| 3a — Pooling ablation | 3 parallel (GPU) | ~47 min |
| 3b — EC classifier | 9 parallel (GPU) | ~50 min (depth=1 is the bottleneck) |
| 4a — Similarity correlation | 3 parallel (CPU) | ~1 min |
| 4b — Random-init baseline | 5 sequential-or-parallel jobs (CPU) | ~5 min each |
| 4c — Nearest neighbors | 3 parallel (CPU) | <1 min |
| 4d — UMAP depth=1 | 3 parallel (CPU) | ~13 min |
| 5 — UMAP depth=2,3 | 6 parallel (CPU) | ~30 sec |
| 6 — Paper stats | 1 job (CPU) | ~3 min |

Total wall time after Phase 1 training completes (Phases 2–5, all
parallelizable across sizes): **~1 hour**.