"""
Compute embedding-structural similarity correlation statistics for Methods
section 2.7.2 / Results section 3.5.

Reads the final similarity-correlation result JSON for all three model
sizes (small, medium, large) — these have genuinely different correlation
values per size (see Table~\\ref{tab:sim-corr} in the paper draft), unlike
the EC-classifier dataset stats — and reports per-size statistics plus the
shared sampling/fingerprint protocol (cross-checked for consistency across
sizes, since all three use the same reaction sample and fingerprint).

Also aggregates the random-init baseline (5 weight-init seeds per size,
from similarity_correlation_random_baseline.sbatch) into mean/std per size,
and reports the pretrained-vs-random-baseline gap. A single random seed is
too noisy to trust as a baseline on its own (Pearson r spread of ~0.12
across seeds for `large`), which is why 5 seeds are averaged here rather
than compared against a single random run.

Reads from the results/final/ mirror using stable, generic filenames (no
run-ID in the path), so this keeps working across retraining runs without
edits.

Output: paper/similarity_correlation_stats.json
Run from project root: python paper/similarity_correlation_stats.py
"""

import json
import statistics
from pathlib import Path

SIZES = ["small", "medium", "large"]
MODEL_SEEDS = [1, 2, 3, 4, 42]
RESULT_ROOT = Path("results/final/3c-similarity-correlation")
OUTPUT_JSON = Path("paper/similarity_correlation_stats.json")


def _load(size: str) -> dict:
    path = RESULT_ROOT / size / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — has similarity_correlation.sbatch been run "
            f"for MODEL_SIZE={size}?"
        )
    return json.loads(path.read_text())


def _load_random(size: str, model_seed: int) -> dict:
    path = RESULT_ROOT / f"{size}_random_s{model_seed}" / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — has similarity_correlation_random_baseline.sbatch "
            f"been run for MODEL_SIZE={size}, MODEL_SEED={model_seed}?"
        )
    return json.loads(path.read_text())


# ── Load all three sizes ──────────────────────────────────────────────────
docs = {size: _load(size) for size in SIZES}

# ── Cross-check sampling protocol is identical across sizes ───────────────
ref_meta = docs["small"]["meta"]
for size in SIZES[1:]:
    meta = docs[size]["meta"]
    for key in ("n_reactions_total", "n_reactions_sampled", "seed"):
        assert meta[key] == ref_meta[key], (
            f"sampling metadata '{key}' differs for {size} vs small "
            f"(expected identical sampling protocol across sizes): "
            f"{meta[key]!r} vs {ref_meta[key]!r}"
        )

# ── Load random-init baseline (5 model seeds per size) ────────────────────
random_docs = {
    size: {seed: _load_random(size, seed) for seed in MODEL_SEEDS} for size in SIZES
}

# Cross-check the random baseline used the same reaction-sampling seed as
# the pretrained runs, so the Tanimoto pairs line up exactly (only the
# model weights differ).
for size in SIZES:
    for seed in MODEL_SEEDS:
        meta = random_docs[size][seed]["meta"]
        for key in ("n_reactions_total", "n_reactions_sampled", "seed"):
            assert meta[key] == ref_meta[key], (
                f"random baseline sampling metadata '{key}' differs for "
                f"{size}/model_seed={seed} vs pretrained (expected identical "
                f"reaction sample across pretrained and random baseline): "
                f"{meta[key]!r} vs {ref_meta[key]!r}"
            )

n_pairs_possible = ref_meta["n_reactions_sampled"] * (ref_meta["n_reactions_sampled"] - 1) // 2

results = {
    "sampling": {
        "n_reactions_total": ref_meta["n_reactions_total"],
        "n_reactions_sampled": ref_meta["n_reactions_sampled"],
        "seed": ref_meta["seed"],
        "n_pairs_possible": n_pairs_possible,
    },
    "fingerprint": {
        "type": "RDKit structural reaction fingerprint",
        "rdkit_function": "rdChemReactions.CreateStructuralFingerprintForReaction",
        "n_bits": 4096,
        "similarity_metric": "Tanimoto coefficient (DataStructs.TanimotoSimilarity)",
    },
    "embedding_similarity": {
        "metric": "cosine similarity",
        "normalisation": "L2 (unit norm)",
        "computation": "upper-triangle pairwise cosine of sampled embedding rows",
    },
    "sizes": {},
    "visualisation": {
        "type": "hexbin density scatter plot",
        "gridsize": 50,
        "overlay": "OLS linear regression",
        "output_figures": [
            f"results/final/3c-similarity-correlation/{size}/figure.pdf" for size in SIZES
        ],
    },
}

for size in SIZES:
    meta = docs[size]["meta"]
    stats = docs[size]["stats"]
    n_pairs_possible_size = meta["n_reactions_sampled"] * (meta["n_reactions_sampled"] - 1) // 2
    results["sizes"][size] = {
        "n_pairs_retained": meta["n_pairs"],
        "pct_pairs_retained": round(meta["n_pairs"] / n_pairs_possible_size * 100, 2),
        "pearson_r": stats["pearson_r"],
        "pearson_p": stats["pearson_p"],
        "spearman_r": round(stats["spearman_r"], 6),
        "spearman_p": stats["spearman_p"],
        "n_pairs": stats["n_pairs"],
        "cosine_mean": stats["cosine_mean"],
        "cosine_std": stats["cosine_std"],
        "tanimoto_mean": stats["tanimoto_mean"],
        "tanimoto_std": stats["tanimoto_std"],
        "total_time_s": meta["total_time_s"],
    }

# ── Random-init baseline: aggregate 5 model seeds per size ────────────────
# A single random seed is too noisy to trust as a lower-bound reference (see
# module docstring), so report mean +/- sample std across seeds, plus the
# individual per-seed values for transparency/reproducibility.
results["random_baseline"] = {
    "n_seeds": len(MODEL_SEEDS),
    "model_seeds": MODEL_SEEDS,
    "sizes": {},
}
results["comparison"] = {"sizes": {}}

for size in SIZES:
    pearson_vals = [random_docs[size][s]["stats"]["pearson_r"] for s in MODEL_SEEDS]
    spearman_vals = [random_docs[size][s]["stats"]["spearman_r"] for s in MODEL_SEEDS]

    results["random_baseline"]["sizes"][size] = {
        "pearson_r_mean": round(statistics.mean(pearson_vals), 6),
        "pearson_r_std": round(statistics.stdev(pearson_vals), 6),
        "spearman_r_mean": round(statistics.mean(spearman_vals), 6),
        "spearman_r_std": round(statistics.stdev(spearman_vals), 6),
        "pearson_r_per_seed": {
            str(s): random_docs[size][s]["stats"]["pearson_r"] for s in MODEL_SEEDS
        },
        "spearman_r_per_seed": {
            str(s): random_docs[size][s]["stats"]["spearman_r"] for s in MODEL_SEEDS
        },
    }

    pretrained_pearson = results["sizes"][size]["pearson_r"]
    pretrained_spearman = results["sizes"][size]["spearman_r"]
    random_pearson_mean = results["random_baseline"]["sizes"][size]["pearson_r_mean"]
    random_spearman_mean = results["random_baseline"]["sizes"][size]["spearman_r_mean"]
    results["comparison"]["sizes"][size] = {
        "pearson_r_gap": round(pretrained_pearson - random_pearson_mean, 6),
        "spearman_r_gap": round(pretrained_spearman - random_spearman_mean, 6),
    }

# ── Print summary ──────────────────────────────────────────────────────────
s = results["sampling"]
f = results["fingerprint"]

print("=== SAMPLING (shared across sizes) ===")
print(f"Total reactions : {s['n_reactions_total']:,}")
print(f"Sampled         : {s['n_reactions_sampled']:,}  (seed={s['seed']})")
print(f"Pairs possible  : {s['n_pairs_possible']:,}")

print()
print("=== FINGERPRINT ===")
print(f"Type            : {f['type']}")
print(f"Bits            : {f['n_bits']}")
print(f"Similarity      : {f['similarity_metric']}")

print()
print("=== CORRELATION STATISTICS BY SIZE ===")
print(f"{'Size':<8} {'Pearson r':>10} {'Spearman ρ':>11} {'N pairs':>12}")
print("-" * 45)
for size in SIZES:
    c = results["sizes"][size]
    print(f"{size:<8} {c['pearson_r']:>10.4f} {c['spearman_r']:>11.4f} {c['n_pairs']:>12,}")

print()
print(f"=== RANDOM-INIT BASELINE ({len(MODEL_SEEDS)} model seeds) VS PRETRAINED ===")
print(f"{'Size':<8} {'Pretrained r':>13} {'Random r (mean±std)':>21} {'Gap':>8}")
print("-" * 55)
for size in SIZES:
    pretrained_r = results["sizes"][size]["pearson_r"]
    rb = results["random_baseline"]["sizes"][size]
    gap = results["comparison"]["sizes"][size]["pearson_r_gap"]
    print(
        f"{size:<8} {pretrained_r:>13.4f} "
        f"{rb['pearson_r_mean']:>10.4f} ± {rb['pearson_r_std']:<7.4f} {gap:>8.4f}"
    )

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")
