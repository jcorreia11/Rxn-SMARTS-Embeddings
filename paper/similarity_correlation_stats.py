"""
Compute embedding-structural similarity correlation statistics for Methods
section 2.7.2 / Results section 3.5.

Reads the final similarity-correlation result JSON for all three model
sizes (small, medium, large) — these have genuinely different correlation
values per size (see Table~\\ref{tab:sim-corr} in the paper draft), unlike
the EC-classifier dataset stats — and reports per-size statistics plus the
shared sampling/fingerprint protocol (cross-checked for consistency across
sizes, since all three use the same reaction sample and fingerprint).

Reads from the results/final/ mirror using stable, generic filenames (no
run-ID in the path), so this keeps working across retraining runs without
edits.

Output: paper/similarity_correlation_stats.json
Run from project root: python paper/similarity_correlation_stats.py
"""

import json
from pathlib import Path

SIZES = ["small", "medium", "large"]
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

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")
