"""
Compute embedding-structural similarity correlation statistics for Methods section 2.7.2.

Reads the final medium-model similarity correlation result JSON and reports
the sampling protocol, fingerprint details, and correlation statistics.

Output: paper/similarity_correlation_stats.json
Run from project root: python paper/similarity_correlation_stats.py
"""

import json
from pathlib import Path

RESULT_JSON = Path("results/final/3c-similarity-correlation/medium/results.json")
OUTPUT_JSON = Path("paper/similarity_correlation_stats.json")

# ── Load ───────────────────────────────────────────────────────────────────
doc = json.loads(RESULT_JSON.read_text())
meta = doc["meta"]
stats = doc["stats"]

n_pairs_possible = meta["n_reactions_sampled"] * (meta["n_reactions_sampled"] - 1) // 2
pct_retained = round(meta["n_pairs"] / n_pairs_possible * 100, 2)

results = {
    "source_json": str(RESULT_JSON),
    "sampling": {
        "n_reactions_total": meta["n_reactions_total"],
        "n_reactions_sampled": meta["n_reactions_sampled"],
        "seed": meta["seed"],
        "n_pairs_possible": n_pairs_possible,
        "n_pairs_retained": meta["n_pairs"],
        "pct_pairs_retained": pct_retained,
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
    "statistics": {
        "pearson_r": stats["pearson_r"],
        "pearson_p": stats["pearson_p"],
        "spearman_r": round(stats["spearman_r"], 6),
        "spearman_p": stats["spearman_p"],
        "n_pairs": stats["n_pairs"],
        "cosine_mean": stats["cosine_mean"],
        "cosine_std": stats["cosine_std"],
        "tanimoto_mean": stats["tanimoto_mean"],
        "tanimoto_std": stats["tanimoto_std"],
    },
    "visualisation": {
        "type": "hexbin density scatter plot",
        "gridsize": 50,
        "overlay": "OLS linear regression",
        "output_figures": [
            "results/final/3c-similarity-correlation/medium/sim_corr.pdf",
        ],
    },
    "total_time_s": meta["total_time_s"],
}

# ── Print summary ──────────────────────────────────────────────────────────
s = results["sampling"]
f = results["fingerprint"]
c = results["statistics"]

print("=== SAMPLING ===")
print(f"Total reactions : {s['n_reactions_total']:,}")
print(f"Sampled         : {s['n_reactions_sampled']:,}  (seed={s['seed']})")
print(f"Pairs possible  : {s['n_pairs_possible']:,}")
print(f"Pairs retained  : {s['n_pairs_retained']:,}  ({s['pct_pairs_retained']:.2f}%)")

print()
print("=== FINGERPRINT ===")
print(f"Type            : {f['type']}")
print(f"Bits            : {f['n_bits']}")
print(f"Similarity      : {f['similarity_metric']}")

print()
print("=== CORRELATION STATISTICS ===")
print(f"Pearson  r      : {c['pearson_r']:.4f}  (p = {c['pearson_p']:.2e})")
print(f"Spearman ρ      : {c['spearman_r']:.4f}  (p = {c['spearman_p']:.2e})")
print(f"N pairs         : {c['n_pairs']:,}")
print(f"Cosine  : mean={c['cosine_mean']:.4f}  std={c['cosine_std']:.4f}")
print(f"Tanimoto: mean={c['tanimoto_mean']:.4f}  std={c['tanimoto_std']:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")