"""Embedding cosine similarity vs structural Tanimoto similarity.

Randomly samples a set of reactions, computes all pairwise cosine similarities
in embedding space and Tanimoto similarities over RDKit structural reaction
fingerprints (Morgan-based, 4096 bits), then produces a density scatter plot
and reports Pearson and Spearman correlation coefficients.

A strong positive correlation indicates that the learned embeddings capture
chemical similarity beyond token-level syntax.

Requirements: rdkit

Usage
-----
    python scripts/similarity_correlation.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --n-reactions 1000 \\
        --output     results/figures/similarity_correlation

    # Save pairwise data to JSON for reproducibility:
    python scripts/similarity_correlation.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --output     results/figures/similarity_correlation \\
        --save-json  results/similarity_correlation.json
"""

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reaction groups (RetroRules radius-siblings)
# ---------------------------------------------------------------------------


def load_reaction_groups(validated_file: str, smarts_list: list[str]) -> list[str]:
    """Return one reaction_group id per entry in *smarts_list*, aligned by position.

    Falls back to each SMARTS as its own singleton group when the validated
    file is missing the ``reaction_group`` column (or a SMARTS has no match),
    so a sibling-pair breakdown can still be computed (as "no siblings found")
    rather than crashing.
    """
    if not Path(validated_file).exists():
        logger.warning(
            "Validated file not found at %s — cannot attach reaction groups; "
            "sibling-pair breakdown will show 0%% within-group pairs",
            validated_file,
        )
        return list(smarts_list)

    header = pd.read_csv(validated_file, nrows=0).columns
    if "reaction_group" not in header:
        logger.warning(
            "%s has no 'reaction_group' column (re-run `dvc repro` to "
            "regenerate it) — sibling-pair breakdown will show 0%% within-"
            "group pairs",
            validated_file,
        )
        return list(smarts_list)

    df = pd.read_csv(validated_file, usecols=["smarts", "reaction_group"])
    group_map = dict(zip(df["smarts"], df["reaction_group"]))
    return [group_map.get(s, s) for s in smarts_list]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def reaction_fingerprint(smarts: str):
    """Return an RDKit structural reaction fingerprint, or None on parse failure."""
    from rdkit.Chem import rdChemReactions

    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if rxn is None:
            return None
        return rdChemReactions.CreateStructuralFingerprintForReaction(rxn)
    except Exception:
        return None


def tanimoto(fp1, fp2) -> float:
    from rdkit.DataStructs import TanimotoSimilarity
    return TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Cosine similarity (vectorised)
# ---------------------------------------------------------------------------


def pairwise_cosine(emb: np.ndarray) -> np.ndarray:
    """Return the upper-triangle cosine similarities for *emb* (N × d).

    Returns a 1-D array of length N*(N-1)//2.
    """
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    normed = emb / norms
    sim_matrix = normed @ normed.T  # (N, N)
    i, j = np.triu_indices(len(emb), k=1)
    return sim_matrix[i, j]


def pairwise_same_group(groups: list[str]) -> np.ndarray:
    """Return a boolean upper-triangle array, True where a pair shares a
    ``reaction_group`` (RetroRules radius-siblings), aligned with
    ``itertools.combinations(range(n), 2)`` / ``np.triu_indices(n, k=1)``
    ordering (both row-major, so index-compatible with the cosine/Tanimoto
    arrays computed the same way).
    """
    g = np.asarray(groups)
    i, j = np.triu_indices(len(g), k=1)
    return g[i] == g[j]


# ---------------------------------------------------------------------------
# Sampling and computation
# ---------------------------------------------------------------------------


def sample_and_compute(
    embeddings: np.ndarray,
    smarts_list: list[str],
    n_reactions: int,
    seed: int,
    groups: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray | None]:
    """Return (cosine_sims, tanimoto_sims, n_sampled, n_pairs, same_group_mask).

    Randomly samples *n_reactions* from the full set, computes all pairwise
    similarities, and drops pairs where fingerprint parsing failed.
    ``same_group_mask`` (aligned with the returned similarity arrays) is
    ``None`` unless *groups* is provided; it flags pairs that are RetroRules
    radius-siblings of the same underlying reaction, useful for checking that
    the correlation isn't dominated by trivial near-duplicate pairs.
    """
    rng = np.random.default_rng(seed)
    n_total = len(smarts_list)
    n_reactions = min(n_reactions, n_total)

    idx = rng.choice(n_total, size=n_reactions, replace=False)
    idx.sort()
    logger.info("Sampled %d reactions (seed=%d)", n_reactions, seed)

    # --- Structural fingerprints ---
    logger.info("Computing structural reaction fingerprints ...")
    t0 = time.perf_counter()
    fps = []
    valid_mask = []
    for i in idx:
        fp = reaction_fingerprint(smarts_list[i])
        fps.append(fp)
        valid_mask.append(fp is not None)

    n_failed = sum(1 for v in valid_mask if not v)
    if n_failed:
        logger.warning("  %d / %d SMARTS failed to parse — pairs involving them will be dropped", n_failed, n_reactions)
    logger.info("  Fingerprints computed in %.1f s", time.perf_counter() - t0)

    # --- Pairwise Tanimoto (upper triangle) ---
    logger.info("Computing pairwise Tanimoto similarities ...")
    t0 = time.perf_counter()

    pair_indices = list(itertools.combinations(range(n_reactions), 2))
    tanimoto_vals = []
    keep = []
    for a, b in pair_indices:
        if valid_mask[a] and valid_mask[b]:
            tanimoto_vals.append(tanimoto(fps[a], fps[b]))
            keep.append(True)
        else:
            tanimoto_vals.append(float("nan"))
            keep.append(False)

    tanimoto_arr = np.array(tanimoto_vals, dtype=np.float32)
    logger.info("  Tanimoto done in %.1f s", time.perf_counter() - t0)

    # --- Pairwise cosine (vectorised) ---
    logger.info("Computing pairwise cosine similarities ...")
    t0 = time.perf_counter()
    cosine_arr = pairwise_cosine(embeddings[idx].astype(np.float32))
    logger.info("  Cosine done in %.1f s", time.perf_counter() - t0)

    # --- Drop failed pairs ---
    keep_arr = np.array(keep, dtype=bool)
    cosine_arr = cosine_arr[keep_arr]
    tanimoto_arr = tanimoto_arr[keep_arr]

    n_pairs = int(keep_arr.sum())
    logger.info("Retained %d pairs (%.1f%% of %d total)", n_pairs, 100 * n_pairs / len(keep_arr), len(keep_arr))

    same_group_arr = None
    if groups is not None:
        sampled_groups = [groups[i] for i in idx]
        same_group_arr = pairwise_same_group(sampled_groups)[keep_arr]
        n_within = int(same_group_arr.sum())
        logger.info(
            "%d / %d pairs (%.2f%%) are RetroRules radius-siblings (same reaction_group)",
            n_within, n_pairs, 100 * n_within / n_pairs if n_pairs else 0.0,
        )

    return cosine_arr, tanimoto_arr, n_reactions, n_pairs, same_group_arr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(
    cosine: np.ndarray,
    tanimoto: np.ndarray,
    same_group: np.ndarray | None = None,
) -> dict:
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(tanimoto, cosine)
    spearman_r, spearman_p = spearmanr(tanimoto, cosine)

    stats = {
        "pearson_r": round(float(pearson_r), 6),
        "pearson_p": float(pearson_p),
        "spearman_r": round(float(spearman_r), 6),
        "spearman_p": float(spearman_p),
        "n_pairs": len(cosine),
        "cosine_mean": round(float(cosine.mean()), 6),
        "cosine_std": round(float(cosine.std()), 6),
        "tanimoto_mean": round(float(tanimoto.mean()), 6),
        "tanimoto_std": round(float(tanimoto.std()), 6),
    }

    if same_group is not None:
        cross_group = ~same_group
        n_within = int(same_group.sum())
        stats["frac_within_group_pairs"] = round(n_within / len(same_group), 6) if len(same_group) else 0.0
        stats["n_within_group_pairs"] = n_within
        stats["n_cross_group_pairs"] = int(cross_group.sum())
        # Robustness check: is the correlation just an artifact of trivial
        # near-duplicate (radius-sibling) pairs? Recompute on cross-group
        # pairs only. Requires >=2 cross-group pairs with variance.
        if cross_group.sum() >= 2:
            cg_pearson_r, cg_pearson_p = pearsonr(tanimoto[cross_group], cosine[cross_group])
            cg_spearman_r, cg_spearman_p = spearmanr(tanimoto[cross_group], cosine[cross_group])
            stats["cross_group_pearson_r"] = round(float(cg_pearson_r), 6)
            stats["cross_group_pearson_p"] = float(cg_pearson_p)
            stats["cross_group_spearman_r"] = round(float(cg_spearman_r), 6)
            stats["cross_group_spearman_p"] = float(cg_spearman_p)

    return stats


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_correlation(
    cosine: np.ndarray,
    tanimoto: np.ndarray,
    stats: dict,
    output_path: Path,
    fmt: str,
    dpi: int,
    gridsize: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.stats import linregress

    fig, ax = plt.subplots(figsize=(7, 6))

    # Hexbin density plot (handles large pair counts gracefully)
    hb = ax.hexbin(
        tanimoto,
        cosine,
        gridsize=gridsize,
        cmap="YlOrRd",
        mincnt=1,
        linewidths=0.1,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Pair count", fontsize=10)

    # Trend line
    slope, intercept, *_ = linregress(tanimoto, cosine)
    x_line = np.array([float(tanimoto.min()), float(tanimoto.max())])
    ax.plot(x_line, slope * x_line + intercept, color="#1d4ed8", linewidth=1.5,
            linestyle="--", label="Linear fit")

    ax.set_xlabel("Tanimoto similarity (structural reaction fingerprint)", fontsize=11)
    ax.set_ylabel("Cosine similarity (embedding space)", fontsize=11)
    ax.set_title(
        "Embedding Similarity vs Structural Similarity",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)

    # Annotation box
    pr = stats["pearson_r"]
    sr = stats["spearman_r"]
    n = stats["n_pairs"]
    info = (
        f"Pearson $r$ = {pr:.3f}\n"
        f"Spearman $\\rho$ = {sr:.3f}\n"
        f"$n$ = {n:,} pairs"
    )
    ax.text(
        0.97, 0.05, info,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = output_path.with_suffix(f".{fmt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding cosine similarity vs structural Tanimoto similarity."
    )
    p.add_argument(
        "--embeddings",
        default="data/embeddings/reaction_embeddings.npy",
        help="Path to embeddings .npy (N × d)",
    )
    p.add_argument(
        "--smarts",
        default="data/embeddings/reaction_smarts.txt",
        help="Path to aligned SMARTS .txt (one per line)",
    )
    p.add_argument(
        "--validated",
        default="data/processed/validated_smarts.csv",
        help=(
            "Validated SMARTS CSV with a 'reaction_group' column, used to "
            "report the fraction of sampled pairs that are RetroRules "
            "radius-siblings (default: data/processed/validated_smarts.csv)"
        ),
    )
    p.add_argument(
        "--n-reactions",
        type=int,
        default=1000,
        help="Number of reactions to sample (default: 1000 → ~500k pairs)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reaction sampling (default: 42)",
    )
    p.add_argument(
        "--output",
        default="results/figures/similarity_correlation",
        help="Output path without extension (default: results/figures/similarity_correlation)",
    )
    p.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats (default: 300)")
    p.add_argument(
        "--gridsize",
        type=int,
        default=50,
        help="Hexbin grid size (default: 50)",
    )
    p.add_argument(
        "--save-json",
        default=None,
        help="Save statistics and metadata to this JSON path (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading embeddings from %s ...", args.embeddings)
    embeddings = np.load(args.embeddings).astype(np.float32)
    logger.info("  shape: %s", embeddings.shape)

    logger.info("Loading SMARTS list from %s ...", args.smarts)
    smarts_list = Path(args.smarts).read_text().splitlines()
    assert len(smarts_list) == embeddings.shape[0], (
        f"Mismatch: {len(smarts_list)} SMARTS vs {embeddings.shape[0]} embeddings"
    )

    groups = load_reaction_groups(args.validated, smarts_list)

    t0_total = time.perf_counter()
    cosine, tanimoto, n_reactions, n_pairs, same_group = sample_and_compute(
        embeddings, smarts_list, args.n_reactions, args.seed, groups=groups
    )

    logger.info("Computing correlation statistics ...")
    stats = compute_stats(cosine, tanimoto, same_group=same_group)
    total_time_s = time.perf_counter() - t0_total

    logger.info(
        "\nResults:\n"
        "  Pearson  r = %.4f  (p = %.2e)\n"
        "  Spearman ρ = %.4f  (p = %.2e)\n"
        "  N pairs    = %d\n"
        "  Cosine  : mean=%.4f  std=%.4f\n"
        "  Tanimoto: mean=%.4f  std=%.4f",
        stats["pearson_r"], stats["pearson_p"],
        stats["spearman_r"], stats["spearman_p"],
        stats["n_pairs"],
        stats["cosine_mean"], stats["cosine_std"],
        stats["tanimoto_mean"], stats["tanimoto_std"],
    )
    if "cross_group_pearson_r" in stats:
        logger.info(
            "  Within-group (sibling) pairs: %d / %d (%.2f%%)\n"
            "  Cross-group-only  Pearson  r = %.4f\n"
            "  Cross-group-only  Spearman ρ = %.4f",
            stats["n_within_group_pairs"], stats["n_pairs"],
            100 * stats["frac_within_group_pairs"],
            stats["cross_group_pearson_r"], stats["cross_group_spearman_r"],
        )

    plot_correlation(
        cosine, tanimoto, stats,
        output_path=Path(args.output),
        fmt=args.format,
        dpi=args.dpi,
        gridsize=args.gridsize,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "embeddings_path": args.embeddings,
                "smarts_path": args.smarts,
                "validated_path": args.validated,
                "n_reactions_total": embeddings.shape[0],
                "n_reactions_sampled": n_reactions,
                "n_pairs": n_pairs,
                "seed": args.seed,
                "fingerprint": "RDKit structural reaction fingerprint (4096 bits)",
                "total_time_s": round(total_time_s, 2),
            },
            "stats": stats,
        }
        out.write_text(json.dumps(payload, indent=2))
        logger.info("Statistics saved to %s", out)


if __name__ == "__main__":
    main()