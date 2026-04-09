"""UMAP visualisation of the reaction SMARTS embedding space.

Loads pretrained embeddings (.npy) and their aligned SMARTS list (.txt),
joins EC class labels from the raw RetroRules CSVs, reduces to 2-D with
UMAP, and produces a publication-quality scatter plot coloured by EC class.

The 2-D coordinates are saved alongside the figure so they can be reused
without re-running the reduction (which is expensive for ~361k points).

Requires: umap-learn  (pip install umap-learn)

Usage
-----
    python scripts/plot_umap.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --output     results/figures/umap_embedding_space \\
        --format     pdf

    # Reuse saved 2-D coordinates (skip UMAP recomputation):
    python scripts/plot_umap.py \\
        --embeddings data/embeddings/reaction_embeddings.npy \\
        --smarts     data/embeddings/reaction_smarts.txt \\
        --umap-cache results/figures/umap_coords.npy \\
        --output     results/figures/umap_embedding_space
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_FILES = [
    "data/raw/retrorules-v3.0-metanetx.csv",
    "data/raw/retrorules-v3.0-rhea.csv",
]

_EC_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

# One colour per EC class (1–7), chosen for print and colour-blind friendliness
_EC_COLORS = {
    "1": "#2563eb",  # blue
    "2": "#dc2626",  # red
    "3": "#16a34a",  # green
    "4": "#d97706",  # amber
    "5": "#7c3aed",  # violet
    "6": "#0891b2",  # cyan
    "7": "#be185d",  # pink
}
_UNLABELLED_COLOR = "#d1d5db"  # light grey for points without an EC label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ec_labels(raw_files: list[str], smarts_set: set[str]) -> dict[str, str]:
    """Return {smarts: top-level EC class} for validated SMARTS in *smarts_set*."""
    frames = []
    for p in raw_files:
        if not Path(p).exists():
            logger.warning("Raw file not found, skipping: %s", p)
            continue
        df = pd.read_csv(p, usecols=["TEMPLATE", "ECS", "VALID"])
        df = df[df["VALID"].astype(str).str.upper() == "TRUE"]
        frames.append(df)

    if not frames:
        logger.warning("No raw files loaded — all points will be unlabelled.")
        return {}

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="TEMPLATE")

    def _top_ec(ecs: str) -> str | None:
        if not isinstance(ecs, str) or not ecs.strip():
            return None
        first = ecs.split(";")[0].strip()
        return first.split(".")[0] if first else None

    raw["ec_class"] = raw["ECS"].apply(_top_ec)
    raw = raw.dropna(subset=["ec_class"])
    raw = raw[raw["TEMPLATE"].isin(smarts_set)]
    logger.info("SMARTS with EC label: %d / %d", len(raw), len(smarts_set))
    return dict(zip(raw["TEMPLATE"], raw["ec_class"]))


# ---------------------------------------------------------------------------
# UMAP reduction
# ---------------------------------------------------------------------------


def run_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """Fit UMAP on *embeddings* and return (coords_2d, elapsed_s)."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn") from None

    logger.info(
        "Fitting UMAP on %d × %d matrix (n_neighbors=%d, min_dist=%.2f, metric=%s)...",
        embeddings.shape[0],
        embeddings.shape[1],
        n_neighbors,
        min_dist,
        metric,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
        low_memory=False,
    )
    t0 = time.perf_counter()
    coords = reducer.fit_transform(embeddings)
    elapsed = time.perf_counter() - t0
    logger.info("UMAP complete in %.1f s", elapsed)
    return coords.astype(np.float32), elapsed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_umap(
    coords: np.ndarray,
    labels: np.ndarray,
    umap_params: dict,
    elapsed_s: float | None,
    output_path: Path,
    fmt: str,
    dpi: int,
    point_size: float,
    alpha: float,
) -> None:
    """Scatter plot of 2-D UMAP coordinates coloured by EC class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot unlabelled points first (background)
    mask_unlabelled = labels == ""
    if mask_unlabelled.any():
        ax.scatter(
            coords[mask_unlabelled, 0],
            coords[mask_unlabelled, 1],
            s=point_size,
            c=_UNLABELLED_COLOR,
            alpha=alpha * 0.5,
            linewidths=0,
            rasterized=True,
            label=f"No EC label (n={mask_unlabelled.sum():,})",
            zorder=1,
        )

    # Plot EC classes — rarest last so they sit on top
    ec_classes = sorted(
        [c for c in _EC_COLORS if (labels == c).any()],
        key=lambda c: -(labels == c).sum(),  # most common first → rarest on top
    )
    for ec in ec_classes:
        mask = labels == ec
        name = _EC_NAMES.get(ec, f"EC {ec}")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=_EC_COLORS[ec],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=f"EC {ec} — {name} (n={mask.sum():,})",
            zorder=2,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        "UMAP of Reaction SMARTS Embedding Space",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    legend = ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        markerscale=4,
        title="EC Class",
        title_fontsize=9,
    )
    legend.get_frame().set_linewidth(0.5)

    # Stats box
    n_labelled = (labels != "").sum()
    n_total = len(labels)
    params_str = (
        f"n_neighbors={umap_params.get('n_neighbors', '?')}, "
        f"min_dist={umap_params.get('min_dist', '?')}, "
        f"metric={umap_params.get('metric', '?')}"
    )
    time_str = f", time={elapsed_s:.0f} s" if elapsed_s is not None else ""
    info = f"N = {n_total:,}  ({n_labelled:,} labelled)\nUMAP: {params_str}{time_str}"
    ax.text(
        0.01,
        0.01,
        info,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#555555",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.85,
        ),
    )

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
        description="UMAP visualisation of the reaction SMARTS embedding space."
    )
    p.add_argument(
        "--embeddings",
        required=True,
        help="Path to embeddings .npy file (N × d_model)",
    )
    p.add_argument(
        "--smarts",
        required=True,
        help="Path to aligned SMARTS .txt file (one per line)",
    )
    p.add_argument(
        "--raw",
        nargs="+",
        default=RAW_FILES,
        help="Raw RetroRules CSV files with ECS column (for EC labels)",
    )
    p.add_argument(
        "--umap-cache",
        default=None,
        help="Path to cached 2-D UMAP coordinates .npy (skips UMAP if found; "
        "saves there if not found)",
    )
    p.add_argument(
        "--output",
        default="results/figures/umap_embedding_space",
        help="Output path without extension "
        "(default: results/figures/umap_embedding_space)",
    )
    p.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15)",
    )
    p.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1)",
    )
    p.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "correlation"],
        help="UMAP distance metric (default: cosine)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=0.8,
        help="Scatter plot marker size (default: 0.8)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Scatter plot marker alpha (default: 0.4)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load embeddings ---
    embeddings_path = Path(args.embeddings)
    smarts_path = Path(args.smarts)
    logger.info("Loading embeddings from %s...", embeddings_path)
    embeddings = np.load(embeddings_path)
    logger.info("Embeddings shape: %s", embeddings.shape)

    logger.info("Loading SMARTS list from %s...", smarts_path)
    smarts_list = smarts_path.read_text().splitlines()
    assert len(smarts_list) == embeddings.shape[0], (
        f"Mismatch: {len(smarts_list)} SMARTS vs {embeddings.shape[0]} embeddings"
    )

    # --- EC labels ---
    ec_map = load_ec_labels(args.raw, set(smarts_list))
    labels = np.array([ec_map.get(s, "") for s in smarts_list])
    n_labelled = (labels != "").sum()
    logger.info(
        "EC labels: %d / %d (%.1f%%)",
        n_labelled,
        len(labels),
        100 * n_labelled / len(labels),
    )
    for ec in sorted(_EC_NAMES):
        n = (labels == ec).sum()
        if n:
            logger.info("  EC %s (%s): %d", ec, _EC_NAMES[ec], n)

    # --- UMAP ---
    umap_params = {
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "random_state": args.random_state,
    }

    cache_path = Path(args.umap_cache) if args.umap_cache else None
    elapsed_s: float | None = None

    if cache_path and cache_path.exists():
        logger.info("Loading cached UMAP coordinates from %s", cache_path)
        coords = np.load(cache_path)
        assert coords.shape == (len(smarts_list), 2), (
            f"Cache shape {coords.shape} does not match expected ({len(smarts_list)}, 2)"
        )
    else:
        coords, elapsed_s = run_umap(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, coords)
            logger.info("UMAP coordinates saved to %s", cache_path)

    # --- Plot ---
    plot_umap(
        coords=coords,
        labels=labels,
        umap_params=umap_params,
        elapsed_s=elapsed_s,
        output_path=Path(args.output),
        fmt=args.format,
        dpi=args.dpi,
        point_size=args.point_size,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
