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

_EC_NAMES_D1 = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

# Depth-1 colours — print and colour-blind friendly
_EC_COLORS_D1 = {
    "1": "#2563eb",
    "2": "#dc2626",
    "3": "#16a34a",
    "4": "#d97706",
    "5": "#7c3aed",
    "6": "#0891b2",
    "7": "#be185d",
}

# Sequential colormaps keyed by top-level EC — used for depth-2+ shading
_FAMILY_CMAPS = {
    "1": "Blues",
    "2": "Reds",
    "3": "Greens",
    "4": "Oranges",
    "5": "Purples",
    "6": "GnBu",
    "7": "RdPu",
}

_UNLABELLED_COLOR = "#d1d5db"  # light grey for points without an EC label


def _make_subclass_palette(subclasses: list[str]) -> tuple[dict, dict]:
    """Return (label→color, label→display_name) for depth-2+ subclasses.

    Subclasses are grouped by their top-level EC number and assigned shades
    from that family's sequential colormap so the top-level structure remains
    readable in the figure.
    """
    import matplotlib.cm as cm

    groups: dict[str, list[str]] = {}
    for sc in subclasses:
        top = sc.split(".")[0]
        groups.setdefault(top, []).append(sc)

    colors: dict[str, object] = {}
    names: dict[str, str] = {}
    for top, subs in sorted(groups.items()):
        cmap = cm.get_cmap(_FAMILY_CMAPS.get(top, "Greys"))
        n = len(subs)
        top_name = _EC_NAMES_D1.get(top, f"EC {top}")
        for i, sub in enumerate(sorted(subs)):
            t = 0.4 + 0.5 * (i / max(1, n - 1)) if n > 1 else 0.65
            colors[sub] = cmap(t)
            names[sub] = f"EC {sub} ({top_name[:4]}…)" if len(top_name) > 4 else f"EC {sub} ({top_name})"
    return colors, names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ec_labels(raw_files: list[str], smarts_set: set[str], depth: int = 1) -> dict[str, str]:
    """Return {smarts: EC label truncated to *depth* levels} for validated SMARTS."""
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

    def _ec_at_depth(ecs: str) -> str | None:
        if not isinstance(ecs, str) or not ecs.strip():
            return None
        first = ecs.split(";")[0].strip()
        if not first:
            return None
        parts = first.split(".")
        return ".".join(parts[:depth])

    raw["ec_class"] = raw["ECS"].apply(_ec_at_depth)
    raw = raw.dropna(subset=["ec_class"])
    raw = raw[raw["TEMPLATE"].isin(smarts_set)]
    logger.info("SMARTS with EC label (depth=%d): %d / %d", depth, len(raw), len(smarts_set))
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
    ec_depth: int = 1,
    ec_colors: dict | None = None,
    ec_names: dict | None = None,
) -> None:
    """Scatter plot of 2-D UMAP coordinates coloured by EC class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    colors = ec_colors or _EC_COLORS_D1
    names = ec_names or _EC_NAMES_D1

    # For depth>1 with many subclasses, cap the legend at the top 20 by frequency
    # so it stays readable; the remaining classes are still plotted but unlabelled.
    present = sorted(
        [c for c in colors if (labels == c).any()],
        key=lambda c: -(labels == c).sum(),
    )
    max_legend = 20
    legend_classes = present[:max_legend]
    unlabelled_in_legend = len(present) > max_legend

    fig, ax = plt.subplots(figsize=(10, 8))

    # Unlabelled points first (background)
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

    # Plot all classes — most common first, rarest on top
    for ec in present:
        mask = labels == ec
        name = names.get(ec, f"EC {ec}")
        in_legend = ec in legend_classes
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            c=[colors[ec]],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=f"EC {ec} — {name} (n={mask.sum():,})" if in_legend else None,
            zorder=2,
        )

    if unlabelled_in_legend:
        n_hidden = sum((labels == c).sum() for c in present[max_legend:])
        ax.scatter([], [], s=0, label=f"… +{len(present) - max_legend} subclasses (n={n_hidden:,})")

    depth_suffix = f" (EC depth {ec_depth})" if ec_depth > 1 else ""
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"UMAP of Reaction SMARTS Embedding Space{depth_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ncols = 2 if ec_depth > 1 else 1
    legend = ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        markerscale=4,
        title=f"EC Class (depth {ec_depth})",
        title_fontsize=9,
        ncols=ncols,
    )
    legend.get_frame().set_linewidth(0.5)

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
    p.add_argument(
        "--ec-depth",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="EC label depth: 1='1', 2='1.14', 3='1.14.13' (default: 1)",
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
    ec_map = load_ec_labels(args.raw, set(smarts_list), depth=args.ec_depth)
    labels = np.array([ec_map.get(s, "") for s in smarts_list])
    n_labelled = (labels != "").sum()
    logger.info(
        "EC labels (depth=%d): %d / %d (%.1f%%)",
        args.ec_depth,
        n_labelled,
        len(labels),
        100 * n_labelled / len(labels),
    )

    if args.ec_depth == 1:
        ec_colors = _EC_COLORS_D1
        ec_names = _EC_NAMES_D1
        for ec in sorted(_EC_NAMES_D1):
            n = (labels == ec).sum()
            if n:
                logger.info("  EC %s (%s): %d", ec, _EC_NAMES_D1[ec], n)
    else:
        subclasses = sorted(set(labels[labels != ""]))
        ec_colors, ec_names = _make_subclass_palette(subclasses)
        logger.info("Subclasses found (depth=%d): %d", args.ec_depth, len(subclasses))
        for sc in subclasses:
            n = (labels == sc).sum()
            logger.info("  EC %s: %d", sc, n)

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
        ec_depth=args.ec_depth,
        ec_colors=ec_colors,
        ec_names=ec_names,
    )


if __name__ == "__main__":
    main()
