"""Generate dataset statistics figures for the Data section of the paper.

Produces a single multi-panel figure (dataset_overview.{fmt}) with:
  Panel A — SMARTS character-length distribution (histogram)
  Panel B — EC class distribution (bar chart, top-level classes)
  Panel C — Train / validation split sizes

Usage
-----
    python scripts/plot_dataset_stats.py
    python scripts/plot_dataset_stats.py \\
        --validated data/processed/validated_smarts.csv \\
        --raw       data/raw/retrorules-v3.0-rhea.csv data/raw/retrorules-v3.0-metanetx.csv \\
        --val-split 0.1 \\
        --output    results/figures/dataset_overview \\
        --format    pdf
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_EC_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#be185d"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_validated(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["valid"].astype(str).str.upper() == "TRUE"].dropna(subset=["smarts"])
    logger.info("Validated SMARTS: %d sequences", len(df))
    return df.reset_index(drop=True)


def load_ec_labels(raw_paths: list[str], validated_smarts: set[str]) -> pd.Series:
    """Return a Series of top-level EC class labels for validated SMARTS."""
    frames = []
    for p in raw_paths:
        if not Path(p).exists():
            logger.warning("Raw file not found, skipping: %s", p)
            continue
        df = pd.read_csv(p, usecols=["TEMPLATE", "ECS", "VALID"])
        df = df[df["VALID"].astype(str).str.upper() == "TRUE"]
        frames.append(df)

    if not frames:
        return pd.Series(dtype=str)

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="TEMPLATE")

    def _top_ec(ecs: str) -> str | None:
        if not isinstance(ecs, str) or not ecs.strip():
            return None
        first = ecs.split(";")[0].strip()
        return first.split(".")[0] if first else None

    raw["ec_class"] = raw["ECS"].apply(_top_ec)
    raw = raw.dropna(subset=["ec_class"])
    raw = raw[raw["TEMPLATE"].isin(validated_smarts)]
    logger.info("Validated SMARTS with EC label: %d", len(raw))
    return raw.set_index("TEMPLATE")["ec_class"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    df: pd.DataFrame,
    ec_labels: pd.Series,
    val_split: float,
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ------------------------------------------------------------------ #
    # Panel A — SMARTS length distribution
    # ------------------------------------------------------------------ #
    ax = axes[0]
    lengths = df["smarts"].str.len()
    p95 = int(lengths.quantile(0.95))
    p50 = int(lengths.median())

    ax.hist(lengths, bins=80, color=_COLORS[0], alpha=0.85, edgecolor="white",
            linewidth=0.3)
    ax.axvline(p50, color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"Median = {p50}")
    ax.axvline(p95, color="#d97706", linestyle=":", linewidth=1.5,
               label=f"95th pct = {p95}")
    ax.set_xlabel("SMARTS length (characters)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("(A) SMARTS Length Distribution", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.97, 0.95,
        f"N = {len(df):,}\nMean = {lengths.mean():.0f}\nMax = {lengths.max():,}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    # ------------------------------------------------------------------ #
    # Panel B — EC class distribution
    # ------------------------------------------------------------------ #
    ax = axes[1]
    if ec_labels.empty:
        ax.text(0.5, 0.5, "EC labels not available\n(raw files not found)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
                color="#888888")
        ax.set_title("(B) EC Class Distribution", fontsize=13, fontweight="bold")
        ax.axis("off")
    else:
        counts = ec_labels.value_counts().sort_index()
        labels = [f"EC {k}\n{_EC_NAMES.get(k, '')}" for k in counts.index]
        colors = [_COLORS[int(k) - 1 % len(_COLORS)] for k in counts.index]
        bars = ax.bar(range(len(counts)), counts.values, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Number of reactions", fontsize=12)
        ax.set_title("(B) EC Class Distribution", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:,}", ha="center", va="bottom", fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel C — Train / validation split
    # ------------------------------------------------------------------ #
    ax = axes[2]
    n_total = len(df)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    split_labels = ["Train", "Validation"]
    split_counts = [n_train, n_val]
    split_colors = [_COLORS[0], _COLORS[1]]

    bars = ax.bar(split_labels, split_counts, color=split_colors,
                  width=0.5, edgecolor="white")
    for bar, val in zip(bars, split_counts):
        pct = val / n_total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of sequences", fontsize=12)
    ax.set_title(
        f"(C) Train / Validation Split\n(val_split = {val_split:.0%})",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(split_counts) * 1.18)
    ax.text(
        0.97, 0.05, f"Total: {n_total:,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout(pad=2.0)
    out = output_path.with_suffix(f".{fmt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dataset overview to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate dataset statistics figures for the paper."
    )
    p.add_argument(
        "--validated",
        default="data/processed/validated_smarts.csv",
        help="Validated SMARTS CSV (default: data/processed/validated_smarts.csv)",
    )
    p.add_argument(
        "--raw",
        nargs="+",
        default=[
            "data/raw/retrorules-v3.0-rhea.csv",
            "data/raw/retrorules-v3.0-metanetx.csv",
        ],
        help="Raw RetroRules CSV files with ECS column (for EC distribution panel)",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation fraction used in training (default: 0.1)",
    )
    p.add_argument(
        "--output",
        default="results/figures/dataset_overview",
        help="Output path without extension (default: results/figures/dataset_overview)",
    )
    p.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_validated(args.validated)
    ec_labels = load_ec_labels(args.raw, set(df["smarts"].tolist()))

    plot_dataset_overview(
        df=df,
        ec_labels=ec_labels,
        val_split=args.val_split,
        output_path=Path(args.output),
        fmt=args.format,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()