"""Generate dataset statistics figures for the Data section of the paper.

Produces three individual panels saved as SVG + PNG, plus one combined figure:
  Panel A — SMARTS character-length distribution (histogram)
  Panel B — EC class distribution (horizontal bar chart)
  Panel C — Train / validation split sizes

Usage
-----
    python scripts/plot_dataset_stats.py
    python scripts/plot_dataset_stats.py \\
        --validated data/processed/validated_smarts.csv \\
        --raw       data/raw/retrorules-v3.0-rhea.csv data/raw/retrorules-v3.0-metanetx.csv \\
        --val-split 0.1 \\
        --output    results/figures/dataset_overview \\
        --dpi       300
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

# Font sizes (publication-ready)
_FS_TITLE = 14
_FS_LABEL = 12
_FS_TICK = 10
_FS_ANNOT = 9


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

    raw_all = pd.concat(frames, ignore_index=True)
    # Resolve each template's ECS across *all* raw occurrences before deduping —
    # a template that appears in both MetaNetX and Rhea can have the annotation
    # on only one of the duplicate rows, so deduplicate-then-check silently
    # drops it (same fix as paper/dataset_stats.py, commit cb19403).
    has_ecs_mask = raw_all["ECS"].notna() & (raw_all["ECS"].str.strip() != "")
    ecs_resolved = (
        raw_all[has_ecs_mask]
        .drop_duplicates(subset="TEMPLATE")
        .set_index("TEMPLATE")["ECS"]
    )
    raw = raw_all.drop_duplicates(subset="TEMPLATE").copy()
    raw["ECS"] = raw["TEMPLATE"].map(ecs_resolved)

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
# Individual panels
# ---------------------------------------------------------------------------


def _panel_a(df: pd.DataFrame):
    """SMARTS length distribution histogram."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    lengths = df["smarts"].str.len()
    p95 = int(lengths.quantile(0.95))
    p50 = int(lengths.median())

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        lengths, bins=80, color=_COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.3
    )
    ax.axvline(
        p50, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Median = {p50}"
    )
    ax.axvline(
        p95, color="#d97706", linestyle=":", linewidth=1.5, label=f"95th pct = {p95}"
    )

    ax.set_xlabel("SMARTS length (characters)", fontsize=_FS_LABEL)
    ax.set_ylabel("Count", fontsize=_FS_LABEL)
    ax.set_title(
        "(A) SMARTS Length Distribution", fontsize=_FS_TITLE, fontweight="bold", pad=10
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=_FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend — upper right
    ax.legend(fontsize=_FS_TICK, loc="upper right", framealpha=0.9)

    # Stats box — lower right (no overlap with legend)
    ax.text(
        0.97,
        0.60,
        f"N = {len(df):,}\nMean = {lengths.mean():.0f}\nMax = {lengths.max():,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_FS_ANNOT,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9
        ),
    )

    fig.tight_layout()
    return fig


def _panel_b(ec_labels: pd.Series):
    """EC class distribution — horizontal bar chart to avoid label overlap."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(7, 5))

    if ec_labels.empty:
        ax.text(
            0.5,
            0.5,
            "EC labels not available\n(raw files not found)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=_FS_LABEL,
            color="#888888",
        )
        ax.set_title(
            "(B) EC Class Distribution", fontsize=_FS_TITLE, fontweight="bold", pad=10
        )
        ax.axis("off")
        fig.tight_layout()
        return fig

    counts = ec_labels.value_counts().sort_index()
    labels = [f"EC {k}  {_EC_NAMES.get(k, '')}" for k in counts.index]
    colors = [_COLORS[int(k) - 1] for k in counts.index]
    y_pos = list(range(len(counts)))

    bars = ax.barh(
        y_pos, counts.values, color=colors, edgecolor="white", linewidth=0.5, height=0.6
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_FS_TICK)
    ax.invert_yaxis()  # EC 1 at top

    ax.set_xlabel("Number of reactions", fontsize=_FS_LABEL)
    ax.set_title(
        "(B) EC Class Distribution", fontsize=_FS_TITLE, fontweight="bold", pad=10
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=_FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    x_max = max(counts.values)
    ax.set_xlim(0, x_max * 1.22)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            ha="left",
            va="center",
            fontsize=_FS_ANNOT,
        )

    fig.tight_layout()
    return fig


def _panel_c(df: pd.DataFrame, val_split: float):
    """Train / validation split bar chart."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n_total = len(df)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    fig, ax = plt.subplots(figsize=(5, 5))

    bars = ax.bar(
        ["Train", "Validation"],
        [n_train, n_val],
        color=[_COLORS[0], _COLORS[1]],
        width=0.45,
        edgecolor="white",
    )

    for bar, val in zip(bars, [n_train, n_val]):
        pct = val / n_total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n_total * 0.005,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=_FS_LABEL,
            fontweight="bold",
        )

    ax.set_ylabel("Number of sequences", fontsize=_FS_LABEL)
    ax.set_title(
        f"(C) Train / Validation Split\n(val_split = {val_split:.0%})",
        fontsize=_FS_TITLE,
        fontweight="bold",
        pad=10,
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=_FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, n_total * 1.25)

    # Total annotation — upper left, away from both bars
    ax.text(
        0.04,
        0.97,
        f"Total: {n_total:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_FS_TICK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save(fig, path: Path, dpi: int) -> None:
    """Save figure as SVG and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Saved %s", out)


def _save_combined(fig_a, fig_b, fig_c, output_path: Path, dpi: int) -> None:
    """Assemble the three panels into one combined PDF figure."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(19, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[1.4, 1.4, 1.0])

    for src_fig, col in [(fig_a, 0), (fig_b, 1), (fig_c, 2)]:
        src_ax = src_fig.axes[0]
        new_ax = fig.add_subplot(gs[col])
        # Copy rendered content by re-drawing into a new axes
        src_ax.get_figure().canvas.draw()
        src_ax_img = src_ax.get_figure().canvas.buffer_rgba()
        new_ax.imshow(src_ax_img)
        new_ax.axis("off")

    out = output_path.with_suffix(".pdf")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined figure to %s", out)


# ---------------------------------------------------------------------------
# Main plotting entry point
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    df: pd.DataFrame,
    ec_labels: pd.Series,
    val_split: float,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_a = _panel_a(df)
    fig_b = _panel_b(ec_labels)
    fig_c = _panel_c(df, val_split)

    _save(fig_a, output_path.parent / "panel_a_smarts_length", dpi)
    _save(fig_b, output_path.parent / "panel_b_ec_distribution", dpi)
    _save(fig_c, output_path.parent / "panel_c_train_val_split", dpi)

    # Combined PDF
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(19, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, width_ratios=[1.4, 1.4, 1.0])

    panels = [fig_a, fig_b, fig_c]
    for src_fig, col in zip(panels, range(3)):
        new_ax = fig.add_subplot(gs[col])
        src_fig.canvas.draw()
        import numpy as np

        buf = np.frombuffer(src_fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = src_fig.canvas.get_width_height()
        img = buf.reshape(h, w, 4)
        new_ax.imshow(img)
        new_ax.axis("off")

    out_pdf = output_path.with_suffix(".pdf")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for f in panels:
        plt.close(f)
    logger.info("Saved combined figure to %s", out_pdf)


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
        help="Output path stem (default: results/figures/dataset_overview)",
    )
    p.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster outputs (default: 300)"
    )
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
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
