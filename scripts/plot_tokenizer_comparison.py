"""Generate figures for the tokenizer comparison experiment.

Reads the JSON outputs of compare_tokenizers.py and train_ec_classifier.py
and produces publication-quality figures saved as SVG and PNG.

Figures produced:
  1. token_length_distribution.svg/png  — overlaid histograms of sequence lengths
  2. throughput_fertility.svg/png        — 2-panel: throughput + fertility bar charts
  3. ec_classifier.svg/png               — accuracy/F1 with error bars + per-class F1

Usage
-----
    python scripts/plot_tokenizer_comparison.py \\
        --metrics   results/tokenizer_comparison_<RUN_ID>/tokenizer_metrics.json \\
        --classifier results/tokenizer_comparison_<RUN_ID>/ec_classifier.json \\
        --output-dir results/tokenizer_comparison_<RUN_ID>/figures
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]


def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _save(fig, path: Path, dpi: int) -> None:
    for ext in ("svg", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 1 — token length distribution
# ---------------------------------------------------------------------------


def plot_length_distribution(metrics: list[dict], out_dir: Path, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, r in enumerate(metrics):
        values = r.get("seq_len_values")
        if not values:
            logger.warning("No seq_len_values for %s — skipping histogram", r["name"])
            continue
        ax.hist(
            values,
            bins=60,
            alpha=0.6,
            color=_COLORS[i % len(_COLORS)],
            label=r["name"],
            density=True,
        )
        ax.axvline(
            r["seq_len_median"],
            color=_COLORS[i % len(_COLORS)],
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Sequence length (tokens)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Token Sequence Length Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = out_dir / "token_length_distribution"
    _save(fig, out, dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — throughput + fertility (2 panels)
# ---------------------------------------------------------------------------


def plot_throughput_fertility(metrics: list[dict], out_dir: Path, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["name"] for r in metrics]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Throughput
    throughput = [r.get("throughput_smarts_per_s", 0) for r in metrics]
    bars = ax1.bar(x, throughput, color=_COLORS[: len(names)], width=0.5, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("SMARTS / second", fontsize=11)
    ax1.set_title("Throughput", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, throughput):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Fertility
    fertility = [r.get("fertility_mean", 0) for r in metrics]
    bars2 = ax2.bar(x, fertility, color=_COLORS[: len(names)], width=0.5, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("Tokens per character", fontsize=11)
    ax2.set_title(
        "Fertility Ratio (lower = better compression)", fontsize=13, fontweight="bold"
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, fertility):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out = out_dir / "throughput_fertility"
    _save(fig, out, dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3 — EC classifier results
# ---------------------------------------------------------------------------


def plot_ec_classifier(clf_results: list[dict], out_dir: Path, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    has_per_class = any(r.get("per_class") for r in clf_results)

    if has_per_class:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = None

    names = [r["name"] for r in clf_results]
    x = np.arange(len(names))
    width = 0.25

    metrics_to_plot = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("f1_macro_mean", "f1_macro_std", "F1 Macro"),
        ("f1_weighted_mean", "f1_weighted_std", "F1 Weighted"),
    ]

    for i, (mean_k, std_k, label) in enumerate(metrics_to_plot):
        means = [r.get(mean_k, 0) for r in clf_results]
        stds = [r.get(std_k, 0) for r in clf_results]
        offset = (i - 1) * width
        ax1.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label,
            color=_COLORS[i % len(_COLORS)],
            alpha=0.85,
            zorder=3,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("EC Class Classification", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # Per-class F1
    if ax2 is not None and has_per_class:
        all_classes = sorted(
            {cls for r in clf_results for cls in (r.get("per_class") or {})}
        )
        n_cls = len(all_classes)
        x_cls = np.arange(n_cls)
        width2 = 0.8 / len(clf_results)

        # Collect support per class (use first result that has it)
        support_by_class: dict[str, int] = {}
        for r in clf_results:
            for cls, stats in (r.get("per_class") or {}).items():
                if cls not in support_by_class:
                    support_by_class[cls] = int(stats.get("support", 0))

        # Identify rare classes (support < 1% of mean support or < 50 samples)
        mean_support = (
            sum(support_by_class.values()) / len(support_by_class)
            if support_by_class
            else 0
        )
        rare_threshold = max(50, mean_support * 0.01)
        rare_classes = {
            cls for cls, n in support_by_class.items() if n < rare_threshold
        }

        for i, r in enumerate(clf_results):
            per_class = r.get("per_class") or {}
            f1s = [per_class.get(cls, {}).get("f1", 0) for cls in all_classes]
            offset = (i - len(clf_results) / 2 + 0.5) * width2
            ax2.bar(
                x_cls + offset,
                f1s,
                width2,
                label=r["name"],
                color=_COLORS[i % len(_COLORS)],
                alpha=0.85,
                zorder=3,
            )

        # Shade rare class columns
        for j, cls in enumerate(all_classes):
            if cls in rare_classes:
                ax2.axvspan(
                    j - 0.5,
                    j + 0.5,
                    color="#fee2e2",
                    alpha=0.55,
                    zorder=0,
                    label="_nolegend_",
                )

        # X-tick labels: "EC N\n(n=X)" — asterisk for rare classes
        tick_labels = []
        for cls in all_classes:
            n = support_by_class.get(cls, 0)
            marker = " *" if cls in rare_classes else ""
            tick_labels.append(f"EC {cls}{marker}\n(n={n:,})")

        ax2.set_xticks(x_cls)
        ax2.set_xticklabels(tick_labels, fontsize=9)
        ax2.set_ylabel("F1 Score", fontsize=11)
        ax2.set_ylim(0, 1.05)
        rare_note = "  (* rare class, shaded)" if rare_classes else ""
        ax2.set_title(f"Per-class F1 Score{rare_note}", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = out_dir / "ec_classifier"
    _save(fig, out, dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate tokenizer comparison figures.")
    p.add_argument("--metrics", required=True, help="tokenizer_metrics.json")
    p.add_argument("--classifier", default=None, help="ec_classifier.json (optional)")
    p.add_argument("--output-dir", default="results/figures", help="Output directory")
    p.add_argument(
        "--dpi", type=int, default=300, help="DPI for raster outputs (default: 300)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load(args.metrics)
    logger.info("Loaded metrics for: %s", [r["name"] for r in metrics])

    plot_length_distribution(metrics, out_dir, args.dpi)
    plot_throughput_fertility(metrics, out_dir, args.dpi)

    if args.classifier and Path(args.classifier).exists():
        clf_raw = _load(args.classifier)
        clf = clf_raw["results"] if isinstance(clf_raw, dict) else clf_raw
        logger.info("Loaded classifier results for: %s", [r["name"] for r in clf])
        plot_ec_classifier(clf, out_dir, args.dpi)
    else:
        logger.info("No classifier JSON provided — skipping EC classifier plot.")


if __name__ == "__main__":
    main()
