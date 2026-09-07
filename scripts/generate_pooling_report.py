"""Generate a self-contained CLS vs mean-pooling ablation report.

Reads the JSON produced by ``ablation_pooling.py`` and an optional environment
JSON from ``collect_env.py``, and writes a Markdown report.

The report covers:
    1. Experimental environment (hardware + software)
    2. Experimental setup (model config, dataset, search parameters)
    3. Results table (all four pooling × head combinations)
    4. Summary (which pooling strategy wins and by how much)

Usage
-----
    python scripts/generate_pooling_report.py \\
        --results results/pooling_ablation.json \\
        --env     results/pooling_ablation_env.json \\
        --output  results/pooling_ablation_report.md
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt(v, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "N/A"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{seconds:.1f} s"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _env_section(env: dict) -> str:
    lines = [
        "## 1. Experimental Environment",
        "",
        "### Hardware",
        "",
        "| Component | Details |",
        "|-----------|---------|",
        f"| GPU | {_opt(env.get('gpu_name'))} |",
        f"| GPU Memory | {_opt(env.get('gpu_memory_gb'), ' GB')} |",
        f"| GPU Count | {_opt(env.get('gpu_count'))} |",
        f"| CUDA Version | {_opt(env.get('cuda_version'))} |",
        f"| Driver Version | {_opt(env.get('driver_version'))} |",
        f"| CPU | {_opt(env.get('cpu_model'))} |",
        f"| CPU Cores | {_opt(env.get('cpu_cores'))} |",
        f"| RAM | {_opt(env.get('ram_gb'), ' GB')} |",
        f"| OS | {_opt(env.get('platform'))} |",
        "",
        "### Software",
        "",
        "| Package | Version |",
        "|---------|---------|",
        f"| Python | {_opt(env.get('python_version'))} |",
        f"| PyTorch | {_opt(env.get('torch_version'))} |",
        f"| NumPy | {_opt(env.get('pkg_numpy'))} |",
        f"| pandas | {_opt(env.get('pkg_pandas'))} |",
        f"| scikit-learn | {_opt(env.get('pkg_scikit_learn'))} |",
        f"| RDKit | {_opt(env.get('pkg_rdkit'))} |",
    ]
    return "\n".join(lines)


def _setup_section(meta: dict) -> str:
    mc = meta.get("model_config", {})
    n_actual = meta.get("n_samples_actual", "N/A")
    n_req = meta.get("n_samples_requested", "N/A")
    label_names = meta.get("label_names", [])

    ec_classes_str = (
        ", ".join(f"EC {c} ({_EC_NAMES.get(c, '?')})" for c in label_names)
        if label_names
        else "N/A"
    )

    lines = [
        "## 2. Experimental Setup",
        "",
        "### Dataset",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Source | RetroRules v3.0 |",
        f"| EC label depth | {meta.get('ec_depth', 'N/A')} |",
        f"| EC classes | {ec_classes_str} |",
        f"| Samples (requested) | {n_req:,} |"
        if isinstance(n_req, int)
        else f"| Samples (requested) | {n_req} |",
        f"| Samples (actual) | {n_actual:,} |"
        if isinstance(n_actual, int)
        else f"| Samples (actual) | {n_actual} |",
        f"| Train / test split | {meta.get('train_test_split', 'N/A')} |",
        f"| CV folds | {meta.get('folds', 'N/A')} |",
        f"| Random seed | {meta.get('random_seed', 'N/A')} |",
        "",
        "### Pretrained Transformer",
        "",
        "| Hyper-parameter | Value |",
        "|-----------------|-------|",
        f"| Hidden dimension (`d_model`) | {mc.get('d_model', 'N/A')} |",
        f"| Attention heads | {mc.get('nhead', 'N/A')} |",
        f"| Encoder layers | {mc.get('num_encoder_layers', 'N/A')} |",
        f"| Feed-forward dimension | {mc.get('dim_feedforward', 'N/A')} |",
        f"| Max sequence length | {mc.get('max_seq_len', 'N/A')} |",
        f"| Vocabulary size | {mc.get('vocab_size', 'N/A'):,} |"
        if isinstance(mc.get("vocab_size"), int)
        else f"| Vocabulary size | {mc.get('vocab_size', 'N/A')} |",
        f"| Weights file | `{Path(meta.get('weights', 'N/A')).name}` |",
        f"| Embedding extraction (both poolings) | {_fmt_time(meta.get('embed_time_s'))} |",
        f"| Total experiment time | {_fmt_time(meta.get('total_time_s'))} |",
        "",
        "Both pooling strategies are extracted from the **same model in a single "
        "forward pass**, so the only variable between conditions is how the "
        "encoder's output sequence is collapsed to a fixed-size vector:",
        "",
        "- **CLS pooling** — uses the representation of the `[BOS]` token.",
        "- **Mean pooling** — averages all non-padding token representations.",
    ]
    return "\n".join(lines)


def _results_section(results: list[dict], figure_path: str | None) -> str:
    if not results:
        return "## 3. Results\n\n_No results available._"

    n_folds = results[0].get("n_folds", "?")
    n_samples = results[0].get("n_samples", "?")
    n_classes = results[0].get("n_classes", "?")

    # Find winners per metric
    def _winner(key: str) -> str:
        return max(results, key=lambda r: r.get(key, 0))["name"]

    w_acc = _winner("accuracy_mean")
    w_f1m = _winner("f1_macro_mean")
    w_f1w = _winner("f1_weighted_mean")

    header = (
        "| Method | Pooling | Head | Accuracy | F1 macro | F1 weighted | CV time (s) |"
    )
    sep = (
        "|--------|---------|------|----------|----------|-------------|-------------|"
    )

    rows = [header, sep]
    for r in results:
        name = r["name"]
        pooling = r.get("pooling", name.split("+")[0]).upper()
        head = r.get("head", name.split("+")[-1])

        def _cell(mean, std, winner_name):
            if mean is None:
                return "N/A"
            s = f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"
            return f"**{s}**" if winner_name == name else s

        rows.append(
            f"| {name} "
            f"| {pooling} "
            f"| {head} "
            f"| {_cell(r.get('accuracy_mean'), r.get('accuracy_std'), w_acc)} "
            f"| {_cell(r.get('f1_macro_mean'), r.get('f1_macro_std'), w_f1m)} "
            f"| {_cell(r.get('f1_weighted_mean'), r.get('f1_weighted_std'), w_f1w)} "
            f"| {r.get('cv_time_s', 'N/A')} |"
        )

    lines = [
        "## 3. Results",
        "",
        f"Evaluated on **{n_samples:,}** labelled SMARTS "
        if isinstance(n_samples, int)
        else f"Evaluated on **{n_samples}** labelled SMARTS ",
        f"across **{n_classes}** EC classes with **{n_folds}**-fold stratified "
        "cross-validation. Bold values indicate the best result per metric.",
        "",
        "\n".join(rows),
    ]

    if figure_path:
        lines += [
            "",
            f"![Pooling ablation]({figure_path})",
            "",
            "*Figure: Accuracy and F1 macro for CLS and mean pooling across both "
            "classifier heads. Error bars show ± 1 standard deviation over CV folds.*",
        ]

    return "\n".join(lines)


def _per_class_section(results: list[dict]) -> str:
    all_classes = sorted({cls for r in results for cls in (r.get("per_class") or {})})
    if not all_classes:
        return ""

    support: dict[str, int] = {}
    for r in results:
        for cls, stats in (r.get("per_class") or {}).items():
            support.setdefault(cls, int(stats.get("support", 0)))

    name_cols = " | ".join(f"F1 ({r['name']})" for r in results)
    header = f"| EC Class | Name | Support | {name_cols} |"
    sep = "|----------|------|---------|" + "---------|" * len(results)

    rows = [header, sep]
    for cls in all_classes:
        name = _EC_NAMES.get(cls, "")
        n = support.get(cls, 0)
        row = f"| EC {cls} | {name} | {n:,} |"
        for r in results:
            f1 = (r.get("per_class") or {}).get(cls, {}).get("f1", 0.0)
            row += f" {f1:.3f} |"
        rows.append(row)

    return "\n".join(
        [
            "## 4. Per-class Breakdown (F1)",
            "",
            "\n".join(rows),
        ]
    )


def _summary_section(results: list[dict]) -> str:
    lines = ["## 5. Summary", ""]

    if not results:
        return "\n".join(lines + ["_No results._"])

    by_name = {r["name"]: r for r in results}

    for head in ("logreg", "mlp"):
        cls_key = f"CLS+{head}"
        mean_key = f"Mean+{head}"
        if cls_key not in by_name or mean_key not in by_name:
            continue
        cls_acc = by_name[cls_key].get("accuracy_mean", 0)
        mean_acc = by_name[mean_key].get("accuracy_mean", 0)
        cls_f1 = by_name[cls_key].get("f1_macro_mean", 0)
        mean_f1 = by_name[mean_key].get("f1_macro_mean", 0)
        winner = "CLS" if cls_acc >= mean_acc else "Mean"
        delta_acc = cls_acc - mean_acc
        delta_f1 = cls_f1 - mean_f1
        lines.append(
            f"- **{head.upper()} head**: {winner} pooling wins. "
            f"CLS accuracy {cls_acc:.4f} vs mean {mean_acc:.4f} "
            f"(Δ acc = {delta_acc:+.4f}; Δ F1 macro = {delta_f1:+.4f})."
        )

    # Overall best
    best = max(results, key=lambda r: r.get("accuracy_mean", 0))
    lines += [
        "",
        f"**Overall best**: {best['name']} "
        f"(accuracy {best.get('accuracy_mean', 0):.4f} ± "
        f"{best.get('accuracy_std', 0):.4f}, "
        f"F1 macro {best.get('f1_macro_mean', 0):.4f} ± "
        f"{best.get('f1_macro_std', 0):.4f}).",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _generate_figure(results: list[dict], output_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping figure.")
        return None

    names = [r["name"] for r in results]
    acc_means = [r.get("accuracy_mean", 0) for r in results]
    acc_stds = [r.get("accuracy_std", 0) for r in results]
    f1_means = [r.get("f1_macro_mean", 0) for r in results]
    f1_stds = [r.get("f1_macro_std", 0) for r in results]

    n = len(names)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(6, n * 1.6), 5))

    ax.bar(
        x - width / 2,
        acc_means,
        width,
        yerr=acc_stds,
        capsize=4,
        label="Accuracy",
        color="#4c72b0",
        alpha=0.85,
        error_kw={"linewidth": 1.2},
    )
    ax.bar(
        x + width / 2,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=4,
        label="F1 macro",
        color="#dd8452",
        alpha=0.85,
        error_kw={"linewidth": 1.2},
    )

    # Shade by pooling strategy
    for i, name in enumerate(names):
        color = "#e8f4f8" if name.startswith("CLS") else "#e8f8e8"
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± std over CV folds)")
    ax.set_title("CLS vs Mean Pooling — EC Classification Ablation")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Group labels
    for label, start, end in [("CLS pooling", 0, 1), ("Mean pooling", 2, 3)]:
        mid = (start + end) / 2
        ax.annotate(
            label,
            xy=(mid, -0.18),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            annotation_clip=False,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    meta: dict,
    results: list[dict],
    env: dict | None,
    figure_path: str | None,
) -> str:
    run_id = meta.get("run_id", "unknown")
    today = date.today().isoformat()

    sections = [
        "# Pooling Ablation Report: CLS vs Mean",
        "",
        f"**Date:** {today}  ",
        f"**Run ID:** `{run_id}`",
        "",
        "This ablation compares **CLS** and **mean** pooling strategies for "
        "collapsing the transformer encoder output into a fixed-size reaction "
        "embedding. Both strategies use the same pretrained weights; the only "
        "difference is how the per-token representations are aggregated. "
        "Downstream performance is measured on EC class prediction.",
        "",
        "---",
        "",
    ]

    if env:
        sections += [_env_section(env), "", "---", ""]
    else:
        sections += [
            "## 1. Experimental Environment",
            "",
            "_Environment information not available. "
            "Pass `--env` to include hardware and software details._",
            "",
            "---",
            "",
        ]

    sections += [_setup_section(meta), "", "---", ""]
    sections += [_results_section(results, figure_path), "", "---", ""]
    sections += [_per_class_section(results), "", "---", ""]
    sections += [_summary_section(results), ""]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a CLS vs mean-pooling ablation report."
    )
    p.add_argument(
        "--results",
        required=True,
        help="Path to pooling_ablation.json produced by ablation_pooling.py",
    )
    p.add_argument(
        "--env",
        default=None,
        help="Environment JSON from collect_env.py (optional; collected live if omitted)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output Markdown file (default: same dir as --results, .md extension)",
    )
    p.add_argument(
        "--figure-format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format (default: pdf)",
    )
    p.add_argument("--no-figure", action="store_true", help="Skip figure generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    raw = json.loads(results_path.read_text())
    meta = raw.get("meta", {})
    results = raw.get("results", [])

    if args.env:
        env = json.loads(Path(args.env).read_text())
        logger.info("Loaded environment from %s", args.env)
    else:
        logger.info("Collecting environment info from current machine ...")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "collect_env", Path(__file__).parent / "collect_env.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = mod.collect_env()

    output_path = Path(args.output) if args.output else results_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_path: str | None = None
    if not args.no_figure:
        fig_out = output_path.with_suffix(f".{args.figure_format}")
        figure_path = _generate_figure(results, fig_out)

    report = build_report(meta, results, env, figure_path)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
