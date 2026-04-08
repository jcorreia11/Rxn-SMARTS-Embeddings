"""Generate a self-contained EC-classification comparison report.

Reads the JSON produced by ``train_ec_classifier.py`` (which contains a
``meta`` block and a ``results`` list) and an optional environment JSON from
``collect_env.py``, and writes a publication-quality Markdown report together
with a comparison figure.

The report covers:
    1. Experimental environment (hardware + software)
    2. Experimental setup (dataset, model config if used)
    3. Cross-validation results table (all methods)
    4. Per-class breakdown (P / R / F1 by EC class)
    5. Summary: pretraining gain vs TF-IDF and random-embedding baselines

Usage
-----
    python scripts/generate_ec_report.py \\
        --results results/ec_classifier_<RUN_ID>.json \\
        --env     results/ec_classifier_<RUN_ID>_env.json \\
        --output  results/ec_classifier_<RUN_ID>_report.md
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

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

# Methods that are TF-IDF baselines (no embeddings)
_TFIDF_NAMES = {"SmartsTokenizer", "SentencePieceTokenizer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt(v, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "N/A"


def _pct(v) -> str:
    return f"{v * 100:.2f}%" if v is not None else "N/A"


def _winner(results: list[dict], key: str) -> str:
    candidates = [(r["name"], r.get(key)) for r in results if r.get(key) is not None]
    if not candidates:
        return ""
    return max(candidates, key=lambda x: x[1])[0]


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
        f"| SentencePiece | {_opt(env.get('pkg_sentencepiece'))} |",
        f"| RDKit | {_opt(env.get('pkg_rdkit'))} |",
    ]
    return "\n".join(lines)


def _setup_section(meta: dict) -> str:
    n_actual = meta.get("n_samples_actual", "N/A")
    n_req = meta.get("n_samples_requested", "N/A")
    label_names = meta.get("label_names", [])
    ec_depth = meta.get("ec_depth", "N/A")
    folds = meta.get("folds", "N/A")
    seed = meta.get("random_seed", "N/A")
    split = meta.get("train_test_split", "80/20 stratified")

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
        f"| Source | RetroRules v3.0 |",
        f"| EC label depth | {ec_depth} |",
        f"| EC classes | {ec_classes_str} |",
        f"| Samples (requested) | {n_req:,} |" if isinstance(n_req, int) else
        f"| Samples (requested) | {n_req} |",
        f"| Samples (actual) | {n_actual:,} |" if isinstance(n_actual, int) else
        f"| Samples (actual) | {n_actual} |",
        f"| Train / test split | {split} |",
        f"| Cross-validation folds | {folds} |",
        f"| Random seed | {seed} |",
    ]

    total_s = meta.get("total_time_s")
    if total_s is not None:
        h, rem = divmod(int(total_s), 3600)
        m, s = divmod(rem, 60)
        total_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")
        lines += [f"| Total experiment time | {total_str} |"]

    mc = meta.get("model_config")
    if mc:
        pooling = meta.get("embedding_pooling", "N/A")
        weights = meta.get("embedding_weights", "N/A")
        lines += [
            "",
            "### Pretrained Transformer (embedding experiments)",
            "",
            "| Hyper-parameter | Value |",
            "|-----------------|-------|",
            f"| Hidden dimension (`d_model`) | {mc.get('d_model', 'N/A')} |",
            f"| Attention heads | {mc.get('nhead', 'N/A')} |",
            f"| Encoder layers | {mc.get('num_encoder_layers', 'N/A')} |",
            f"| Feed-forward dimension | {mc.get('dim_feedforward', 'N/A')} |",
            f"| Max sequence length | {mc.get('max_seq_len', 'N/A')} |",
            f"| Vocabulary size | {mc.get('vocab_size', 'N/A'):,} |"
            if isinstance(mc.get("vocab_size"), int) else
            f"| Vocabulary size | {mc.get('vocab_size', 'N/A')} |",
            f"| Pooling strategy | {pooling} |",
            f"| Weights file | `{Path(weights).name if weights != 'N/A' else 'N/A'}` |",
        ]
        t_pt = meta.get("embed_time_pretrained_s")
        t_rand = meta.get("embed_time_random_s")
        n_total = meta.get("n_samples_actual", "?")
        if t_pt is not None:
            lines.append(f"| Embedding extraction (pretrained) | {t_pt:.1f} s ({n_total} sequences) |")
        if t_rand is not None:
            lines.append(f"| Embedding extraction (random-init) | {t_rand:.1f} s ({n_total} sequences) |")
        lines += [
            "",
            "The random-embedding baseline uses the **same architecture with randomly "
            "initialised weights** (no pretraining), isolating the contribution of "
            "the MLM pretraining objective.",
        ]

    return "\n".join(lines)


def _results_table(results: list[dict], figure_path: str | None) -> str:
    if not results:
        return "_No results available._"

    n_folds = results[0].get("n_folds", "?")
    n_samples = results[0].get("n_samples", "?")
    n_classes = results[0].get("n_classes", "?")

    w_acc = _winner(results, "accuracy_mean")
    w_f1m = _winner(results, "f1_macro_mean")
    w_f1w = _winner(results, "f1_weighted_mean")

    header = "| Method | Accuracy | F1 macro | F1 weighted | CV time (s) | Fit time (s) |"
    sep    = "|--------|----------|----------|-------------|-------------|--------------|"

    rows = [header, sep]
    for r in results:
        name  = r["name"]
        acc   = r.get("accuracy_mean")
        acc_s = r.get("accuracy_std")
        f1m   = r.get("f1_macro_mean")
        f1m_s = r.get("f1_macro_std")
        f1w   = r.get("f1_weighted_mean")
        f1w_s = r.get("f1_weighted_std")
        cv_t  = r.get("cv_time_s")
        fit_t = r.get("fit_time_s")

        def _cell(mean, std, winner_name, this_name):
            if mean is None:
                return "N/A"
            s = f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"
            return f"**{s}**" if winner_name == this_name else s

        rows.append(
            f"| {name} "
            f"| {_cell(acc, acc_s, w_acc, name)} "
            f"| {_cell(f1m, f1m_s, w_f1m, name)} "
            f"| {_cell(f1w, f1w_s, w_f1w, name)} "
            f"| {cv_t:.1f} " if cv_t is not None else "| N/A "
            f"| {fit_t:.1f} |" if fit_t is not None else "| N/A |"
        )

    lines = [
        "## 3. Results",
        "",
        f"Evaluated on **{n_samples:,}** labelled SMARTS "
        if isinstance(n_samples, int) else
        f"Evaluated on **{n_samples}** labelled SMARTS ",
        f"across **{n_classes}** EC classes "
        f"with **{n_folds}**-fold stratified cross-validation. "
        "Bold values indicate the best result per metric. "
        "Scores are mean ± std over folds (CV set); final test accuracy is on "
        "the held-out 20 % split.",
        "",
        "\n".join(rows),
    ]

    if figure_path:
        lines += [
            "",
            f"![EC classification comparison]({figure_path})",
            "",
            "*Figure: Accuracy and F1 macro for all methods on EC class prediction. "
            "Error bars show ± 1 standard deviation across CV folds.*",
        ]

    return "\n".join(lines)


def _per_class_table(results: list[dict]) -> str:
    all_classes = sorted({cls for r in results for cls in (r.get("per_class") or {})})
    if not all_classes:
        return ""

    support: dict[str, int] = {}
    for r in results:
        for cls, stats in (r.get("per_class") or {}).items():
            support.setdefault(cls, int(stats.get("support", 0)))

    mean_support = sum(support.values()) / len(support) if support else 0
    rare_threshold = max(50, mean_support * 0.01)

    name_cols = " | ".join(
        f"P ({r['name']}) | R ({r['name']}) | F1 ({r['name']})" for r in results
    )
    header = f"| EC Class | Name | Support | {name_cols} |"
    sep = "|----------|------|---------|" + "---------|" * (3 * len(results))

    rows = [header, sep]
    for cls in all_classes:
        n = support.get(cls, 0)
        rare = " ★" if n < rare_threshold else ""
        name = _EC_NAMES.get(cls, "")
        row = f"| EC {cls}{rare} | {name} | {n:,} |"
        for r in results:
            pc = (r.get("per_class") or {}).get(cls, {})
            row += f" {pc.get('precision', 0.0):.3f} | {pc.get('recall', 0.0):.3f} | {pc.get('f1', 0.0):.3f} |"
        rows.append(row)

    lines = [
        "## 4. Per-class Breakdown",
        "",
        "\n".join(rows),
        "",
    ]
    if any(support.get(cls, 0) < rare_threshold for cls in all_classes):
        lines += [
            "> **★ Rare class** — fewer than 1 % of the per-class sample mean "
            "or < 50 examples. Low scores reflect data scarcity, not model "
            "failure. EC 7 (Translocases) is heavily under-represented in "
            "RetroRules.",
            "",
        ]
    return "\n".join(lines)


def _summary_section(results: list[dict]) -> str:
    lines: list[str] = ["## 5. Summary", ""]

    if not results:
        return "\n".join(lines + ["_No results._"])

    # Best overall
    best = max(results, key=lambda r: r.get("accuracy_mean", 0.0))
    lines.append(
        f"- **Best method**: {best['name']} "
        f"(accuracy {best.get('accuracy_mean', 0):.4f} ± "
        f"{best.get('accuracy_std', 0):.4f}, "
        f"F1 macro {best.get('f1_macro_mean', 0):.4f} ± "
        f"{best.get('f1_macro_std', 0):.4f})."
    )

    # Pretraining gain vs random init
    by_name = {r["name"]: r for r in results}
    for head in ("logreg", "mlp"):
        pt_key = f"Pretrained+{head}"
        rand_key = f"Random+{head}"
        if pt_key in by_name and rand_key in by_name:
            pt_acc = by_name[pt_key].get("accuracy_mean", 0)
            rand_acc = by_name[rand_key].get("accuracy_mean", 0)
            pt_f1 = by_name[pt_key].get("f1_macro_mean", 0)
            rand_f1 = by_name[rand_key].get("f1_macro_mean", 0)
            lines.append(
                f"- **Pretraining gain ({head})**: "
                f"pretrained embeddings achieve accuracy {pt_acc:.4f} vs "
                f"random-init {rand_acc:.4f} "
                f"(Δ acc = {pt_acc - rand_acc:+.4f}; "
                f"Δ F1 macro = {pt_f1 - rand_f1:+.4f}). "
                + (
                    "Positive gain confirms that MLM pretraining encodes "
                    "reaction-type information beyond random projection."
                    if pt_acc > rand_acc
                    else "Gain is negative — the pretrained model may require "
                    "longer training or a different pooling strategy."
                )
            )

    # Pretrained vs best TF-IDF
    pretrained_results = [r for r in results if r["name"].startswith("Pretrained+")]
    tfidf_results = [r for r in results if r["name"] in _TFIDF_NAMES]
    if pretrained_results and tfidf_results:
        best_pt = max(pretrained_results, key=lambda r: r.get("accuracy_mean", 0))
        best_tfidf = max(tfidf_results, key=lambda r: r.get("accuracy_mean", 0))
        pt_acc = best_pt.get("accuracy_mean", 0)
        tf_acc = best_tfidf.get("accuracy_mean", 0)
        lines.append(
            f"- **Embeddings vs TF-IDF**: best pretrained embedding method "
            f"({best_pt['name']}, {pt_acc:.4f}) vs best TF-IDF baseline "
            f"({best_tfidf['name']}, {tf_acc:.4f}) "
            f"(Δ acc = {pt_acc - tf_acc:+.4f})."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _generate_figure(results: list[dict], output_path: Path) -> str | None:
    """Bar chart comparing accuracy and F1 macro for all methods.

    Returns the path written, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping figure generation.")
        return None

    names = [r["name"] for r in results]
    acc_means = [r.get("accuracy_mean", 0) for r in results]
    acc_stds  = [r.get("accuracy_std",  0) for r in results]
    f1_means  = [r.get("f1_macro_mean", 0) for r in results]
    f1_stds   = [r.get("f1_macro_std",  0) for r in results]

    n = len(names)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))

    bars_acc = ax.bar(
        x - width / 2, acc_means, width,
        yerr=acc_stds, capsize=4,
        label="Accuracy", color="#4c72b0", alpha=0.85, error_kw={"linewidth": 1.2}
    )
    bars_f1 = ax.bar(
        x + width / 2, f1_means, width,
        yerr=f1_stds, capsize=4,
        label="F1 macro", color="#dd8452", alpha=0.85, error_kw={"linewidth": 1.2}
    )

    # Colour-code groups
    group_colours = {
        "tfidf":     "#e8f4f8",
        "pretrained": "#e8f8e8",
        "random":    "#f8f0e8",
    }
    for i, name in enumerate(names):
        if name in _TFIDF_NAMES:
            col = group_colours["tfidf"]
        elif name.startswith("Pretrained+"):
            col = group_colours["pretrained"]
        elif name.startswith("Random+"):
            col = group_colours["random"]
        else:
            col = "white"
        ax.axvspan(i - 0.5, i + 0.5, color=col, alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± std over CV folds)")
    ax.set_title("EC Class Classification — Method Comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Group labels
    _add_group_annotations(ax, names, n)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", output_path)
    return str(output_path)


def _add_group_annotations(ax, names: list[str], n: int) -> None:
    """Draw bracketed group labels below the x-axis."""
    groups = []
    i = 0
    while i < n:
        name = names[i]
        if name in _TFIDF_NAMES:
            label = "TF-IDF baselines"
        elif name.startswith("Pretrained+"):
            label = "Pretrained embeddings"
        elif name.startswith("Random+"):
            label = "Random embeddings"
        else:
            i += 1
            continue
        # Find contiguous group
        j = i
        while j < n and (
            (label == "TF-IDF baselines" and names[j] in _TFIDF_NAMES)
            or (label == "Pretrained embeddings" and names[j].startswith("Pretrained+"))
            or (label == "Random embeddings" and names[j].startswith("Random+"))
        ):
            j += 1
        groups.append((i, j - 1, label))
        i = j

    for start, end, label in groups:
        mid = (start + end) / 2
        ax.annotate(
            label,
            xy=(mid, -0.18),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
            annotation_clip=False,
        )


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
        "# EC Class Classification Report",
        "",
        f"**Date:** {today}  ",
        f"**Run ID:** `{run_id}`",
        "",
        "This report compares reaction SMARTS featurisation strategies on the task "
        "of predicting enzyme class (EC number) from reaction templates. "
        "It demonstrates whether transformer embeddings learned via masked language "
        "modelling (MLM) encode more reaction-type information than TF-IDF baselines "
        "and randomly initialised (unlearned) embeddings.",
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
    sections += [_results_table(results, figure_path), "", "---", ""]
    sections += [_per_class_table(results), "---", ""]
    sections += [_summary_section(results), ""]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a publication-quality EC classification comparison report."
    )
    p.add_argument(
        "--results",
        required=True,
        help="Path to ec_classifier_<RUN_ID>.json produced by train_ec_classifier.py",
    )
    p.add_argument(
        "--env",
        default=None,
        help="Environment JSON from collect_env.py (optional; collected live if omitted)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output Markdown file (default: same directory as --results, .md extension)",
    )
    p.add_argument(
        "--figure-format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format (default: pdf)",
    )
    p.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip figure generation",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    results_path = Path(args.results)
    raw = json.loads(results_path.read_text())

    if isinstance(raw, dict) and "results" in raw:
        meta = raw.get("meta", {})
        results = raw["results"]
    else:
        # Legacy: plain list
        meta = {}
        results = raw

    if args.env:
        env = json.loads(Path(args.env).read_text())
        logger.info("Loaded environment from %s", args.env)
    else:
        logger.info("Collecting environment info from current machine...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "collect_env", Path(__file__).parent / "collect_env.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        env = mod.collect_env()

    output_path = (
        Path(args.output) if args.output
        else results_path.with_suffix(".md")
    )
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