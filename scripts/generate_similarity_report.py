"""Generate a self-contained embedding vs structural similarity report.

Reads the JSON produced by ``similarity_correlation.py`` (which contains a
``meta`` block and a ``stats`` block) and an optional environment JSON from
``collect_env.py``, and writes a Markdown report.

The report covers:
    1. Experimental environment (hardware + software)
    2. Experimental setup (embeddings, fingerprint, sampling parameters, timing)
    3. Results (Pearson r, Spearman ρ, distribution statistics)
    4. Summary (interpretation of the correlation strength)

Usage
-----
    python scripts/generate_similarity_report.py \\
        --results results/sim_corr_<JOB_ID>.json \\
        --env     results/sim_corr_<JOB_ID>_env.json \\
        --figure  results/figures/sim_corr_<JOB_ID>.pdf \\
        --output  results/sim_corr_<JOB_ID>_report.md
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "N/A"
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


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
        f"| NumPy | {_opt(env.get('pkg_numpy'))} |",
        f"| pandas | {_opt(env.get('pkg_pandas'))} |",
        f"| RDKit | {_opt(env.get('pkg_rdkit'))} |",
        f"| scikit-learn | {_opt(env.get('pkg_scikit_learn'))} |",
    ]
    return "\n".join(lines)


def _setup_section(meta: dict) -> str:
    n_total = meta.get("n_reactions_total", "N/A")
    n_sampled = meta.get("n_reactions_sampled", "N/A")
    n_pairs = meta.get("n_pairs", "N/A")

    lines = [
        "## 2. Experimental Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Embeddings file | `{Path(meta.get('embeddings_path', 'N/A')).name}` |",
        f"| SMARTS file | `{Path(meta.get('smarts_path', 'N/A')).name}` |",
        f"| Total reactions in index | {n_total:,} |"
        if isinstance(n_total, int)
        else f"| Total reactions in index | {n_total} |",
        f"| Reactions sampled | {n_sampled:,} |"
        if isinstance(n_sampled, int)
        else f"| Reactions sampled | {n_sampled} |",
        f"| Pairs evaluated | {n_pairs:,} |"
        if isinstance(n_pairs, int)
        else f"| Pairs evaluated | {n_pairs} |",
        f"| Random seed | {meta.get('seed', 'N/A')} |",
        f"| Structural fingerprint | {meta.get('fingerprint', 'N/A')} |",
        f"| Total runtime | {_fmt_time(meta.get('total_time_s'))} |",
        "",
        "All pairwise combinations of the sampled reactions are evaluated "
        "(upper triangle, excluding self-pairs). "
        "Pairs where RDKit fails to parse the reaction SMARTS are dropped.",
        "",
        "**Structural similarity** is measured by Tanimoto coefficient over "
        "RDKit structural reaction fingerprints, which encode the atom "
        "environments present in both reactant and product templates.",
        "",
        "**Embedding similarity** is measured by cosine similarity in the "
        "learned representation space.",
    ]
    return "\n".join(lines)


def _results_section(stats: dict, figure_path: str | None) -> str:
    pr = stats.get("pearson_r")
    pp = stats.get("pearson_p")
    sr = stats.get("spearman_r")
    sp = stats.get("spearman_p")
    n = stats.get("n_pairs", "N/A")

    lines = [
        "## 3. Results",
        "",
        "### Correlation",
        "",
        "| Metric | Value | p-value |",
        "|--------|-------|---------|",
        f"| Pearson $r$ | {pr:.4f} | {_fmt_p(pp)} |"
        if pr is not None else "| Pearson $r$ | N/A | N/A |",
        f"| Spearman $\\rho$ | {sr:.4f} | {_fmt_p(sp)} |"
        if sr is not None else "| Spearman $\\rho$ | N/A | N/A |",
        f"| Pairs | {n:,} | — |"
        if isinstance(n, int) else f"| Pairs | {n} | — |",
        "",
        "### Distribution",
        "",
        "| Statistic | Cosine similarity | Tanimoto similarity |",
        "|-----------|-------------------|---------------------|",
        f"| Mean | {stats.get('cosine_mean', 'N/A'):.4f} | {stats.get('tanimoto_mean', 'N/A'):.4f} |",
        f"| Std | {stats.get('cosine_std', 'N/A'):.4f} | {stats.get('tanimoto_std', 'N/A'):.4f} |",
    ]

    if figure_path:
        lines += [
            "",
            f"![Embedding vs structural similarity]({figure_path})",
            "",
            "*Figure: Hexbin density plot of pairwise cosine similarity (embedding "
            "space) against Tanimoto similarity (structural reaction fingerprint). "
            "Each bin is coloured by the number of reaction pairs it contains. "
            "The dashed line shows the linear fit.*",
        ]

    return "\n".join(lines)


def _summary_section(stats: dict) -> str:
    pr = stats.get("pearson_r", 0.0)
    sr = stats.get("spearman_r", 0.0)
    n = stats.get("n_pairs", 0)

    r_ref = max(abs(pr), abs(sr))

    if r_ref >= 0.6:
        strength = "strong"
        interpretation = (
            "The embedding space reliably reflects structural reaction similarity: "
            "chemically similar reactions are placed close together regardless of "
            "superficial syntactic differences."
        )
    elif r_ref >= 0.35:
        strength = "moderate"
        interpretation = (
            "The embedding space partially captures structural similarity. "
            "The model has learned reaction-type information beyond syntax, "
            "but some chemical signal is still absent — possibly due to limited "
            "training data or the expressiveness of the fingerprint."
        )
    else:
        strength = "weak"
        interpretation = (
            "The correlation between embedding and structural similarity is weak. "
            "The embeddings may primarily encode syntactic rather than chemical "
            "features, or the fingerprint may not be the right proxy for the "
            "similarity the model has learned."
        )

    lines = [
        "## 4. Summary",
        "",
        f"Across **{n:,}** reaction pairs, the Pearson correlation between "
        f"embedding cosine similarity and structural Tanimoto similarity is "
        f"**r = {pr:.3f}** (Spearman ρ = {sr:.3f}), indicating a **{strength}** "
        f"positive relationship.",
        "",
        interpretation,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    meta: dict,
    stats: dict,
    env: dict | None,
    figure_path: str | None,
) -> str:
    today = date.today().isoformat()

    sections = [
        "# Embedding vs Structural Similarity Report",
        "",
        f"**Date:** {today}  ",
        f"**Embeddings:** `{Path(meta.get('embeddings_path', 'N/A')).name}`  ",
        f"**Reactions sampled:** {meta.get('n_reactions_sampled', 'N/A')}  ",
        f"**Pairs evaluated:** {meta.get('n_pairs', 'N/A'):,}"
        if isinstance(meta.get("n_pairs"), int)
        else f"**Pairs evaluated:** {meta.get('n_pairs', 'N/A')}",
        "",
        "This report assesses whether the learned reaction embeddings capture "
        "chemical similarity beyond token-level syntax. For a random sample of "
        "reactions, pairwise cosine similarity in embedding space is compared "
        "against pairwise Tanimoto similarity over structural reaction fingerprints "
        "(RDKit, 4096 bits). A strong positive correlation indicates the model "
        "encodes chemistry, not just syntax.",
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
    sections += [_results_section(stats, figure_path), "", "---", ""]
    sections += [_summary_section(stats), ""]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a similarity correlation report from JSON results."
    )
    p.add_argument(
        "--results", required=True,
        help="Path to sim_corr_<JOB_ID>.json produced by similarity_correlation.py",
    )
    p.add_argument(
        "--env", default=None,
        help="Environment JSON from collect_env.py (optional; collected live if omitted)",
    )
    p.add_argument(
        "--figure", default=None,
        help="Path to the figure file to embed in the report (optional)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output Markdown file (default: same dir as --results, .md extension)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    raw = json.loads(results_path.read_text())
    meta = raw.get("meta", {})
    stats = raw.get("stats", {})

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

    report = build_report(meta, stats, env, args.figure)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()