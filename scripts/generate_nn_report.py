"""Generate a self-contained nearest-neighbor analysis report.

Reads the JSON produced by ``nearest_neighbors.py`` (which contains a
``meta`` block, a ``query`` block, and a ``neighbors`` list) and an optional
environment JSON from ``collect_env.py``, and writes a publication-quality
Markdown report.

The report covers:
    1. Experimental environment (hardware + software)
    2. Experimental setup (embeddings, query, search parameters)
    3. Results table (top-K neighbors with scores and EC labels)
    4. Summary (EC class agreement)

Usage
-----
    python scripts/generate_nn_report.py \\
        --results results/nn_results.json \\
        --env     results/nn_results_env.json \\
        --output  results/nn_results_report.md
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
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{seconds:.2f} s"


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
    ]

    if env.get("cuda_available"):
        lines += [
            f"| GPU | {_opt(env.get('gpu_name'))} |",
            f"| GPU Memory | {_opt(env.get('gpu_memory_gb'), ' GB')} |",
            f"| GPU Count | {_opt(env.get('gpu_count'))} |",
            f"| CUDA Version | {_opt(env.get('cuda_version'))} |",
            f"| Driver Version | {_opt(env.get('driver_version'))} |",
        ]
    else:
        lines.append("| GPU | None (CPU-only run) |")

    lines += [
        "",
        "### Software",
        "",
        "| Package | Version |",
        "|---------|---------|",
        f"| Python | {_opt(env.get('python_version'))} |",
        f"| NumPy | {_opt(env.get('pkg_numpy'))} |",
        f"| pandas | {_opt(env.get('pkg_pandas'))} |",
        f"| PyTorch | {_opt(env.get('torch_version'))} |",
        f"| scikit-learn | {_opt(env.get('pkg_scikit_learn'))} |",
        f"| RDKit | {_opt(env.get('pkg_rdkit'))} |",
    ]
    return "\n".join(lines)


def _setup_section(meta: dict, query: dict) -> str:
    metric = meta.get("metric", "N/A")
    score_label = "Cosine similarity (higher = closer)" if metric == "cosine" else "Euclidean distance (lower = closer)"

    lines = [
        "## 2. Experimental Setup",
        "",
        "### Embedding Index",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Source | RetroRules v3.0 |",
        f"| Embeddings file | `{Path(meta.get('embeddings_path', 'N/A')).name}` |",
        f"| SMARTS file | `{Path(meta.get('smarts_path', 'N/A')).name}` |",
        f"| Total reactions indexed | {meta.get('n_reactions', 'N/A'):,} |"
        if isinstance(meta.get("n_reactions"), int)
        else f"| Total reactions indexed | {meta.get('n_reactions', 'N/A')} |",
        f"| Embedding dimension | {meta.get('embedding_dim', 'N/A')} |",
        f"| Data load time | {_fmt_time(meta.get('load_time_s'))} |",
        "",
        "### Search Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Distance metric | {metric} |",
        f"| Score interpretation | {score_label} |",
        f"| Neighbors retrieved (K) | {meta.get('top_k', 'N/A')} |",
        f"| Search time | {_fmt_time(meta.get('search_time_s'))} |",
        "",
        "### Query Reaction",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Index | {query.get('index', 'N/A')} |",
        f"| EC class | EC {query.get('ec_class', '?')} — {query.get('ec_name', 'Unknown')} |",
        f"| Full ECS | `{query.get('ecs', 'N/A') or 'N/A'}` |",
        "",
        f"**Query SMARTS:** `{query.get('smarts', 'N/A')}`",
    ]
    return "\n".join(lines)


def _results_section(query: dict, neighbors: list[dict], summary: dict, metric: str) -> str:
    if not neighbors:
        return "## 3. Results\n\n_No neighbors found._"

    score_col = "Cosine Sim." if metric == "cosine" else "Euclidean Dist."
    query_cls = query.get("ec_class", "?")

    header = f"| Rank | {score_col} | EC Class | EC Name | Same EC? | Index | SMARTS |"
    sep =     "|------|------------|----------|---------|----------|-------|--------|"

    rows = [header, sep]
    for n in neighbors:
        smarts = n["smarts"]
        smarts_cell = smarts if len(smarts) <= 60 else smarts[:57] + "..."
        same = "**Yes**" if n["same_ec_class"] else "No"
        rows.append(
            f"| {n['rank']} "
            f"| {n['score']:+.4f} "
            f"| EC {n['ec_class']} "
            f"| {n['ec_name']} "
            f"| {same} "
            f"| {n['index']} "
            f"| `{smarts_cell}` |"
        )

    n_same = summary.get("n_same_ec_class", 0)
    frac = summary.get("fraction_same_ec_class", 0.0)
    k = len(neighbors)

    lines = [
        "## 3. Results",
        "",
        f"Query reaction is EC {query_cls} ({query.get('ec_name', 'Unknown')}). "
        f"Neighbors are ranked by {score_col.lower()}.",
        "",
        "\n".join(rows),
    ]
    return "\n".join(lines)


def _summary_section(query: dict, neighbors: list[dict], summary: dict) -> str:
    query_cls = query.get("ec_class", "?")
    n_same = summary.get("n_same_ec_class", 0)
    frac = summary.get("fraction_same_ec_class", 0.0)
    k = len(neighbors)

    lines = ["## 4. Summary", ""]

    if query_cls == "?":
        lines.append(
            "The query reaction has no EC label in the RetroRules database. "
            "EC class agreement cannot be assessed."
        )
    else:
        query_name = _EC_NAMES.get(query_cls, "Unknown")
        lines.append(
            f"- **Query class**: EC {query_cls} ({query_name})."
        )
        lines.append(
            f"- **EC class agreement**: {n_same} / {k} neighbors ({frac * 100:.0f}%) "
            f"share the query's EC class."
        )
        if frac >= 0.8:
            lines.append(
                "- The embedding space places reactions of the same enzyme class in "
                "close proximity, demonstrating that the learned representations "
                "capture reaction-type information."
            )
        elif frac >= 0.4:
            lines.append(
                "- A moderate fraction of neighbors share the query's EC class, "
                "suggesting partial capture of reaction-type information in the "
                "embedding space."
            )
        else:
            lines.append(
                "- Few neighbors share the query's EC class. This may reflect a "
                "rare or boundary reaction, or that the embedding space organises "
                "reactions by structural features that cut across EC boundaries."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    meta: dict,
    query: dict,
    neighbors: list[dict],
    summary: dict,
    env: dict | None,
) -> str:
    today = date.today().isoformat()
    metric = meta.get("metric", "cosine")
    k = meta.get("top_k", len(neighbors))

    sections = [
        "# Nearest-Neighbor Analysis Report",
        "",
        f"**Date:** {today}  ",
        f"**Embeddings:** `{Path(meta.get('embeddings_path', 'N/A')).name}`  ",
        f"**Query index:** {query.get('index', 'N/A')}  ",
        f"**K:** {k}  ",
        f"**Metric:** {metric}",
        "",
        "This report presents the top-K nearest neighbors retrieved from the "
        "reaction SMARTS embedding space for a given query reaction. "
        "EC class labels from RetroRules v3.0 are used to assess whether "
        "neighboring reactions are chemically related.",
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

    sections += [_setup_section(meta, query), "", "---", ""]
    sections += [_results_section(query, neighbors, summary, metric), "", "---", ""]
    sections += [_summary_section(query, neighbors, summary), ""]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a nearest-neighbor analysis report from JSON results."
    )
    p.add_argument(
        "--results",
        required=True,
        help="Path to nn_results.json produced by nearest_neighbors.py",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    raw = json.loads(results_path.read_text())

    meta = raw.get("meta", {})
    query = raw.get("query", {})
    neighbors = raw.get("neighbors", [])
    summary = raw.get("summary", {})

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

    report = build_report(meta, query, neighbors, summary, env)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()