"""Generate a Markdown comparison report from the two tokenizer benchmark JSONs.

Reads:
    results/tokenizer_metrics.json   (from compare_tokenizers.py)
    results/ec_classifier.json       (from train_ec_classifier.py)

Writes:
    results/tokenizer_comparison_report.md

Usage
-----
    python scripts/generate_report.py
    python scripts/generate_report.py \\
        --metrics results/tokenizer_metrics.json \\
        --classifier results/ec_classifier.json \\
        --output results/tokenizer_comparison_report.md
"""

import argparse
import importlib.util
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_METRICS = "results/tokenizer_metrics.json"
DEFAULT_CLASSIFIER = "results/ec_classifier.json"
DEFAULT_OUTPUT = "results/tokenizer_comparison_report.md"


def _load_collect_env():
    spec = importlib.util.spec_from_file_location(
        "collect_env", Path(__file__).parent / "collect_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%"


def _num(v, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{decimals}f}"


def _winner(results: list[dict], key: str, higher_is_better: bool = True) -> str:
    """Return the name of the tokenizer with the better value for *key*."""
    candidates = [(r["name"], r.get(key)) for r in results if r.get(key) is not None]
    if len(candidates) < 2:
        return ""
    best = (
        max(candidates, key=lambda x: x[1])
        if higher_is_better
        else min(candidates, key=lambda x: x[1])
    )
    return best[0]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _intrinsic_table(results: list[dict]) -> str:
    names = [r["name"] for r in results]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|--------|" + "--------|" * len(names)

    def row(
        label: str, key: str, fmt_fn=_num, higher_is_better: bool | None = None
    ) -> str:
        cells = [fmt_fn(r.get(key)) for r in results]
        if higher_is_better is not None and len(results) > 1:
            w = _winner(results, key, higher_is_better)
            cells = [
                f"**{c}**" if r["name"] == w else c for c, r in zip(cells, results)
            ]
        return "| " + label + " | " + " | ".join(cells) + " |"

    lines = [
        header,
        sep,
        row(
            "Vocabulary size",
            "vocab_size",
            fmt_fn=lambda v: "rule-based" if v is None else f"{v:,}",
        ),
        row("Samples evaluated", "n_samples", fmt_fn=lambda v: f"{v:,}"),
        row("Error rate", "error_rate", fmt_fn=_pct, higher_is_better=False),
        row(
            "Throughput (SMARTS/s)",
            "throughput_smarts_per_s",
            fmt_fn=lambda v: _num(v, 0),
            higher_is_better=True,
        ),
        row(
            "Throughput (chars/s)",
            "throughput_chars_per_s",
            fmt_fn=lambda v: _num(v, 0),
            higher_is_better=True,
        ),
        row("Seq length — mean", "seq_len_mean", higher_is_better=False),
        row("Seq length — median", "seq_len_median", higher_is_better=False),
        row("Seq length — std", "seq_len_std", higher_is_better=False),
        row("Seq length — min", "seq_len_min", fmt_fn=lambda v: _num(v, 0)),
        row("Seq length — max", "seq_len_max", fmt_fn=lambda v: _num(v, 0)),
        row(
            "Fertility ratio (tokens/char)",
            "fertility_mean",
            fmt_fn=lambda v: _num(v, 4),
            higher_is_better=False,
        ),
        row("Unique tokens used", "unique_tokens_used", fmt_fn=lambda v: f"{v:,}"),
        row("Vocabulary utilisation", "vocab_utilisation", fmt_fn=_pct),
        row(
            "Round-trip fidelity", "round_trip_rate", fmt_fn=_pct, higher_is_better=True
        ),
    ]
    return "\n".join(lines)


def _classifier_table(results: list[dict]) -> str:
    names = [r["name"] for r in results]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|--------|" + "--------|" * len(names)

    def row(label: str, mean_key: str, std_key: str) -> str:
        cells = []
        for r in results:
            m = r.get(mean_key)
            s = r.get(std_key)
            cells.append(
                f"{m:.4f} ± {s:.4f}" if m is not None and s is not None else "N/A"
            )
        w = _winner(results, mean_key, higher_is_better=True)
        cells = [f"**{c}**" if r["name"] == w else c for c, r in zip(cells, results)]
        return "| " + label + " | " + " | ".join(cells) + " |"

    meta = results[0]
    n_folds = meta.get("n_folds", "?")
    n_classes = meta.get("n_classes", "?")
    n_samples = meta.get("n_samples", "?")

    lines = [
        f"Evaluated on **{n_samples:}** labelled SMARTS, "
        f"**{n_classes}** EC classes, "
        f"**{n_folds}**-fold stratified cross-validation.\n",
        header,
        sep,
        row("Accuracy", "accuracy_mean", "accuracy_std"),
        row("F1 macro", "f1_macro_mean", "f1_macro_std"),
        row("F1 weighted", "f1_weighted_mean", "f1_weighted_std"),
    ]
    return "\n".join(lines)


_EC_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def _per_class_table(clf: list[dict]) -> str:
    """Markdown table of per-class P/R/F1/support for each tokenizer."""
    if not clf:
        return ""

    # Gather all EC classes across tokenizers
    all_classes = sorted({cls for r in clf for cls in (r.get("per_class") or {})})
    if not all_classes:
        return ""

    # Collect support counts (first result wins)
    support: dict[str, int] = {}
    for r in clf:
        for cls, stats in (r.get("per_class") or {}).items():
            support.setdefault(cls, int(stats.get("support", 0)))

    mean_support = sum(support.values()) / len(support) if support else 0
    rare_threshold = max(50, mean_support * 0.01)

    # Header
    name_cols = " | ".join(
        f"P ({r['name']}) | R ({r['name']}) | F1 ({r['name']})" for r in clf
    )
    header = f"| EC Class | Name | Support | {name_cols} |"
    n_metric_cols = 3 * len(clf)
    sep = "|----------|------|---------|" + "---------|" * n_metric_cols

    rows = [header, sep]
    for cls in all_classes:
        n = support.get(cls, 0)
        rare_marker = " ★" if n < rare_threshold else ""
        name = _EC_NAMES.get(cls, "")
        cols = f"| EC {cls}{rare_marker} | {name} | {n:,} |"
        for r in clf:
            pc = (r.get("per_class") or {}).get(cls, {})
            p = pc.get("precision", 0.0)
            rec = pc.get("recall", 0.0)
            f1 = pc.get("f1", 0.0)
            cols += f" {p:.3f} | {rec:.3f} | {f1:.3f} |"
        rows.append(cols)

    lines = [
        "### Per-class breakdown",
        "",
        "\n".join(rows),
        "",
    ]
    if any(support.get(cls, 0) < rare_threshold for cls in all_classes):
        lines += [
            "> **★ Rare class** — fewer samples than 1 % of class mean or < 50 examples. "
            "Low scores for these classes reflect data scarcity, not a model failure. "
            "EC 7 (Translocases) is heavily under-represented in RetroRules.",
            "",
        ]
    return "\n".join(lines)


def _summary(metrics: list[dict], clf: list[dict]) -> str:
    lines: list[str] = []

    # Throughput winner
    w_speed = _winner(metrics, "throughput_smarts_per_s", higher_is_better=True)
    if w_speed:
        vals = {r["name"]: r.get("throughput_smarts_per_s", 0) for r in metrics}
        names = list(vals.keys())
        if len(names) == 2:
            ratio = vals[names[0]] / vals[names[1]] if vals[names[1]] else float("inf")
            slower = names[1] if ratio > 1 else names[0]
            faster = names[0] if ratio > 1 else names[1]
            ratio = max(ratio, 1 / ratio)
            lines.append(
                f"- **{faster}** is **{ratio:.1f}×** faster than {slower} "
                f"({vals[faster]:,.0f} vs {vals[slower]:,.0f} SMARTS/s)."
            )

    # Fertility winner
    w_fert = _winner(metrics, "fertility_mean", higher_is_better=False)
    if w_fert:
        vals = {
            r["name"]: r.get("fertility_mean")
            for r in metrics
            if r.get("fertility_mean") is not None
        }
        if len(vals) == 2:
            names = list(vals.keys())
            lines.append(
                f"- **{w_fert}** produces shorter sequences "
                f"(fertility {vals[w_fert]:.4f} vs "
                f"{vals[[n for n in names if n != w_fert][0]]:.4f} tokens/char)."
            )

    # Round-trip
    for r in metrics:
        rt = r.get("round_trip_rate")
        if rt is not None and rt < 1.0:
            lines.append(
                f"- **{r['name']}** round-trip fidelity is {rt * 100:.2f}% "
                "(some SMARTS are not exactly reconstructed from token IDs)."
            )

    # Downstream task
    if clf and len(clf) >= 2:
        w_acc = _winner(clf, "accuracy_mean", higher_is_better=True)
        vals = {r["name"]: r.get("accuracy_mean", 0) for r in clf}
        names = list(vals.keys())
        loser = [n for n in names if n != w_acc][0]
        lines.append(
            f"- On the EC classification task, **{w_acc}** achieves higher accuracy "
            f"({vals[w_acc]:.4f} vs {vals[loser]:.4f}), suggesting its token sequences "
            f"carry more reaction-type signal."
        )

    return (
        "\n".join(lines) if lines else "_Insufficient data to auto-generate summary._"
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _env_section(env: dict) -> str:
    def _o(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "N/A"

    lines = [
        "## 0. Experimental Environment",
        "",
        "**Hardware**",
        "",
        "| Component | Details |",
        "|-----------|---------|",
        f"| CPU | {_o(env.get('cpu_model'))} |",
        f"| CPU Cores | {_o(env.get('cpu_cores'))} |",
        f"| RAM | {_o(env.get('ram_gb'), ' GB')} |",
        f"| OS | {_o(env.get('platform'))} |",
        "",
        "**Software**",
        "",
        "| Package | Version |",
        "|---------|---------|",
        f"| Python | {_o(env.get('python_version'))} |",
        f"| PyTorch | {_o(env.get('torch_version'))} |",
        f"| NumPy | {_o(env.get('pkg_numpy'))} |",
        f"| pandas | {_o(env.get('pkg_pandas'))} |",
        f"| scikit-learn | {_o(env.get('pkg_scikit_learn'))} |",
        f"| SentencePiece | {_o(env.get('pkg_sentencepiece'))} |",
        f"| RDKit | {_o(env.get('pkg_rdkit'))} |",
    ]
    return "\n".join(lines)


def build_report(
    metrics: list[dict],
    clf: list[dict],
    env: dict | None = None,
    figures_dir: str | None = None,
) -> str:
    today = date.today().isoformat()
    tokenizer_names = " vs ".join(r["name"] for r in metrics)

    def _fig(name: str) -> str:
        if not figures_dir:
            return ""
        p = Path(figures_dir) / name
        # find whichever format exists
        for ext in ("pdf", "png", "svg"):
            candidate = p.with_suffix(f".{ext}")
            if candidate.exists():
                return f"\n![{name}]({candidate})\n"
        return f"\n*Figure: {name} (not found at {figures_dir})*\n"

    sections = [
        "# Tokenizer Comparison Report",
        "",
        f"**Date:** {today}  ",
        f"**Tokenizers:** {tokenizer_names}",
        "",
        "---",
        "",
    ]

    if env:
        sections += [_env_section(env), "", "---", ""]

    sections += [
        "## 1. Intrinsic Metrics",
        "",
        "Properties measured directly from the tokenization of a shared SMARTS sample.",
        "Bold values indicate the better result where applicable.",
        "",
        _intrinsic_table(metrics),
        _fig("token_length_distribution"),
        _fig("throughput_fertility"),
        "",
        "---",
        "",
        "## 2. Downstream Task — EC Class Classification",
        "",
        "Each tokenizer's sequences are featurised with TF-IDF and fed into a "
        "logistic regression classifier predicting the top-level EC enzyme class "
        "(oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase, translocase). "
        "Higher scores indicate the tokenizer encodes more reaction-type information.",
        "",
    ]

    if clf:
        sections += [
            _classifier_table(clf),
            _fig("ec_classifier"),
            "",
            _per_class_table(clf),
        ]
    else:
        sections += [
            f"_No classifier results found. "
            f"Run `train_ec_classifier.py --output {DEFAULT_CLASSIFIER}` first._",
            "",
        ]

    sections += [
        "---",
        "",
        "## 3. Summary",
        "",
        _summary(metrics, clf),
        "",
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown comparison report from benchmark JSONs."
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"Intrinsic metrics JSON (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER,
        help=f"Classifier results JSON (default: {DEFAULT_CLASSIFIER})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--env", default=None, help="Environment JSON from collect_env.py (optional)"
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory containing generated figures (optional)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    metrics_path = Path(args.metrics)
    clf_path = Path(args.classifier)

    if not metrics_path.exists():
        logger.error("Metrics file not found: %s", metrics_path)
        raise SystemExit(1)

    metrics = _load(str(metrics_path))
    clf = _load(str(clf_path)) if clf_path.exists() else []

    if not clf:
        logger.warning(
            "Classifier results not found at %s — Section 2 will be empty.", clf_path
        )

    env = None
    if args.env and Path(args.env).exists():
        env = json.loads(Path(args.env).read_text())
        logger.info("Loaded environment from %s", args.env)
    elif args.env is None:
        logger.info("No --env provided; collecting environment from current machine...")
        collect_env = _load_collect_env()
        env = collect_env()

    report = build_report(metrics, clf, env=env, figures_dir=args.figures_dir)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    logger.info("Report written to %s", out)


if __name__ == "__main__":
    main()
