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
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_METRICS = "results/tokenizer_metrics.json"
DEFAULT_CLASSIFIER = "results/ec_classifier.json"
DEFAULT_OUTPUT = "results/tokenizer_comparison_report.md"


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
    best = max(candidates, key=lambda x: x[1]) if higher_is_better else min(candidates, key=lambda x: x[1])
    return best[0]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _intrinsic_table(results: list[dict]) -> str:
    names = [r["name"] for r in results]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|--------|" + "--------|" * len(names)

    def row(label: str, key: str, fmt_fn=_num, higher_is_better: bool | None = None) -> str:
        cells = [fmt_fn(r.get(key)) for r in results]
        if higher_is_better is not None and len(results) > 1:
            w = _winner(results, key, higher_is_better)
            cells = [f"**{c}**" if r["name"] == w else c for c, r in zip(cells, results)]
        return "| " + label + " | " + " | ".join(cells) + " |"

    lines = [
        header, sep,
        row("Vocabulary size", "vocab_size", fmt_fn=lambda v: "rule-based" if v is None else f"{v:,}"),
        row("Samples evaluated", "n_samples", fmt_fn=lambda v: f"{v:,}"),
        row("Error rate", "error_rate", fmt_fn=_pct, higher_is_better=False),
        row("Throughput (SMARTS/s)", "throughput_smarts_per_s", fmt_fn=lambda v: _num(v, 0), higher_is_better=True),
        row("Throughput (chars/s)", "throughput_chars_per_s", fmt_fn=lambda v: _num(v, 0), higher_is_better=True),
        row("Seq length — mean", "seq_len_mean", higher_is_better=False),
        row("Seq length — median", "seq_len_median", higher_is_better=False),
        row("Seq length — std", "seq_len_std", higher_is_better=False),
        row("Seq length — min", "seq_len_min", fmt_fn=lambda v: _num(v, 0)),
        row("Seq length — max", "seq_len_max", fmt_fn=lambda v: _num(v, 0)),
        row("Fertility ratio (tokens/char)", "fertility_mean", fmt_fn=lambda v: _num(v, 4), higher_is_better=False),
        row("Unique tokens used", "unique_tokens_used", fmt_fn=lambda v: f"{v:,}"),
        row("Vocabulary utilisation", "vocab_utilisation", fmt_fn=_pct),
        row("Round-trip fidelity", "round_trip_rate", fmt_fn=_pct, higher_is_better=True),
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
            cells.append(f"{m:.4f} ± {s:.4f}" if m is not None and s is not None else "N/A")
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
        header, sep,
        row("Accuracy", "accuracy_mean", "accuracy_std"),
        row("F1 macro", "f1_macro_mean", "f1_macro_std"),
        row("F1 weighted", "f1_weighted_mean", "f1_weighted_std"),
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
        vals = {r["name"]: r.get("fertility_mean") for r in metrics if r.get("fertility_mean") is not None}
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

    return "\n".join(lines) if lines else "_Insufficient data to auto-generate summary._"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(metrics: list[dict], clf: list[dict]) -> str:
    today = date.today().isoformat()
    tokenizer_names = " vs ".join(r["name"] for r in metrics)

    sections = [
        f"# Tokenizer Comparison Report",
        f"",
        f"**Date:** {today}  ",
        f"**Tokenizers:** {tokenizer_names}",
        f"",
        f"---",
        f"",
        f"## 1. Intrinsic Metrics",
        f"",
        f"Properties measured directly from the tokenization of a shared SMARTS sample.",
        f"Bold values indicate the better result where applicable.",
        f"",
        _intrinsic_table(metrics),
        f"",
        f"---",
        f"",
        f"## 2. Downstream Task — EC Class Classification",
        f"",
        f"Each tokenizer's sequences are featurised with TF-IDF and fed into a "
        f"logistic regression classifier predicting the top-level EC enzyme class "
        f"(oxidoreductase, transferase, hydrolase, lyase, isomerase, ligase, translocase). "
        f"Higher scores indicate the tokenizer encodes more reaction-type information.",
        f"",
    ]

    if clf:
        sections += [_classifier_table(clf), f""]
    else:
        sections += [
            f"_No classifier results found. "
            f"Run `train_ec_classifier.py --output {DEFAULT_CLASSIFIER}` first._",
            f"",
        ]

    sections += [
        f"---",
        f"",
        f"## 3. Summary",
        f"",
        _summary(metrics, clf),
        f"",
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown comparison report from benchmark JSONs."
    )
    parser.add_argument("--metrics", default=DEFAULT_METRICS,
                        help=f"Intrinsic metrics JSON (default: {DEFAULT_METRICS})")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help=f"Classifier results JSON (default: {DEFAULT_CLASSIFIER})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output Markdown file (default: {DEFAULT_OUTPUT})")
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
        logger.warning("Classifier results not found at %s — Section 2 will be empty.", clf_path)

    report = build_report(metrics, clf)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    logger.info("Report written to %s", out)


if __name__ == "__main__":
    main()