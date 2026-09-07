"""Compare a leaky (random-split) run against its group-split fix.

RetroRules generates several templates per underlying reaction at different
context radii; a plain (stratified) random train/test split lets a
template's near-duplicate radius-siblings leak across the split, inflating
every downstream metric. ``train_ec_classifier.py`` and
``ablation_pooling.py`` both support ``--split-strategy {group,random}`` so
the same experiment can be run both ways for an apples-to-apples comparison.

This script reads the two resulting JSON files (same experiment, same
config, differing only in ``meta.split_strategy``) and writes a single
Markdown table of metric deltas — the one artifact to cite when writing up
"how much did the leaky split inflate the numbers".

Usage
-----
    python scripts/generate_split_comparison_report.py \\
        --random results/ec_classifier_<RANDOM_RUN_ID>.json \\
        --group  results/ec_classifier_<GROUP_RUN_ID>.json \\
        --label  "EC classifier (depth=1)" \\
        --output results/ec_classifier_split_comparison.md
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_METRICS = [
    ("accuracy_mean", "accuracy_std", "Accuracy"),
    ("f1_macro_mean", "f1_macro_std", "F1-macro"),
    ("f1_weighted_mean", "f1_weighted_std", "F1-weighted"),
]


def _pct(v: float | None) -> str:
    return f"{v * 100:.2f}%" if v is not None else "N/A"


def _pp(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f} pp"


def _check_comparable(random_meta: dict, group_meta: dict) -> list[str]:
    """Return warnings for config fields that differ besides split_strategy.

    A fair before/after comparison requires everything else (samples, folds,
    EC depth, model weights) held constant — only the split logic should
    differ.
    """
    warnings = []
    for key in (
        "ec_depth",
        "n_samples_requested",
        "folds",
        "random_seed",
        "embedding_weights",
        "ablation",
    ):
        r, g = random_meta.get(key), group_meta.get(key)
        if key in random_meta or key in group_meta:
            if r != g:
                warnings.append(f"'{key}' differs: random={r!r} vs group={g!r}")
    return warnings


def build_report(
    random_doc: dict,
    group_doc: dict,
    label: str,
) -> str:
    random_meta, group_meta = random_doc["meta"], group_doc["meta"]
    random_results = {r["name"]: r for r in random_doc["results"]}
    group_results = {r["name"]: r for r in group_doc["results"]}

    if random_meta.get("split_strategy", "random") != "random":
        logger.warning(
            "--random file's meta.split_strategy is not 'random': %r",
            random_meta.get("split_strategy"),
        )
    if group_meta.get("split_strategy") != "group":
        logger.warning(
            "--group file's meta.split_strategy is not 'group': %r",
            group_meta.get("split_strategy"),
        )

    warnings = _check_comparable(random_meta, group_meta)

    names = [n for n in group_results if n in random_results]
    missing = [n for n in group_results if n not in random_results] + [
        n for n in random_results if n not in group_results
    ]

    lines = [
        f"# Split-Strategy Comparison — {label}",
        "",
        f"**Generated:** {date.today().isoformat()}",
        "",
        (
            "RetroRules templates sharing a `reaction_group` (radius-siblings "
            "of the same underlying reaction) are near-duplicates with "
            "identical EC labels. Plain random/stratified splitting lets "
            "siblings leak across train/test; `stratified_group_holdout_split` "
            "keeps each group on one side. This report quantifies the "
            "resulting metric inflation."
        ),
        "",
    ]

    if warnings:
        lines += [
            "> **Warning — runs are not directly comparable:**",
            *[f"> - {w}" for w in warnings],
            "",
        ]
    if missing:
        lines += [
            f"> Methods present in only one run (excluded below): {', '.join(missing)}",
            "",
        ]

    lines += [
        "## Setup",
        "",
        "| Field | Random-split (leaky) | Group-split (fixed) |",
        "|---|---|---|",
        f"| n_available | {random_meta.get('n_available', 'N/A')} | {group_meta.get('n_available', 'N/A')} |",
        f"| n_samples_actual | {random_meta.get('n_samples_actual', 'N/A')} | {group_meta.get('n_samples_actual', 'N/A')} |",
        f"| folds | {random_meta.get('folds', 'N/A')} | {group_meta.get('folds', 'N/A')} |",
        f"| run_id | {random_meta.get('run_id', 'N/A')} | {group_meta.get('run_id', 'N/A')} |",
        "",
        "## Metric deltas (group-split minus random-split)",
        "",
        "Negative deltas mean the leaky random split over-reported performance.",
        "",
    ]

    header = "| Method |"
    sep = "|---|"
    for _, _, label_m in _METRICS:
        header += f" Random {label_m} | Group {label_m} | Δ ({label_m}) |"
        sep += "---|---|---|"
    lines += [header, sep]

    for name in names:
        r, g = random_results[name], group_results[name]
        row = f"| {name} |"
        for mean_key, _, _ in _METRICS:
            r_val, g_val = r.get(mean_key), g.get(mean_key)
            delta = (
                (g_val - r_val) if (r_val is not None and g_val is not None) else None
            )
            row += f" {_pct(r_val)} | {_pct(g_val)} | {_pp(delta)} |"
        lines.append(row)

    lines += [
        "",
        (
            "See `results/run_log.md` for which run IDs these correspond to, "
            "and `--split-strategy` in `train_ec_classifier.py` / "
            "`ablation_pooling.py` for how the random-split run was "
            "reproduced (kept only for this comparison — do not report it "
            "as a result)."
        ),
    ]

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a random-split (leaky) run against its group-split fix."
    )
    p.add_argument(
        "--random", required=True, help="JSON produced with --split-strategy random"
    )
    p.add_argument(
        "--group", required=True, help="JSON produced with --split-strategy group"
    )
    p.add_argument(
        "--label",
        default="Experiment",
        help="Human-readable label for the report title (e.g. 'EC classifier depth=1')",
    )
    p.add_argument("--output", required=True, help="Output Markdown path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    random_doc = json.loads(Path(args.random).read_text())
    group_doc = json.loads(Path(args.group).read_text())

    report = build_report(random_doc, group_doc, args.label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
