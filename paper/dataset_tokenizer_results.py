"""
Compile Dataset and Tokenization result statistics for Results section 3.1.

Sources:
  - data/raw/retrorules-v3.0-{metanetx,rhea}.csv  — provenance counts
  - paper/tokenizer_stats.json                      — intrinsic metrics
  - results/tokenizer_comparison_<RUN_ID>/ec_classifier.json — extrinsic EC
    results; auto-discovers the lexicographically-latest (i.e. most recent,
    since RUN_ID is a YYYYMMDD_HHMMSS timestamp) tokenizer_comparison_*
    directory rather than a hardcoded RUN_ID, so this keeps working across
    reruns of compare_tokenizers.sbatch without edits.

Output: paper/dataset_tokenizer_results.json
Run from project root: python paper/dataset_tokenizer_results.py
"""

import json
from pathlib import Path

import pandas as pd

OUTPUT_JSON = Path("paper/dataset_tokenizer_results.json")

# ── 1. Dataset provenance ─────────────────────────────────────────────────
metanetx = pd.read_csv("data/raw/retrorules-v3.0-metanetx.csv", usecols=["TEMPLATE", "VALID"])
rhea     = pd.read_csv("data/raw/retrorules-v3.0-rhea.csv",     usecols=["TEMPLATE", "VALID"])

n_metanetx = (metanetx["VALID"].astype(str).str.upper() == "TRUE").sum()
n_rhea     = (rhea["VALID"].astype(str).str.upper() == "TRUE").sum()

combined = pd.concat([metanetx, rhea]).drop_duplicates(subset="TEMPLATE")
n_combined = len(combined)

val = pd.read_csv("data/processed/validated_smarts.csv")
n_final = (val["valid"].astype(str).str.upper() == "TRUE").sum()

# EC label coverage. Resolve each template's ECS across *all* raw
# occurrences before deduping — a template appearing in both MetaNetX and
# Rhea can have the annotation on only one duplicate row, so
# deduplicate-then-check silently drops it (the 212,952 vs 214,007
# discrepancy documented in PAPER_TEMPLATE_JCIM.md §2.1).
raw_all = pd.concat([
    pd.read_csv("data/raw/retrorules-v3.0-metanetx.csv", usecols=["TEMPLATE", "ECS", "VALID"]),
    pd.read_csv("data/raw/retrorules-v3.0-rhea.csv",     usecols=["TEMPLATE", "ECS", "VALID"]),
])
raw_all = raw_all[raw_all["VALID"].astype(str).str.upper() == "TRUE"]
has_ecs_mask = raw_all["ECS"].notna() & (raw_all["ECS"].str.strip() != "")
annotated_templates = set(raw_all.loc[has_ecs_mask, "TEMPLATE"])
n_with_ec = val["smarts"].isin(annotated_templates).sum()

dataset = {
    "n_metanetx_valid": int(n_metanetx),
    "n_rhea_valid": int(n_rhea),
    "n_combined_before_dedup": int(n_metanetx + n_rhea),
    "n_after_dedup": int(n_combined),
    "n_final_corpus": int(n_final),
    "n_duplicates_removed": int(n_metanetx + n_rhea - n_combined),
    "n_with_ec_annotation": int(n_with_ec),
    "pct_with_ec": round(int(n_with_ec) / int(n_final) * 100, 1),
}

# ── 2. Intrinsic tokenizer metrics ────────────────────────────────────────
tok = json.loads(Path("paper/tokenizer_stats.json").read_text())
rb  = tok["rule_based"]
sp  = tok["sentencepiece"]
cmp = tok["comparison"]

intrinsic = {
    "rule_based": {
        "vocab_size": rb["vocab_size"],
        "seq_len_median": rb["seq_len_median"],
        "seq_len_mean": round(rb["seq_len_mean"], 1),
        "seq_len_max": rb["seq_len_max"],
        "fertility_mean": round(rb["fertility_mean"], 3),
        "coverage_256": rb["coverage_pct"]["max_len_256"],
        "coverage_512": rb["coverage_pct"]["max_len_512"],
        "n_truncated_512": rb["n_truncated_at_512"],
        "throughput_smarts_per_s": rb["throughput_smarts_per_s"],
        "round_trip_fidelity_pct": rb["round_trip_fidelity_pct"],
        "error_rate_pct": rb["error_rate_pct"],
    },
    "sentencepiece": {
        "vocab_size": sp["vocab_size"],
        "seq_len_median": sp["seq_len_median"],
        "seq_len_mean": round(sp["seq_len_mean"], 1),
        "seq_len_max": sp["seq_len_max"],
        "fertility_mean": round(sp["fertility_mean"], 3),
        "coverage_256": sp["coverage_pct"]["max_len_256"],
        "coverage_512": sp["coverage_pct"]["max_len_512"],
        "throughput_smarts_per_s": sp["throughput_smarts_per_s"],
        "round_trip_fidelity_pct": sp["round_trip_fidelity_pct"],
        "error_rate_pct": sp["error_rate_pct"],
    },
    "deltas": {
        "median_len_ratio_rb_vs_sp": round(rb["seq_len_median"] / sp["seq_len_median"], 3),
        "throughput_speedup_rb_vs_sp": cmp["throughput_speedup_rb_vs_sp"],
        "coverage_gain_rb_at_256_pp": round(rb["coverage_pct"]["max_len_256"] - sp["coverage_pct"]["max_len_256"], 1),
        "coverage_gain_rb_at_512_pp": round(rb["coverage_pct"]["max_len_512"] - sp["coverage_pct"]["max_len_512"], 1),
    },
}

# ── 3. Extrinsic tokenizer metrics (TF-IDF + LR, EC depth=1) ─────────────
_tok_comparison_dirs = sorted(Path("results").glob("tokenizer_comparison_*"))
if not _tok_comparison_dirs:
    raise FileNotFoundError(
        "No results/tokenizer_comparison_* directory found — has "
        "compare_tokenizers.sbatch been run?"
    )
ec_raw = json.loads((_tok_comparison_dirs[-1] / "ec_classifier.json").read_text())
ec_meta = ec_raw["meta"]
ec_results = {r["name"]: r for r in ec_raw["results"]}

rb_ec = ec_results["SmartsTokenizer"]
sp_ec = ec_results["SentencePieceTokenizer"]

extrinsic = {
    "n_samples": ec_meta["n_samples_actual"],
    "n_train": rb_ec["n_train"],
    "n_test": rb_ec["n_test"],
    "n_classes": rb_ec["n_classes"],
    "n_folds": rb_ec["n_folds"],
    "rule_based": {
        "accuracy_mean": rb_ec["accuracy_mean"],
        "accuracy_std": rb_ec["accuracy_std"],
        "f1_macro_mean": rb_ec["f1_macro_mean"],
        "f1_macro_std": rb_ec["f1_macro_std"],
        "f1_weighted_mean": rb_ec["f1_weighted_mean"],
        "f1_weighted_std": rb_ec["f1_weighted_std"],
    },
    "sentencepiece": {
        "accuracy_mean": sp_ec["accuracy_mean"],
        "accuracy_std": sp_ec["accuracy_std"],
        "f1_macro_mean": sp_ec["f1_macro_mean"],
        "f1_macro_std": sp_ec["f1_macro_std"],
        "f1_weighted_mean": sp_ec["f1_weighted_mean"],
        "f1_weighted_std": sp_ec["f1_weighted_std"],
    },
    "delta_accuracy_sp_minus_rb": round(sp_ec["accuracy_mean"] - rb_ec["accuracy_mean"], 4),
    "delta_f1_macro_sp_minus_rb": round(sp_ec["f1_macro_mean"] - rb_ec["f1_macro_mean"], 4),
}

# ── Assemble & save ────────────────────────────────────────────────────────
results = {
    "dataset": dataset,
    "intrinsic_tokenizer": intrinsic,
    "extrinsic_tokenizer_ec": extrinsic,
}

# ── Print summary ──────────────────────────────────────────────────────────
d = dataset
print("=== DATASET ===")
print(f"MetaNetX valid      : {d['n_metanetx_valid']:,}")
print(f"Rhea valid          : {d['n_rhea_valid']:,}")
print(f"Combined before dedup: {d['n_combined_before_dedup']:,}")
print(f"Duplicates removed  : {d['n_duplicates_removed']:,}")
print(f"Final corpus        : {d['n_final_corpus']:,}")
print(f"With EC annotation  : {d['n_with_ec_annotation']:,}  ({d['pct_with_ec']}%)")

print()
rb = intrinsic["rule_based"]
sp = intrinsic["sentencepiece"]
dl = intrinsic["deltas"]
print("=== INTRINSIC TOKENIZER METRICS ===")
print(f"{'Metric':<35} {'Rule-based':>14} {'SentencePiece':>14}")
print("-" * 65)
print(f"{'Vocabulary size':<35} {rb['vocab_size']:>14,} {sp['vocab_size']:>14,}")
print(f"{'Median seq len (tokens)':<35} {rb['seq_len_median']:>14} {sp['seq_len_median']:>14}")
print(f"{'Mean seq len (tokens)':<35} {rb['seq_len_mean']:>14.1f} {sp['seq_len_mean']:>14.1f}")
print(f"{'Max seq len (tokens)':<35} {rb['seq_len_max']:>14} {sp['seq_len_max']:>14}")
print(f"{'Fertility (tokens/char)':<35} {rb['fertility_mean']:>14.3f} {sp['fertility_mean']:>14.3f}")
print(f"{'Coverage @ 256 (%)':<35} {rb['coverage_256']:>14} {sp['coverage_256']:>14}")
print(f"{'Coverage @ 512 (%)':<35} {rb['coverage_512']:>14} {sp['coverage_512']:>14}")
print(f"{'Throughput (SMARTS/s)':<35} {rb['throughput_smarts_per_s']:>14,.0f} {sp['throughput_smarts_per_s']:>14,.0f}")
print(f"{'Round-trip fidelity (%)':<35} {rb['round_trip_fidelity_pct']:>14} {sp['round_trip_fidelity_pct']:>14}")
print(f"\nDeltas: median_ratio={dl['median_len_ratio_rb_vs_sp']:.3f}, "
      f"speedup={dl['throughput_speedup_rb_vs_sp']}×, "
      f"Δcoverage@256={dl['coverage_gain_rb_at_256_pp']:+.1f}pp, "
      f"Δcoverage@512={dl['coverage_gain_rb_at_512_pp']:+.1f}pp")

print()
ex = extrinsic
rb_e = ex["rule_based"]
sp_e = ex["sentencepiece"]
print("=== EXTRINSIC: TF-IDF + LR EC depth=1 ===")
print(f"N={ex['n_samples']:,}  train={ex['n_train']:,}  test={ex['n_test']:,}  "
      f"classes={ex['n_classes']}  folds={ex['n_folds']}")
print(f"{'Metric':<20} {'Rule-based':>18} {'SentencePiece':>18}")
print("-" * 58)
print(f"{'Accuracy':<20} {rb_e['accuracy_mean']:.4f}±{rb_e['accuracy_std']:.4f}   "
      f"{sp_e['accuracy_mean']:.4f}±{sp_e['accuracy_std']:.4f}")
print(f"{'F1-macro':<20} {rb_e['f1_macro_mean']:.4f}±{rb_e['f1_macro_std']:.4f}   "
      f"{sp_e['f1_macro_mean']:.4f}±{sp_e['f1_macro_std']:.4f}")
print(f"{'F1-weighted':<20} {rb_e['f1_weighted_mean']:.4f}±{rb_e['f1_weighted_std']:.4f}   "
      f"{sp_e['f1_weighted_mean']:.4f}±{sp_e['f1_weighted_std']:.4f}")
print(f"\nΔ accuracy (SP − RB): {ex['delta_accuracy_sp_minus_rb']:+.4f}  "
      f"Δ F1-macro: {ex['delta_f1_macro_sp_minus_rb']:+.4f}")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")