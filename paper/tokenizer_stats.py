"""Compute tokenizer statistics for Methods section 2.2.

Reads pre-computed tokenizer metrics from the comparison run and the vocab
files; recomputes coverage thresholds from the stored per-sequence lengths.
Must be run from the project root.

Output: paper/tokenizer_stats.json
"""
import json
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
VOCAB_JSON         = Path("data/processed/vocab.json")
SP_VOCAB_FILE      = Path("data/processed/sp_tokenizer.vocab")
METRICS_JSON       = Path(
    "results/tokenizer_comparison_20260409_131636/tokenizer_metrics.json"
)
OUTPUT_JSON = Path("paper/tokenizer_stats.json")

# ── Load inputs ────────────────────────────────────────────────────────────
vocab = json.loads(VOCAB_JSON.read_text())
sp_vocab_size = sum(1 for _ in SP_VOCAB_FILE.open())

metrics = json.loads(METRICS_JSON.read_text())
rb_m  = next(m for m in metrics if m["name"] == "SmartsTokenizer")
sp_m  = next(m for m in metrics if m["name"] == "SentencePieceTokenizer")

rb_lens = np.array(rb_m["seq_len_values"], dtype=np.int32)
sp_lens = np.array(sp_m["seq_len_values"], dtype=np.int32)

n = len(rb_lens)
assert n == len(sp_lens), "Sample counts differ between tokenizers"

# ── Coverage at max-length thresholds ─────────────────────────────────────
def coverage(lengths: np.ndarray, max_len: int) -> float:
    return round(float((lengths <= max_len).mean() * 100), 1)

# ── Assemble results ───────────────────────────────────────────────────────
results = {
    "n_samples": n,
    "rule_based": {
        "vocab_size": vocab["size"],
        "n_chemical_tokens": vocab["size"] - 4,  # excludes [PAD],[UNK],[BOS],[EOS]
        "special_tokens": ["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        "mask_token_injected_at_training_only": True,
        "vocab_size_during_training": vocab["size"] + 1,  # +[MASK]
        "error_rate_pct": round(rb_m["error_rate"] * 100, 2),
        "round_trip_fidelity_pct": round(rb_m["round_trip_rate"] * 100, 1),
        "throughput_smarts_per_s": round(rb_m["throughput_smarts_per_s"], 0),
        "throughput_chars_per_s": round(rb_m["throughput_chars_per_s"], 0),
        "seq_len_mean": round(rb_m["seq_len_mean"], 2),
        "seq_len_median": int(rb_m["seq_len_median"]),
        "seq_len_std": round(rb_m["seq_len_std"], 2),
        "seq_len_min": int(rb_m["seq_len_min"]),
        "seq_len_max": int(rb_m["seq_len_max"]),
        "fertility_mean": round(rb_m["fertility_mean"], 4),
        "unique_tokens_used": rb_m["unique_tokens_used"],
        "coverage_pct": {
            "max_len_128": coverage(rb_lens, 128),
            "max_len_256": coverage(rb_lens, 256),
            "max_len_512": coverage(rb_lens, 512),
        },
        "n_truncated_at_512": int((rb_lens > 512).sum()),
    },
    "sentencepiece": {
        "model_type": "bpe",
        "vocab_size": sp_vocab_size,
        "character_coverage": 1.0,
        "normalization": "identity",
        "error_rate_pct": round(sp_m["error_rate"] * 100, 2),
        "round_trip_fidelity_pct": round(sp_m["round_trip_rate"] * 100, 1),
        "vocab_utilisation_pct": round(sp_m["vocab_utilisation"] * 100, 2),
        "throughput_smarts_per_s": round(sp_m["throughput_smarts_per_s"], 0),
        "throughput_chars_per_s": round(sp_m["throughput_chars_per_s"], 0),
        "seq_len_mean": round(sp_m["seq_len_mean"], 2),
        "seq_len_median": int(sp_m["seq_len_median"]),
        "seq_len_std": round(sp_m["seq_len_std"], 2),
        "seq_len_min": int(sp_m["seq_len_min"]),
        "seq_len_max": int(sp_m["seq_len_max"]),
        "fertility_mean": round(sp_m["fertility_mean"], 4),
        "unique_tokens_used": sp_m["unique_tokens_used"],
        "coverage_pct": {
            "max_len_128": coverage(sp_lens, 128),
            "max_len_256": coverage(sp_lens, 256),
            "max_len_512": coverage(sp_lens, 512),
        },
    },
    "comparison": {
        "throughput_speedup_rb_vs_sp": round(
            rb_m["throughput_smarts_per_s"] / sp_m["throughput_smarts_per_s"], 1
        ),
        "fertility_ratio_rb_vs_sp": round(
            rb_m["fertility_mean"] / sp_m["fertility_mean"], 3
        ),
        "median_len_ratio_rb_vs_sp": round(
            rb_m["seq_len_median"] / sp_m["seq_len_median"], 3
        ),
        "coverage_gain_rb_over_sp_at_256": round(
            coverage(rb_lens, 256) - coverage(sp_lens, 256), 1
        ),
        "coverage_gain_rb_over_sp_at_512": round(
            coverage(rb_lens, 512) - coverage(sp_lens, 512), 1
        ),
    },
}

# ── Print summary ──────────────────────────────────────────────────────────
rb = results["rule_based"]
sp = results["sentencepiece"]
cmp = results["comparison"]

print("=== RULE-BASED SMARTS TOKENIZER ===")
print(f"Vocab size (static)    : {rb['vocab_size']:,}  (+1 [MASK] at training → {rb['vocab_size_during_training']:,})")
print(f"Chemical tokens        : {rb['n_chemical_tokens']:,}")
print(f"Special tokens         : {rb['special_tokens']}")
print(f"Error rate             : {rb['error_rate_pct']:.2f}%")
print(f"Round-trip fidelity    : {rb['round_trip_fidelity_pct']:.1f}%")
print(f"Throughput             : {rb['throughput_smarts_per_s']:,.0f} SMARTS/s")
print(f"Seq len (median/mean)  : {rb['seq_len_median']} / {rb['seq_len_mean']:.1f}")
print(f"Seq len max            : {rb['seq_len_max']}")
print(f"Fertility              : {rb['fertility_mean']:.4f} tokens/char")
print(f"Coverage ≤128 tokens   : {rb['coverage_pct']['max_len_128']:.1f}%")
print(f"Coverage ≤256 tokens   : {rb['coverage_pct']['max_len_256']:.1f}%")
print(f"Coverage ≤512 tokens   : {rb['coverage_pct']['max_len_512']:.1f}%")
print(f"Truncated at 512       : {rb['n_truncated_at_512']:,}")

print("\n=== SENTENCEPIECE (BPE) TOKENIZER ===")
print(f"Vocab size             : {sp['vocab_size']:,}")
print(f"Vocab utilisation      : {sp['vocab_utilisation_pct']:.1f}%")
print(f"Error rate             : {sp['error_rate_pct']:.2f}%")
print(f"Round-trip fidelity    : {sp['round_trip_fidelity_pct']:.1f}%")
print(f"Throughput             : {sp['throughput_smarts_per_s']:,.0f} SMARTS/s")
print(f"Seq len (median/mean)  : {sp['seq_len_median']} / {sp['seq_len_mean']:.1f}")
print(f"Seq len max            : {sp['seq_len_max']}")
print(f"Fertility              : {sp['fertility_mean']:.4f} tokens/char")
print(f"Coverage ≤128 tokens   : {sp['coverage_pct']['max_len_128']:.1f}%")
print(f"Coverage ≤256 tokens   : {sp['coverage_pct']['max_len_256']:.1f}%")
print(f"Coverage ≤512 tokens   : {sp['coverage_pct']['max_len_512']:.1f}%")

print("\n=== COMPARISON ===")
print(f"Throughput speedup (RB/SP)      : {cmp['throughput_speedup_rb_vs_sp']:.1f}×")
print(f"Fertility ratio (RB/SP)         : {cmp['fertility_ratio_rb_vs_sp']:.3f}")
print(f"Median length ratio (RB/SP)     : {cmp['median_len_ratio_rb_vs_sp']:.3f}")
print(f"Coverage gain at ≤256 (RB−SP)   : +{cmp['coverage_gain_rb_over_sp_at_256']:.1f} p.p.")
print(f"Coverage gain at ≤512 (RB−SP)   : +{cmp['coverage_gain_rb_over_sp_at_512']:.1f} p.p.")

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")