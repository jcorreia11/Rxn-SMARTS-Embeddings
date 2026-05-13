"""
Compute MLM / span-masking statistics for Methods section 2.4.

Reads vocab.json and the final training run config; reports vocabulary
composition, masking budget statistics, and final pretraining metrics.

Output: paper/mlm_stats.json
Run from project root: python paper/mlm_stats.py
"""

import sys
sys.path.insert(0, ".")

import json
import math
from pathlib import Path

OUTPUT_JSON = Path("paper/mlm_stats.json")

# ── Inputs ─────────────────────────────────────────────────────────────────
vocab = json.load(open("data/processed/vocab.json"))
run   = json.load(open("models/smarts_transformer_20260428_160310.json"))
tok2id = vocab["token_to_id"]

# ── MLM hyperparameters (from collator defaults / training config) ──────────
MASK_PROB  = run["training_config"]["mask_prob"]   # 0.25
MEAN_SPAN  = 3.0
MAX_SPAN   = 10
P_STOP     = 1.0 / MEAN_SPAN                       # geometric stop probability
WARMUP_STEPS = run["training_config"]["warmup_steps"]
BATCH_SIZE   = run["training_config"]["batch_size"]
N_EPOCHS     = run["training_config"]["num_epochs"]
N_TRAIN      = 325_576

SPECIAL_TOKENS    = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"}
STRUCTURAL_TOKENS = {"(", ")", ".", ">>", ">"}

# ── Vocabulary composition ─────────────────────────────────────────────────
special_toks    = [k for k in tok2id if k in SPECIAL_TOKENS]
structural_toks = [k for k in tok2id if k in STRUCTURAL_TOKENS]
content_toks    = [k for k in tok2id if k not in SPECIAL_TOKENS
                   and k not in STRUCTURAL_TOKENS]

# ── Geometric span-length distribution (truncated at max_span) ─────────────
def geom_pmf(k, p):
    if k < MAX_SPAN:
        return (1 - p) ** (k - 1) * p
    else:  # absorb remaining mass at cap
        return (1 - p) ** (k - 1)

span_distribution = {k: geom_pmf(k, P_STOP) for k in range(1, MAX_SPAN + 1)}
expected_span_len = sum(k * v for k, v in span_distribution.items())

# ── Masking budget for representative sequence lengths ──────────────────────
budget_stats = {}
for seq_len, label in [(101, "median"), (120, "mean"), (256, "max_len_256"),
                        (512, "max_len_512")]:
    budget = max(1, round(MASK_PROB * seq_len))
    budget_stats[label] = {
        "seq_len": seq_len,
        "n_masked_tokens": budget,
        "pct_masked": round(budget / seq_len * 100, 1),
        "expected_n_spans": round(budget / expected_span_len, 1),
    }

# ── Training schedule ───────────────────────────────────────────────────────
steps_per_epoch = math.ceil(N_TRAIN / BATCH_SIZE)
total_steps = steps_per_epoch * N_EPOCHS

# ── Final pretraining metrics ───────────────────────────────────────────────
h = run["history"]
final = {
    "train_loss":        round(h["train_losses"][-1], 4),
    "val_loss":          round(h["val_losses"][-1], 4),
    "val_top1_acc":      round(h["val_top1_accuracy"][-1] * 100, 2),
    "val_top5_acc":      round(h["val_top5_accuracy"][-1] * 100, 2),
    "content_top1_acc":  round(h["val_content_top1_accuracy"][-1] * 100, 2),
    "content_top5_acc":  round(h["val_content_top5_accuracy"][-1] * 100, 2),
    "training_time_h":   round(h["training_time_s"] / 3600, 2),
}

# ── Assemble results ───────────────────────────────────────────────────────
results = {
    "masking_hyperparameters": {
        "mask_prob": MASK_PROB,
        "mean_span": MEAN_SPAN,
        "max_span": MAX_SPAN,
        "p_stop_geometric": round(P_STOP, 4),
        "substitution_rule": "80/10/10 at span level",
        "span_start_sampling": "uniform from unmasked real positions",
    },
    "vocabulary_composition": {
        "total_static": vocab["size"],
        "total_with_mask": vocab["size"] + 1,
        "n_special_tokens": len(special_toks),
        "special_tokens": sorted(special_toks),
        "n_structural_tokens": len(structural_toks),
        "structural_tokens": sorted(structural_toks),
        "n_content_tokens": len(content_toks),
        "pct_content": round(len(content_toks) / vocab["size"] * 100, 1),
    },
    "span_length_distribution": {
        str(k): round(v, 4) for k, v in span_distribution.items()
    },
    "expected_span_length": round(expected_span_len, 3),
    "masking_budget_by_seq_len": budget_stats,
    "lr_schedule": {
        "type": "linear warmup then cosine annealing",
        "warmup_steps": WARMUP_STEPS,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_pct_of_training": round(WARMUP_STEPS / total_steps * 100, 2),
        "note": "Template incorrectly states 'constant' after warmup; actual is cosine decay.",
    },
    "final_pretraining_metrics": final,
}

# ── Print summary ──────────────────────────────────────────────────────────
m = results["masking_hyperparameters"]
v = results["vocabulary_composition"]
b = results["masking_budget_by_seq_len"]
lr = results["lr_schedule"]

print("=== MLM / SPAN MASKING ===")
print(f"Mask probability     : {m['mask_prob']} ({m['mask_prob']*100:.0f}% of real tokens)")
print(f"Span distribution    : Geometric(p={m['p_stop_geometric']}), mean={m['mean_span']}, cap={m['max_span']}")
print(f"Expected span length : {results['expected_span_length']:.3f} tokens")
print(f"Substitution rule    : {m['substitution_rule']}")
print()
print("=== VOCABULARY COMPOSITION ===")
print(f"Total (static)       : {v['total_static']:,}")
print(f"  Special tokens     : {v['n_special_tokens']}  {v['special_tokens']}")
print(f"  Structural tokens  : {v['n_structural_tokens']}  {v['structural_tokens']}")
print(f"  Content tokens     : {v['n_content_tokens']:,}  ({v['pct_content']}% of vocab)")
print()
print("=== MASKING BUDGET ===")
for label, s in b.items():
    print(f"  {label:14s} (L={s['seq_len']:3d}): {s['n_masked_tokens']:3d} tokens masked "
          f"({s['pct_masked']:.1f}%), ~{s['expected_n_spans']:.1f} spans")
print()
print("=== LR SCHEDULE ===")
print(f"Type                 : {lr['type']}")
print(f"Warmup steps         : {lr['warmup_steps']:,}  ({lr['warmup_pct_of_training']:.2f}% of training)")
print(f"Total steps          : {lr['total_steps']:,}  ({lr['steps_per_epoch']:,}/epoch)")
print(f"NOTE: {lr['note']}")
print()
print("=== FINAL PRETRAINING METRICS (epoch 100) ===")
for k, v in final.items():
    unit = " h" if "time" in k else (" %" if "acc" in k else "")
    print(f"  {k:22s}: {v}{unit}")

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")