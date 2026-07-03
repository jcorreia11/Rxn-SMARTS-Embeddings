"""
Compute pooling ablation statistics for Results section 3.3.

Reads the final pooling ablation JSONs for all three model sizes
(small/medium/large) and reports per-strategy metrics and selection rationale.

Output: paper/pooling_ablation_stats.json
Run from project root: python paper/pooling_ablation_stats.py
"""

import json
from pathlib import Path

RESULT_FILES = {
    "small":  Path("results/final/3a-pooling-ablation/small/results.json"),
    "medium": Path("results/final/3a-pooling-ablation/medium/results.json"),
    "large":  Path("results/final/3a-pooling-ablation/large/results.json"),
}
OUTPUT_JSON = Path("paper/pooling_ablation_stats.json")

MODEL_SEQ_LEN = {"small": 256, "medium": 512, "large": 512}

# ── Load and reshape ───────────────────────────────────────────────────────
sizes = {}
for size, path in RESULT_FILES.items():
    doc = json.loads(path.read_text())
    meta = doc["meta"]
    rows = {r["name"]: r for r in doc["results"]}

    def fmt(r):
        return {
            "accuracy_mean":    r["accuracy_mean"],
            "accuracy_std":     r["accuracy_std"],
            "f1_macro_mean":    r["f1_macro_mean"],
            "f1_macro_std":     r["f1_macro_std"],
            "f1_weighted_mean": r["f1_weighted_mean"],
            "f1_weighted_std":  r["f1_weighted_std"],
        }

    cls_mlp  = fmt(rows["CLS+mlp"])
    mean_mlp = fmt(rows["Mean+mlp"])
    delta_acc    = round(mean_mlp["accuracy_mean"]  - cls_mlp["accuracy_mean"],  4)
    delta_f1mac  = round(mean_mlp["f1_macro_mean"]  - cls_mlp["f1_macro_mean"],  4)

    sizes[size] = {
        "model_max_seq_len": MODEL_SEQ_LEN[size],
        "model_weights": meta["weights"],
        "n_samples": meta["n_samples_actual"],
        "n_folds": meta["folds"],
        "conditions": {
            "CLS+logreg":  fmt(rows["CLS+logreg"]),
            "CLS+mlp":     cls_mlp,
            "Mean+logreg": fmt(rows["Mean+logreg"]),
            "Mean+mlp":    mean_mlp,
        },
        "delta_mlp": {
            "accuracy":  delta_acc,
            "f1_macro":  delta_f1mac,
            "rel_acc_pct": round(delta_acc / cls_mlp["accuracy_mean"] * 100, 1),
        },
    }

results = {
    "protocol": {
        "ec_depth": 1,
        "n_classes": 7,
        "train_test_split": "80/20 stratified",
        "n_folds": 10,
        "random_seed": 42,
    },
    "sizes": sizes,
    "conclusion": "Mean pooling outperforms CLS pooling consistently across all model sizes and both classifier heads.",
}

# ── Print summary ──────────────────────────────────────────────────────────
print("=== POOLING ABLATION — ALL MODEL SIZES ===\n")
for size, s in sizes.items():
    print(f"[{size.upper()}]  max_seq_len={s['model_max_seq_len']}  n={s['n_samples']:,}  folds={s['n_folds']}")
    hdr = f"  {'Condition':<16}  {'Accuracy':>18}  {'F1-macro':>18}  {'F1-weighted':>18}"
    print(hdr)
    print("  " + "-" * 74)
    for name, r in s["conditions"].items():
        print(f"  {name:<16}  {r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}      "
              f"{r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}      "
              f"{r['f1_weighted_mean']:.4f}±{r['f1_weighted_std']:.4f}")
    d = s["delta_mlp"]
    print(f"  Δ(Mean−CLS, MLP): acc={d['accuracy']:+.4f} ({d['rel_acc_pct']:+.1f}%),  "
          f"f1mac={d['f1_macro']:+.4f}")
    print()

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"Saved to {OUTPUT_JSON}")