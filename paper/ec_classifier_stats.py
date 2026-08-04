"""
Compute EC classifier dataset and protocol statistics for Methods section 2.7.1.

Reads the final EC classifier result JSONs for all three model sizes (small,
medium, large) at all three EC depths, cross-checks that dataset/protocol
metadata (sample counts, class counts, split settings) agree across sizes —
they should, since the labelled dataset and protocol don't depend on which
embedding model is used — and reports the shared values.

Reads from the results/final/ mirror using stable, generic filenames (no
run-ID in the path), so this keeps working across retraining runs without
edits.

Also aggregates pretraining seed variance: medium_seed1/medium_seed2 are
independent Phase 1 reruns (different SEED, everything else identical),
carried through Phase 2/3b on their own. This complements the random-init
baseline comparison in similarity_correlation_stats.py — that shows the
pretrained model beats random projections; this shows the result isn't a
fluke of one training run, by reporting EC-classifier accuracy/F1 stability
across independent pretraining seeds.

Output: paper/ec_classifier_stats.json
Run from project root: python paper/ec_classifier_stats.py
"""

import json
import statistics
from pathlib import Path

SIZES = ["small", "medium", "large"]
DEPTHS = [1, 2, 3]
SEED_VARIANTS = ["medium", "medium_seed1", "medium_seed2"]
HEADLINE_METHOD = "Pretrained+mlp"
RESULT_ROOT = Path("results/final/3b-ec-classifier")
OUTPUT_JSON = Path("paper/ec_classifier_stats.json")


def _load(size: str, depth: int) -> dict:
    path = RESULT_ROOT / size / f"depth{depth}" / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — has train_ec_classifier.sbatch been run for "
            f"MODEL_SIZE={size}, EC_DEPTH={depth}?"
        )
    return json.loads(path.read_text())


# ── Load all (size, depth) combinations and cross-check dataset metadata ──
per_size_depth = {}
for size in SIZES:
    for depth in DEPTHS:
        doc = _load(size, depth)
        m = doc["meta"]
        per_size_depth[(size, depth)] = {
            "n_samples": m["n_samples_actual"],
            "n_classes": len(m["label_names"]),
            "train_test_split": m["train_test_split"],
            "n_folds": m["folds"],
            "random_seed": m["random_seed"],
            "methods": [r["name"] for r in doc["results"]],
        }

depths: dict[int, dict] = {}
for depth in DEPTHS:
    values = [per_size_depth[(size, depth)] for size in SIZES]
    ref = values[0]
    for key in ("n_samples", "n_classes", "train_test_split", "n_folds", "random_seed"):
        mismatched = [(SIZES[i], v[key]) for i, v in enumerate(values) if v[key] != ref[key]]
        assert not mismatched, (
            f"depth={depth} '{key}' differs across model sizes (expected identical "
            f"dataset/protocol regardless of embedding model): {ref[key]!r} vs {mismatched}"
        )
    depths[depth] = ref

# ── Shared protocol (consistent across all depths) ────────────────────────
ref = depths[1]
assert all(
    depths[d]["train_test_split"] == ref["train_test_split"] for d in depths
), "train_test_split mismatch across depths"
assert all(
    depths[d]["n_folds"] == ref["n_folds"] for d in depths
), "n_folds mismatch across depths"

results = {
    "task": "EC number classification at depth=1, 2, 3",
    "source_data": [
        "data/raw/retrorules-v3.0-metanetx.csv",
        "data/raw/retrorules-v3.0-rhea.csv",
    ],
    "label_construction": (
        "EC annotations joined with validated SMARTS via TEMPLATE column; "
        "first EC number used; truncated to k components for depth k; "
        "rare classes pruned to ensure stratified splitting feasibility."
    ),
    "cross_checked_across_sizes": SIZES,
    "depths": {
        str(d): {
            "n_samples": v["n_samples"],
            "n_classes": v["n_classes"],
        }
        for d, v in depths.items()
    },
    "protocol": {
        "train_test_split": ref["train_test_split"],
        "n_cv_folds": ref["n_folds"],
        "random_seed": ref["random_seed"],
        "embedding_normalisation": "StandardScaler (zero mean, unit variance) per feature",
        "cv_metric": "stratified k-fold on training set; final report on held-out test set",
    },
    "methods": {
        "tfidf_smarts":    {"type": "TF-IDF + logistic regression", "tokenizer": "SmartsTokenizer (§2.2.1)"},
        "tfidf_sp":        {"type": "TF-IDF + logistic regression", "tokenizer": "SentencePieceTokenizer (§2.2.2)"},
        "random_logreg":   {"type": "Random-init embeddings + logistic regression", "note": "untrained control"},
        "random_mlp":      {"type": "Random-init embeddings + MLP",               "note": "untrained control"},
        "pretrained_logreg": {"type": "Pretrained embeddings + logistic regression", "note": "proposed method"},
        "pretrained_mlp":    {"type": "Pretrained embeddings + MLP",               "note": "proposed method"},
    },
    "classifier_details": {
        "tfidf": {
            "vectoriser": "TfidfVectorizer",
            "sublinear_tf": True,
            "lowercase": False,
            "token_pattern": "custom (tokenizer function)",
        },
        "logistic_regression": {
            "solver": "lbfgs",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
        },
        "mlp": {
            "hidden_layer_sizes": [256, 128],
            "activation": "relu",
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
    },
    "metrics": ["accuracy", "F1-macro", "F1-weighted"],
}


def _headline_metrics(doc: dict) -> dict:
    for r in doc["results"]:
        if r["name"] == HEADLINE_METHOD:
            return {
                "accuracy_mean": r["accuracy_mean"],
                "f1_macro_mean": r["f1_macro_mean"],
                "f1_weighted_mean": r["f1_weighted_mean"],
            }
    raise KeyError(f"method '{HEADLINE_METHOD}' not found in results")


# ── Pretraining seed variance (medium model, 3 independent seeds) ─────────
seed_variant_docs = {
    (variant, depth): _load(variant, depth)
    for variant in SEED_VARIANTS
    for depth in DEPTHS
}

results["pretraining_seed_variance"] = {
    "model_size": "medium",
    "method": HEADLINE_METHOD,
    "seeds": SEED_VARIANTS,
    "note": (
        "medium_seed1/medium_seed2 are independent Phase 1 pretraining runs "
        "with different SEED values (weight init + data-loader shuffling/"
        "masking), carried through Phase 2 (embeddings) and Phase 3b (this "
        "classifier) independently of the headline medium run (seed=42)."
    ),
    "depths": {},
}

for depth in DEPTHS:
    per_seed = {
        variant: _headline_metrics(seed_variant_docs[(variant, depth)])
        for variant in SEED_VARIANTS
    }
    accs = [v["accuracy_mean"] for v in per_seed.values()]
    f1s = [v["f1_macro_mean"] for v in per_seed.values()]
    results["pretraining_seed_variance"]["depths"][str(depth)] = {
        "accuracy_mean": round(statistics.mean(accs), 6),
        "accuracy_std": round(statistics.stdev(accs), 6),
        "f1_macro_mean": round(statistics.mean(f1s), 6),
        "f1_macro_std": round(statistics.stdev(f1s), 6),
        "per_seed": per_seed,
    }

# ── Print summary ──────────────────────────────────────────────────────────
print("=== EC CLASSIFIER — DATASET & PROTOCOL ===")
print(f"Task            : {results['task']}")
print(f"Cross-checked across: {', '.join(SIZES)}")
print(f"Split           : {results['protocol']['train_test_split']}, "
      f"{results['protocol']['n_cv_folds']}-fold CV, seed={results['protocol']['random_seed']}")
print()
print(f"{'Depth':>6}  {'Samples':>10}  {'Classes':>8}")
print("-" * 30)
for d, v in results["depths"].items():
    print(f"{d:>6}  {v['n_samples']:>10,}  {v['n_classes']:>8,}")

print()
print("=== METHODS ===")
for name, m in results["methods"].items():
    note = f"  [{m.get('note', '')}]" if m.get('note') else ""
    print(f"  {name:<20} — {m['type']}{note}")

print()
print("=== METRICS ===")
print(f"  {', '.join(results['metrics'])}")

print()
print(f"=== PRETRAINING SEED VARIANCE (medium, {HEADLINE_METHOD}, {len(SEED_VARIANTS)} seeds) ===")
print(f"{'Depth':>6} {'Accuracy (mean±std)':>21} {'F1-macro (mean±std)':>21}")
print("-" * 52)
for depth in DEPTHS:
    v = results["pretraining_seed_variance"]["depths"][str(depth)]
    print(
        f"{depth:>6} "
        f"{v['accuracy_mean']:>10.4f} ± {v['accuracy_std']:<8.4f} "
        f"{v['f1_macro_mean']:>10.4f} ± {v['f1_macro_std']:<8.4f}"
    )

# ── Save ───────────────────────────────────────────────────────────────────
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {OUTPUT_JSON}")