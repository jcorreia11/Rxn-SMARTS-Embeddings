"""CLS vs mean-pooling ablation on the EC classification task.

Loads the pretrained transformer once, extracts embeddings with both CLS and
mean pooling in a single forward pass, then evaluates logistic regression and
MLP classifiers for each strategy.  Results are written to a JSON file
compatible with ``generate_pooling_report.py``.

Usage
-----
    python scripts/ablation_pooling.py \\
        --weights models/smarts_transformer.pt \\
        --config  models/smarts_transformer.json \\
        --vocab   data/processed/vocab.json \\
        --output  results/pooling_ablation.json
"""

import argparse
import json
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smart_rxn_embeddings.evaluation.ec_labels import load_labelled_smarts
from smart_rxn_embeddings.evaluation.splitting import stratified_group_holdout_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAW_FILES = [
    "data/raw/retrorules-v3.0-metanetx.csv",
    "data/raw/retrorules-v3.0-rhea.csv",
]
VALIDATED_FILE = "data/processed/validated_smarts.csv"
DEFAULT_N_SAMPLES = 10_000
DEFAULT_EC_DEPTH = 1
DEFAULT_FOLDS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading — see smart_rxn_embeddings.evaluation.ec_labels for
# load_labelled_smarts() / primary_ec(), shared with train_ec_classifier.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Embedding extraction  (both strategies in one forward pass)
# ---------------------------------------------------------------------------


def extract_both_poolings(
    smarts_list: list[str],
    weights_path: str,
    config_path: str,
    vocab_path: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (cls_embeddings, mean_embeddings, elapsed_s).

    Loads the model once and runs each batch through the encoder a single time,
    deriving both CLS and mean-pooled representations without redundant computation.
    """
    import torch
    from smart_rxn_embeddings.models.smarts_transformer import (
        SmartsMLMModel,
        TransformerConfig,
    )
    from smart_rxn_embeddings.models.embed import SmartsEmbedder

    cfg = json.loads(Path(config_path).read_text())
    model_config = TransformerConfig.from_dict(cfg["model_config"])
    model = SmartsMLMModel(model_config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    logger.info(
        "Loaded pretrained weights from %s (d_model=%d, layers=%d)",
        weights_path,
        model_config.d_model,
        model_config.num_encoder_layers,
    )

    vocab = json.loads(Path(vocab_path).read_text())
    token_to_id: dict[str, int] = vocab["token_to_id"]

    # Extract with CLS pooling, then mean pooling, reusing the same model object.
    # SmartsEmbedder is lightweight — the model weights are shared.
    t0 = time.perf_counter()
    cls_embedder = SmartsEmbedder(model, token_to_id, pooling="cls")
    cls_emb = cls_embedder.embed(smarts_list, batch_size=batch_size)
    logger.info("CLS embeddings extracted: %s", cls_emb.shape)

    mean_embedder = SmartsEmbedder(model, token_to_id, pooling="mean")
    mean_emb = mean_embedder.embed(smarts_list, batch_size=batch_size)
    logger.info("Mean embeddings extracted: %s", mean_emb.shape)
    elapsed = time.perf_counter() - t0

    return cls_emb, mean_emb, elapsed


# ---------------------------------------------------------------------------
# Classifier evaluation  (mirrors train_ec_classifier.py)
# ---------------------------------------------------------------------------


def _build_classifier(head: str) -> object:
    if head == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=300,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )


def evaluate_embeddings(
    name: str,
    X_emb_train: np.ndarray,
    X_emb_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    n_folds: int,
    label_names: list[str],
    head: str,
) -> dict:
    """The ``StandardScaler`` is fit inside the cross-validated pipeline (not
    once on the full training split beforehand) so no fold's held-out
    portion leaks into the scaling statistics used to transform it."""

    def _make_pipeline() -> Pipeline:
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", _build_classifier(head))]
        )

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    t0_cv = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            _make_pipeline(),
            X_emb_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=1,
        )
    cv_time_s = time.perf_counter() - t0_cv

    pipeline = _make_pipeline()
    t0_fit = time.perf_counter()
    pipeline.fit(X_emb_train, y_train)
    fit_time_s = time.perf_counter() - t0_fit
    y_pred = pipeline.predict(X_emb_test)

    # Group-aware splitting can move a whole rare class's group(s) to one
    # side of the split, so y_test may not contain every label — pass the
    # labels actually present rather than assuming all of label_names appear.
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [label_names[int(lbl)] for lbl in present_labels]
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    logger.info(
        "\n%s — %d-fold CV (train=%d, test=%d, head=%s):\n"
        "  accuracy : %.4f ± %.4f\n"
        "  f1_macro : %.4f ± %.4f\n%s",
        name,
        n_folds,
        len(X_emb_train),
        len(X_emb_test),
        head,
        scores["test_accuracy"].mean(),
        scores["test_accuracy"].std(),
        scores["test_f1_macro"].mean(),
        scores["test_f1_macro"].std(),
        report_str,
    )

    return {
        "name": name,
        "pooling": name.split("+")[0].lower(),
        "head": head,
        "n_train": len(X_emb_train),
        "n_test": len(X_emb_test),
        "n_classes": int(np.unique(y_train).size),
        "n_folds": n_folds,
        "n_samples": len(X_emb_train) + len(X_emb_test),
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
        "f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
        "f1_weighted_mean": round(float(scores["test_f1_weighted"].mean()), 4),
        "f1_weighted_std": round(float(scores["test_f1_weighted"].std()), 4),
        "cv_time_s": round(cv_time_s, 2),
        "fit_time_s": round(fit_time_s, 2),
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in label_names
            if cls in report_dict
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLS vs mean-pooling ablation on the EC classification task."
    )
    p.add_argument("--weights", required=True, help="Pretrained model weights (.pt)")
    p.add_argument("--config", required=True, help="Model config (.json)")
    p.add_argument("--vocab", default="data/processed/vocab.json", help="vocab.json")
    p.add_argument("--validated", default=VALIDATED_FILE, help="Validated SMARTS CSV")
    p.add_argument("--raw", nargs="+", default=RAW_FILES, help="Raw RetroRules CSVs")
    p.add_argument(
        "--ec-depth",
        type=int,
        default=DEFAULT_EC_DEPTH,
        choices=[1, 2, 3],
        help="EC label depth (default: 1)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Max SMARTS to use, 0=all (default: 10000)",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="CV folds (default: 5)",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64)",
    )
    p.add_argument(
        "--output",
        default="results/pooling_ablation.json",
        help="Output JSON path (default: results/pooling_ablation.json)",
    )
    p.add_argument(
        "--split-strategy",
        default="group",
        choices=["group", "random"],
        help=(
            "'group' (default) keeps RetroRules radius-siblings together, "
            "avoiding train/test leakage. 'random' reproduces the old "
            "plain-stratified split, kept only for before/after comparison."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0_total = time.perf_counter()

    n_samples = args.n_samples if args.n_samples > 0 else None

    logger.info("Loading labelled SMARTS (EC depth=%d)...", args.ec_depth)
    df, n_available = load_labelled_smarts(
        args.raw, args.validated, args.ec_depth, n_samples, RANDOM_SEED
    )
    logger.info(
        "%d labelled templates available (post rare-class filter) — using %d",
        n_available,
        len(df),
    )

    X = df["smarts"].tolist()
    groups = df["group"].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    label_names = list(le.classes_)
    logger.info("Labels: %s", label_names)

    indices = np.arange(len(X))
    if args.split_strategy == "group":
        train_idx, test_idx = stratified_group_holdout_split(
            y, groups, test_size=0.2, seed=RANDOM_SEED
        )
    else:
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED
        )
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    X_train_smarts = [X[i] for i in train_idx]
    X_test_smarts = [X[i] for i in test_idx]
    logger.info(
        "Train: %d | Test: %d (80/20, split_strategy=%s)",
        len(X_train_smarts),
        len(X_test_smarts),
        args.split_strategy,
    )

    # --- Extract embeddings (model loaded once) ---
    logger.info("\n=== Extracting CLS and mean embeddings (single model load) ===")
    cls_emb, mean_emb, embed_time_s = extract_both_poolings(
        X, args.weights, args.config, args.vocab, args.embed_batch_size
    )
    logger.info("Total embedding extraction time: %.1f s", embed_time_s)

    # --- Evaluate all four combinations ---
    results: list[dict] = []
    for pooling, emb in [("CLS", cls_emb), ("Mean", mean_emb)]:
        emb_train = emb[train_idx]
        emb_test = emb[test_idx]
        for head in ("logreg", "mlp"):
            name = f"{pooling}+{head}"
            logger.info("\n=== %s ===", name)
            results.append(
                evaluate_embeddings(
                    name,
                    emb_train,
                    emb_test,
                    y_train,
                    y_test,
                    groups_train,
                    args.folds,
                    label_names,
                    head,
                )
            )

    total_time_s = time.perf_counter() - t0_total
    logger.info("\nTotal ablation time: %.1f s", total_time_s)

    # --- Save ---
    cfg_raw = json.loads(Path(args.config).read_text())
    output_doc = {
        "meta": {
            "run_id": Path(args.output).stem,
            "date": date.today().isoformat(),
            "ablation": "cls_vs_mean_pooling",
            "weights": args.weights,
            "config": args.config,
            "ec_depth": args.ec_depth,
            "n_samples_requested": args.n_samples if args.n_samples > 0 else "all",
            "n_available": n_available,
            "n_samples_actual": len(X),
            "split_strategy": args.split_strategy,
            "train_test_split": f"80/20 ({args.split_strategy}-aware, stratified)",
            "random_seed": RANDOM_SEED,
            "folds": args.folds,
            "label_names": label_names,
            "embed_time_s": round(embed_time_s, 2),
            "total_time_s": round(total_time_s, 2),
            "model_config": cfg_raw.get("model_config", {}),
        },
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output_doc, indent=2))
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
