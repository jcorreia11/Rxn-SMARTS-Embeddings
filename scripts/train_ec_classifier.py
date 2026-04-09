"""EC-class classification to compare tokenizer representations.

Each tokenizer is used to featurize reaction SMARTS into a bag-of-tokens TF-IDF
matrix.  A logistic regression is trained on top and evaluated with stratified
k-fold cross-validation.  The downstream classification score quantifies how
much chemically useful information each tokenizer packs into its token sequence.

EC top-level classes (labels):
    1 = Oxidoreductases   2 = Transferases    3 = Hydrolases
    4 = Lyases            5 = Isomerases      6 = Ligases
    7 = Translocases

Pipeline per tokenizer:
    SMARTS → tokenize → join with spaces → TF-IDF → LogisticRegression → CV

Embedding-based pipelines (when --weights/--config/--vocab are provided):
    SMARTS → SmartsEmbedder (pretrained)  → LogisticRegression / MLP → CV
    SMARTS → SmartsEmbedder (random init) → LogisticRegression / MLP → CV

Usage
-----
    python scripts/train_ec_classifier.py                      # defaults
    python scripts/train_ec_classifier.py --ec-depth 2        # e.g. "1.14" labels
    python scripts/train_ec_classifier.py --n-samples 20000 --folds 5
    python scripts/train_ec_classifier.py --output results/ec_clf.json
    python scripts/train_ec_classifier.py \\
        --weights models/smarts_transformer.pt \\
        --config  models/smarts_transformer.json \\
        --vocab   data/processed/vocab.json
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smart_rxn_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer
from smart_rxn_embeddings.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)

logger = logging.getLogger(__name__)

RAW_FILES = [
    "data/raw/retrorules-v3.0-metanetx.csv",
    "data/raw/retrorules-v3.0-rhea.csv",
]
VALIDATED_FILE = "data/processed/validated_smarts.csv"
DEFAULT_SP_MODEL = "data/processed/sp_tokenizer.model"
DEFAULT_SP_VOCAB_SIZE = 1000
DEFAULT_N_SAMPLES = 10_000
DEFAULT_EC_DEPTH = 1  # 1 = top-level class, 2 = subclass, 3 = sub-subclass
DEFAULT_FOLDS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _primary_ec(ecs_field: str, depth: int) -> str | None:
    """Return the first EC number truncated to *depth* levels, or None."""
    if not isinstance(ecs_field, str) or not ecs_field.strip():
        return None
    first = ecs_field.split(";")[0].strip()
    if not first:
        return None
    parts = first.split(".")
    return ".".join(parts[:depth])


def load_labelled_smarts(
    raw_files: list[str],
    validated_file: str,
    ec_depth: int,
    n_samples: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Return a DataFrame with columns [smarts, label]."""
    # --- EC labels from raw files ---
    frames = []
    for p in raw_files:
        df = pd.read_csv(p, usecols=["TEMPLATE", "ECS", "VALID"])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["VALID"].astype(str).str.upper() == "TRUE"]
    raw["label"] = raw["ECS"].apply(lambda x: _primary_ec(x, ec_depth))
    raw = raw.dropna(subset=["label"])
    raw = raw.rename(columns={"TEMPLATE": "smarts"})
    raw = raw.drop_duplicates(subset="smarts")
    ec_map = raw.set_index("smarts")["label"].to_dict()
    logger.info("EC map built: %d labelled SMARTS", len(ec_map))

    # --- Intersection with validated SMARTS ---
    val = pd.read_csv(validated_file)
    val = val[val["valid"].astype(str).str.upper() == "TRUE"]
    val = val.dropna(subset=["smarts"])
    val["label"] = val["smarts"].map(ec_map)
    val = val.dropna(subset=["label"])
    logger.info("After join with validated set: %d SMARTS with EC labels", len(val))

    # --- Class balance summary ---
    counts = val["label"].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    # --- Optional sample (stratified) ---
    if n_samples and n_samples < len(val):
        total = len(val)
        sampled = [
            group.sample(
                min(len(group), max(1, int(n_samples * len(group) / total))),
                random_state=random_seed,
            )
            for _, group in val.groupby("label")
        ]
        val = pd.concat(sampled).sample(frac=1, random_state=random_seed).reset_index(drop=True)
        logger.info("Sampled %d SMARTS (stratified)", len(val))

    return val[["smarts", "label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Featurisers — wrap tokenizers as callable for TfidfVectorizer
# ---------------------------------------------------------------------------


def make_smarts_tokenizer_fn() -> callable:
    tok = SmartsTokenizer()

    def _tokenize(smarts: str) -> list[str]:
        try:
            return tok.tokenize(smarts)
        except ValueError:
            return []

    return _tokenize


def make_sp_tokenizer_fn(model_path: str) -> callable:
    tok = SentencePieceTokenizer(model_path)

    def _tokenize(smarts: str) -> list[str]:
        return tok.tokenize(smarts)

    return _tokenize


# ---------------------------------------------------------------------------
# Classifier pipeline
# ---------------------------------------------------------------------------


def build_pipeline(tokenizer_fn: callable) -> Pipeline:
    """TF-IDF + Logistic Regression pipeline using *tokenizer_fn* for tokenization."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    tokenizer=tokenizer_fn,
                    token_pattern=None,  # rely entirely on our tokenizer
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: list[str],
    y_train: np.ndarray,
    X_test: list[str],
    y_test: np.ndarray,
    n_folds: int,
    label_names: list[str],
) -> dict:
    import time

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    t0_cv = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=1,
        )
    cv_time_s = time.perf_counter() - t0_cv

    # Fit on full train split, evaluate on held-out test split
    t0_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0_fit
    y_pred = pipeline.predict(X_test)

    result = {
        "name": name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": int(np.unique(y_train).size),
        "n_folds": n_folds,
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
        "f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
        "f1_weighted_mean": round(float(scores["test_f1_weighted"].mean()), 4),
        "f1_weighted_std": round(float(scores["test_f1_weighted"].std()), 4),
        "cv_time_s": round(cv_time_s, 2),
        "fit_time_s": round(fit_time_s, 2),
    }

    logger.info(
        "\n%s — %d-fold CV results (train=%d, test=%d):\n"
        "  accuracy : %.4f ± %.4f\n"
        "  f1_macro : %.4f ± %.4f\n"
        "  f1_weighted: %.4f ± %.4f\n"
        "  cv_time  : %.1f s | fit_time: %.1f s",
        name,
        n_folds,
        len(X_train),
        len(X_test),
        result["accuracy_mean"],
        result["accuracy_std"],
        result["f1_macro_mean"],
        result["f1_macro_std"],
        result["f1_weighted_mean"],
        result["f1_weighted_std"],
        cv_time_s,
        fit_time_s,
    )

    report_str = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    logger.info("\nClassification report (held-out test set):\n%s", report_str)
    logger.info(
        "Note: classes with very few samples (e.g. EC class 7, n=%d) "
        "are expected to score 0 — insufficient data, not a model failure.",
        int((y_test == label_names.index("7")).sum()) if "7" in label_names else 0,
    )

    result["per_class"] = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall": round(report_dict[cls]["recall"], 4),
            "f1": round(report_dict[cls]["f1-score"], 4),
            "support": int(report_dict[cls]["support"]),
        }
        for cls in label_names
        if cls in report_dict
    }
    result["n_samples"] = len(X_train) + len(X_test)

    return result


# ---------------------------------------------------------------------------
# Embedding-based evaluation
# ---------------------------------------------------------------------------


def _build_embedding_classifier(head: str) -> object:
    """Return a scikit-learn classifier to place on top of embeddings."""
    if head == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    if head == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1,
        )
    raise ValueError(f"Unknown head: {head!r}. Choose 'logreg' or 'mlp'.")


def evaluate_embeddings(
    name: str,
    X_emb_train: np.ndarray,
    X_emb_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_folds: int,
    label_names: list[str],
    head: str = "logreg",
) -> dict:
    """Evaluate a classifier trained on precomputed embeddings.

    Parameters
    ----------
    name:
        Label for this experiment (used in reports).
    X_emb_train / X_emb_test:
        Precomputed embedding arrays of shape ``(N, d)``.
    head:
        ``"logreg"`` or ``"mlp"``.
    """
    import time

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_emb_train)
    X_te = scaler.transform(X_emb_test)

    clf = _build_embedding_classifier(head)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    t0_cv = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            _build_embedding_classifier(head),  # fresh instance for CV
            X_tr,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=1,
        )
    cv_time_s = time.perf_counter() - t0_cv

    t0_fit = time.perf_counter()
    clf.fit(X_tr, y_train)
    fit_time_s = time.perf_counter() - t0_fit
    y_pred = clf.predict(X_te)

    result = {
        "name": name,
        "n_train": len(X_emb_train),
        "n_test": len(X_emb_test),
        "n_classes": int(np.unique(y_train).size),
        "n_folds": n_folds,
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
        "f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
        "f1_weighted_mean": round(float(scores["test_f1_weighted"].mean()), 4),
        "f1_weighted_std": round(float(scores["test_f1_weighted"].std()), 4),
        "cv_time_s": round(cv_time_s, 2),
        "fit_time_s": round(fit_time_s, 2),
    }

    logger.info(
        "\n%s — %d-fold CV results (train=%d, test=%d, head=%s):\n"
        "  accuracy : %.4f ± %.4f\n"
        "  f1_macro : %.4f ± %.4f\n"
        "  f1_weighted: %.4f ± %.4f\n"
        "  cv_time  : %.1f s | fit_time: %.1f s",
        name,
        n_folds,
        len(X_emb_train),
        len(X_emb_test),
        head,
        result["accuracy_mean"],
        result["accuracy_std"],
        result["f1_macro_mean"],
        result["f1_macro_std"],
        result["f1_weighted_mean"],
        result["f1_weighted_std"],
        cv_time_s,
        fit_time_s,
    )

    report_str = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )
    logger.info("\nClassification report (held-out test set):\n%s", report_str)

    result["per_class"] = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall": round(report_dict[cls]["recall"], 4),
            "f1": round(report_dict[cls]["f1-score"], 4),
            "support": int(report_dict[cls]["support"]),
        }
        for cls in label_names
        if cls in report_dict
    }
    result["n_samples"] = len(X_emb_train) + len(X_emb_test)

    return result


def _extract_embeddings(
    smarts_list: list[str],
    weights_path: str,
    config_path: str,
    vocab_path: str,
    pooling: str,
    batch_size: int,
    random_init: bool = False,
) -> np.ndarray:
    """Return ``(N, d_model)`` float32 embeddings for *smarts_list*.

    Parameters
    ----------
    random_init:
        When ``True`` the model weights are **not** loaded — this produces a
        random-embedding baseline with the same architecture.
    """
    import torch

    from smart_rxn_embeddings.models.embed import SmartsEmbedder
    from smart_rxn_embeddings.models.smarts_transformer import (
        SmartsMLMModel,
        TransformerConfig,
    )

    cfg = json.loads(Path(config_path).read_text())
    model_config = TransformerConfig.from_dict(cfg["model_config"])
    model = SmartsMLMModel(model_config)

    if not random_init:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info(
            "Loaded pretrained weights from %s (d_model=%d, layers=%d)",
            weights_path,
            model_config.d_model,
            model_config.num_encoder_layers,
        )
    else:
        logger.info(
            "Using randomly initialized model (d_model=%d, layers=%d) — baseline",
            model_config.d_model,
            model_config.num_encoder_layers,
        )

    vocab = json.loads(Path(vocab_path).read_text())
    token_to_id: dict[str, int] = vocab["token_to_id"]

    import time

    embedder = SmartsEmbedder(model, token_to_id, pooling=pooling)
    t0 = time.perf_counter()
    embeddings = embedder.embed(smarts_list, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    label = "pretrained" if not random_init else "random-init"
    logger.info(
        "Embedding extraction (%s): %.1f s for %d sequences",
        label,
        elapsed,
        len(smarts_list),
    )
    return embeddings, elapsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_ROWS: list[tuple[str, str]] = [
    ("n_train", "Train samples"),
    ("n_test", "Test samples"),
    ("n_classes", "Classes (EC depth)"),
    ("n_folds", "CV folds"),
    ("accuracy_mean", "Accuracy (mean)"),
    ("accuracy_std", "Accuracy (std)"),
    ("f1_macro_mean", "F1 macro (mean)"),
    ("f1_macro_std", "F1 macro (std)"),
    ("f1_weighted_mean", "F1 weighted (mean)"),
    ("f1_weighted_std", "F1 weighted (std)"),
]


def print_report(results: list[dict]) -> None:
    col_w = 26
    name_w = 32

    header = f"{'Metric':<{col_w}}" + "".join(f"{r['name']:<{name_w}}" for r in results)
    print()
    print(header)
    print("-" * (col_w + name_w * len(results)))

    for key, label in _ROWS:
        row = f"{label:<{col_w}}"
        for r in results:
            v = r.get(key, "N/A")
            row += f"{v!s:<{name_w}}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EC classifier with each tokenizer and compare."
    )
    parser.add_argument(
        "--validated",
        default=VALIDATED_FILE,
        help=f"Validated SMARTS CSV (default: {VALIDATED_FILE})",
    )
    parser.add_argument(
        "--raw",
        nargs="+",
        default=RAW_FILES,
        help="Raw RetroRules CSV files with ECS column",
    )
    parser.add_argument(
        "--sp-model",
        default=DEFAULT_SP_MODEL,
        help=f"Path to SentencePiece .model file (default: {DEFAULT_SP_MODEL})",
    )
    parser.add_argument(
        "--ec-depth",
        type=int,
        default=DEFAULT_EC_DEPTH,
        choices=[1, 2, 3],
        help="EC label depth: 1='1', 2='1.14', 3='1.14.13' (default: 1)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Max SMARTS to use (stratified sample, 0=all, default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help=f"Number of CV folds (default: {DEFAULT_FOLDS})",
    )
    parser.add_argument(
        "--sp-vocab-size",
        type=int,
        default=DEFAULT_SP_VOCAB_SIZE,
        help=f"Vocab size if SP model needs to be trained (default: {DEFAULT_SP_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write results as JSON",
    )

    # --- Embedding-based experiments ---
    emb = parser.add_argument_group(
        "embedding experiments",
        "When all three are provided, runs pretrained- and random-embedding baselines.",
    )
    emb.add_argument(
        "--weights",
        default=None,
        help="Path to pretrained SmartsMLMModel .pt file",
    )
    emb.add_argument(
        "--config",
        default=None,
        help="Path to companion model config .json file",
    )
    emb.add_argument(
        "--vocab",
        default=None,
        help="Path to vocab.json produced by build_vocab.py",
    )
    emb.add_argument(
        "--pooling",
        default="mean",
        choices=["cls", "mean"],
        help="Pooling strategy for embedding extraction (default: mean)",
    )
    emb.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64)",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    n_samples = args.n_samples if args.n_samples > 0 else None

    logger.info("Loading labelled SMARTS (EC depth=%d)...", args.ec_depth)
    df = load_labelled_smarts(
        args.raw,
        args.validated,
        args.ec_depth,
        n_samples,
        RANDOM_SEED,
    )

    import time as _time

    t0_total = _time.perf_counter()

    X = df["smarts"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    label_names = list(le.classes_)
    logger.info("Labels: %s", label_names)

    # Stratified 80/20 train/test split — CV runs on train, report on test
    indices = np.arange(len(X))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    logger.info(
        "Train: %d samples | Test: %d samples (stratified 80/20)",
        len(X_train),
        len(X_test),
    )

    results: list[dict] = []

    # --- SmartsTokenizer ---
    logger.info("\n--- SmartsTokenizer ---")
    smarts_pipeline = build_pipeline(make_smarts_tokenizer_fn())
    results.append(
        evaluate(
            "SmartsTokenizer",
            smarts_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            args.folds,
            label_names,
        )
    )

    # --- SentencePieceTokenizer ---
    sp_model = Path(args.sp_model)
    if not sp_model.exists():
        logger.info(
            "\nSentencePiece model not found at %s — training on current dataset "
            "(vocab_size=%d)...",
            sp_model,
            args.sp_vocab_size,
        )
        sp_model.parent.mkdir(parents=True, exist_ok=True)
        SentencePieceTokenizer.train(
            X_train,
            str(sp_model.with_suffix("")),  # prefix without .model extension
            vocab_size=args.sp_vocab_size,
        )
        logger.info("SentencePiece model saved to %s", sp_model)

    logger.info("\n--- SentencePieceTokenizer (%s) ---", sp_model)
    sp_pipeline = build_pipeline(make_sp_tokenizer_fn(str(sp_model)))
    results.append(
        evaluate(
            "SentencePieceTokenizer",
            sp_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            args.folds,
            label_names,
        )
    )

    # --- Embedding-based experiments ---
    embedding_paths_provided = all([args.weights, args.config, args.vocab])
    if embedding_paths_provided:
        missing = [
            name
            for name, p in [
                ("--weights", args.weights),
                ("--config", args.config),
                ("--vocab", args.vocab),
            ]
            if not Path(p).exists()
        ]
        if missing:
            logger.error(
                "The following embedding files were not found: %s", ", ".join(missing)
            )
            raise SystemExit(1)

        logger.info(
            "\n--- Extracting pretrained embeddings (pooling=%s) ---", args.pooling
        )
        X_emb, embed_time_pretrained_s = _extract_embeddings(
            X,
            args.weights,
            args.config,
            args.vocab,
            pooling=args.pooling,
            batch_size=args.embed_batch_size,
            random_init=False,
        )
        X_emb_train = X_emb[train_idx]
        X_emb_test = X_emb[test_idx]

        logger.info("\n--- Extracting random-init embeddings (baseline) ---")
        X_emb_rand, embed_time_random_s = _extract_embeddings(
            X,
            args.weights,
            args.config,
            args.vocab,
            pooling=args.pooling,
            batch_size=args.embed_batch_size,
            random_init=True,
        )
        X_emb_rand_train = X_emb_rand[train_idx]
        X_emb_rand_test = X_emb_rand[test_idx]

        for head in ("logreg", "mlp"):
            logger.info("\n--- Pretrained embeddings + %s ---", head)
            results.append(
                evaluate_embeddings(
                    f"Pretrained+{head}",
                    X_emb_train,
                    X_emb_test,
                    y_train,
                    y_test,
                    args.folds,
                    label_names,
                    head=head,
                )
            )

            logger.info("\n--- Random embeddings + %s (baseline) ---", head)
            results.append(
                evaluate_embeddings(
                    f"Random+{head}",
                    X_emb_rand_train,
                    X_emb_rand_test,
                    y_train,
                    y_test,
                    args.folds,
                    label_names,
                    head=head,
                )
            )
    else:
        if any([args.weights, args.config, args.vocab]):
            logger.warning(
                "Provide --weights, --config, AND --vocab together to enable "
                "embedding-based experiments. Skipping."
            )

    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        from datetime import date as _date

        total_time_s = _time.perf_counter() - t0_total

        meta: dict = {
            "run_id": out.stem,
            "date": _date.today().isoformat(),
            "ec_depth": args.ec_depth,
            "n_samples_requested": args.n_samples if args.n_samples > 0 else "all",
            "n_samples_actual": len(X),
            "train_test_split": "80/20 stratified",
            "random_seed": RANDOM_SEED,
            "folds": args.folds,
            "label_names": label_names,
            "total_time_s": round(total_time_s, 2),
        }
        if embedding_paths_provided:
            # Pull model config for reproducibility
            try:
                cfg = json.loads(Path(args.config).read_text())
                meta["model_config"] = cfg.get("model_config", {})
            except Exception:
                pass
            meta["embedding_weights"] = args.weights
            meta["embedding_pooling"] = args.pooling
            meta["embed_time_pretrained_s"] = round(embed_time_pretrained_s, 2)
            meta["embed_time_random_s"] = round(embed_time_random_s, 2)

        output_doc = {"meta": meta, "results": results}
        out.write_text(json.dumps(output_doc, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()
