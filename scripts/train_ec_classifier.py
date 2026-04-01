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

Usage
-----
    python scripts/train_ec_classifier.py                      # defaults
    python scripts/train_ec_classifier.py --ec-depth 2        # e.g. "1.14" labels
    python scripts/train_ec_classifier.py --n-samples 20000 --folds 5
    python scripts/train_ec_classifier.py --output results/ec_clf.json
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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
DEFAULT_EC_DEPTH = 1   # 1 = top-level class, 2 = subclass, 3 = sub-subclass
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
    logger.info(
        "After join with validated set: %d SMARTS with EC labels", len(val)
    )

    # --- Class balance summary ---
    counts = val["label"].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    # --- Optional sample (stratified) ---
    if n_samples and n_samples < len(val):
        val = val.groupby("label", group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), max(1, int(n_samples * len(g) / len(val)))),
                random_state=random_seed,
            )
        )
        val = val.sample(frac=1, random_state=random_seed).reset_index(drop=True)
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
                    token_pattern=None,   # rely entirely on our tokenizer
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="multinomial",
                    C=1.0,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
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
    X: list[str],
    y: np.ndarray,
    n_folds: int,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=1,   # pipeline is already parallelised internally
        )

    # Full-dataset fit for the classification report
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    result = {
        "name": name,
        "n_samples": len(X),
        "n_classes": int(np.unique(y).size),
        "n_folds": n_folds,
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 4),
        "f1_macro_std": round(float(scores["test_f1_macro"].std()), 4),
        "f1_weighted_mean": round(float(scores["test_f1_weighted"].mean()), 4),
        "f1_weighted_std": round(float(scores["test_f1_weighted"].std()), 4),
    }

    logger.info(
        "\n%s — %d-fold CV results:\n"
        "  accuracy : %.4f ± %.4f\n"
        "  f1_macro : %.4f ± %.4f\n"
        "  f1_weighted: %.4f ± %.4f",
        name,
        n_folds,
        result["accuracy_mean"],
        result["accuracy_std"],
        result["f1_macro_mean"],
        result["f1_macro_std"],
        result["f1_weighted_mean"],
        result["f1_weighted_std"],
    )

    report = classification_report(y, y_pred, zero_division=0)
    logger.info("\nClassification report (full-dataset fit):\n%s", report)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_ROWS: list[tuple[str, str]] = [
    ("n_samples",          "Samples"),
    ("n_classes",          "Classes (EC depth)"),
    ("n_folds",            "CV folds"),
    ("accuracy_mean",      "Accuracy (mean)"),
    ("accuracy_std",       "Accuracy (std)"),
    ("f1_macro_mean",      "F1 macro (mean)"),
    ("f1_macro_std",       "F1 macro (std)"),
    ("f1_weighted_mean",   "F1 weighted (mean)"),
    ("f1_weighted_std",    "F1 weighted (std)"),
]


def print_report(results: list[dict]) -> None:
    col_w = 26
    name_w = 32

    header = f"{'Metric':<{col_w}}" + "".join(
        f"{r['name']:<{name_w}}" for r in results
    )
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

    X = df["smarts"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    logger.info("Labels: %s", list(le.classes_))

    results: list[dict] = []

    # --- SmartsTokenizer ---
    logger.info("\n--- SmartsTokenizer ---")
    smarts_pipeline = build_pipeline(make_smarts_tokenizer_fn())
    results.append(
        evaluate("SmartsTokenizer", smarts_pipeline, X, y, args.folds)
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
            X,
            str(sp_model.with_suffix("")),  # prefix without .model extension
            vocab_size=args.sp_vocab_size,
        )
        logger.info("SentencePiece model saved to %s", sp_model)

    logger.info("\n--- SentencePieceTokenizer (%s) ---", sp_model)
    sp_pipeline = build_pipeline(make_sp_tokenizer_fn(str(sp_model)))
    results.append(
        evaluate("SentencePieceTokenizer", sp_pipeline, X, y, args.folds)
    )

    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()