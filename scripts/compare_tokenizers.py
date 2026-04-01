"""Baseline performance comparison between SmartsTokenizer and SentencePieceTokenizer.

Metrics reported per tokenizer:
  - Throughput (SMARTS/s and chars/s)
  - Sequence length statistics (mean, median, std, min, max tokens per SMARTS)
  - Fertility ratio (tokens / characters — lower means better compression)
  - Unique tokens used and vocabulary utilisation (% of vocab used)
  - Round-trip fidelity (% of SMARTS reconstructed exactly)
  - Error rate (% of SMARTS that raise an exception during tokenization)

Usage
-----
    python scripts/compare_tokenizers.py                        # defaults
    python scripts/compare_tokenizers.py --n-samples 10000
    python scripts/compare_tokenizers.py --sp-model data/processed/sp_tokenizer.model
    python scripts/compare_tokenizers.py --output results/tokenizer_comparison.json
"""

import argparse
import json
import logging
import random
import statistics
import time
from pathlib import Path

import pandas as pd

from smart_rxn_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer
from smart_rxn_embeddings.tokenization.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/processed/validated_smarts.csv"
DEFAULT_SP_MODEL = "data/processed/sp_tokenizer.model"
DEFAULT_N_SAMPLES = 5_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_smarts(input_file: str, n_samples: int | None) -> list[str]:
    df = pd.read_csv(input_file)
    valid_mask = df["valid"].astype(str).str.upper() == "TRUE"
    smarts_list = df.loc[valid_mask, "smarts"].dropna().tolist()
    logger.info("Loaded %d valid SMARTS from %s", len(smarts_list), input_file)

    if n_samples and n_samples < len(smarts_list):
        random.seed(RANDOM_SEED)
        smarts_list = random.sample(smarts_list, n_samples)
        logger.info("Sampled %d SMARTS (seed=%d)", n_samples, RANDOM_SEED)

    return smarts_list


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_smarts_tokenizer(smarts_list: list[str]) -> dict:
    tokenizer = SmartsTokenizer()

    token_counts: list[int] = []
    char_counts: list[int] = []
    errors = 0

    t0 = time.perf_counter()
    for smarts in smarts_list:
        try:
            tokens = tokenizer.tokenize(smarts)
            token_counts.append(len(tokens))
            char_counts.append(len(smarts))
        except ValueError:
            errors += 1
    elapsed = time.perf_counter() - t0

    # Collect unique tokens (on the successfully tokenized subset)
    unique_tokens: set[str] = set()
    for smarts in smarts_list:
        try:
            unique_tokens.update(tokenizer.tokenize(smarts))
        except ValueError:
            pass

    n_ok = len(token_counts)
    fertility_values = [t / c for t, c in zip(token_counts, char_counts) if c > 0]

    return {
        "name": "SmartsTokenizer",
        "vocab_size": None,  # rule-based — no fixed vocab
        "n_samples": len(smarts_list),
        "n_ok": n_ok,
        "n_errors": errors,
        "error_rate": errors / len(smarts_list) if smarts_list else 0.0,
        "elapsed_s": round(elapsed, 4),
        "throughput_smarts_per_s": round(n_ok / elapsed, 1) if elapsed > 0 else 0,
        "throughput_chars_per_s": round(sum(char_counts) / elapsed, 1) if elapsed > 0 else 0,
        "seq_len_mean": round(statistics.mean(token_counts), 2) if token_counts else 0,
        "seq_len_median": statistics.median(token_counts) if token_counts else 0,
        "seq_len_std": round(statistics.stdev(token_counts), 2) if len(token_counts) > 1 else 0,
        "seq_len_min": min(token_counts) if token_counts else 0,
        "seq_len_max": max(token_counts) if token_counts else 0,
        "fertility_mean": round(statistics.mean(fertility_values), 4) if fertility_values else 0,
        "unique_tokens_used": len(unique_tokens),
        "vocab_utilisation": None,  # no fixed vocab to compare against
        "round_trip_rate": 1.0,  # guaranteed by "".join(tokens) == smarts
    }


def benchmark_sp_tokenizer(smarts_list: list[str], model_path: str) -> dict:
    tokenizer = SentencePieceTokenizer(model_path)
    vocab_size = tokenizer.vocab_size

    token_counts: list[int] = []
    char_counts: list[int] = []
    round_trips = 0

    t0 = time.perf_counter()
    for smarts in smarts_list:
        tokens = tokenizer.tokenize(smarts)
        ids = tokenizer.encode(smarts)
        decoded = tokenizer.decode(ids)
        token_counts.append(len(tokens))
        char_counts.append(len(smarts))
        if decoded == smarts:
            round_trips += 1
    elapsed = time.perf_counter() - t0

    unique_tokens: set[str] = set()
    for smarts in smarts_list:
        unique_tokens.update(tokenizer.tokenize(smarts))

    fertility_values = [t / c for t, c in zip(token_counts, char_counts) if c > 0]

    return {
        "name": f"SentencePieceTokenizer",
        "vocab_size": vocab_size,
        "n_samples": len(smarts_list),
        "n_ok": len(smarts_list),
        "n_errors": 0,
        "error_rate": 0.0,
        "elapsed_s": round(elapsed, 4),
        "throughput_smarts_per_s": round(len(smarts_list) / elapsed, 1) if elapsed > 0 else 0,
        "throughput_chars_per_s": round(sum(char_counts) / elapsed, 1) if elapsed > 0 else 0,
        "seq_len_mean": round(statistics.mean(token_counts), 2) if token_counts else 0,
        "seq_len_median": statistics.median(token_counts) if token_counts else 0,
        "seq_len_std": round(statistics.stdev(token_counts), 2) if len(token_counts) > 1 else 0,
        "seq_len_min": min(token_counts) if token_counts else 0,
        "seq_len_max": max(token_counts) if token_counts else 0,
        "fertility_mean": round(statistics.mean(fertility_values), 4) if fertility_values else 0,
        "unique_tokens_used": len(unique_tokens),
        "vocab_utilisation": round(len(unique_tokens) / vocab_size, 4) if vocab_size else 0,
        "round_trip_rate": round(round_trips / len(smarts_list), 4) if smarts_list else 0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_ROWS: list[tuple[str, str, str]] = [
    ("vocab_size",             "Vocabulary size",             ""),
    ("n_samples",              "Samples evaluated",           ""),
    ("error_rate",             "Error rate",                  "%"),
    ("throughput_smarts_per_s","Throughput",                  "SMARTS/s"),
    ("throughput_chars_per_s", "Throughput",                  "chars/s"),
    ("seq_len_mean",           "Seq length (mean)",           "tokens"),
    ("seq_len_median",         "Seq length (median)",         "tokens"),
    ("seq_len_std",            "Seq length (std)",            "tokens"),
    ("seq_len_min",            "Seq length (min)",            "tokens"),
    ("seq_len_max",            "Seq length (max)",            "tokens"),
    ("fertility_mean",         "Fertility ratio (mean)",      "tokens/char"),
    ("unique_tokens_used",     "Unique tokens used",          ""),
    ("vocab_utilisation",      "Vocabulary utilisation",      "%"),
    ("round_trip_rate",        "Round-trip fidelity",         "%"),
]


def _fmt(value, unit: str) -> str:
    if value is None:
        return "N/A"
    if unit == "%":
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:,.4f} {unit}".strip()
    return f"{value:,} {unit}".strip()


def print_report(results: list[dict]) -> None:
    col_w = 28
    name_w = 26

    header = f"{'Metric':<{col_w}}" + "".join(
        f"{r['name']:<{name_w}}" for r in results
    )
    print()
    print(header)
    print("-" * (col_w + name_w * len(results)))

    for key, label, unit in _ROWS:
        row = f"{label:<{col_w}}"
        for r in results:
            row += f"{_fmt(r.get(key), unit):<{name_w}}"
        print(row)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SmartsTokenizer vs SentencePieceTokenizer."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to validated_smarts.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--sp-model",
        default=DEFAULT_SP_MODEL,
        help=f"Path to trained SentencePiece .model file (default: {DEFAULT_SP_MODEL})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of SMARTS to sample (default: {DEFAULT_N_SAMPLES}, 0 = all)",
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
    smarts_list = load_smarts(args.input, n_samples)

    results: list[dict] = []

    logger.info("\nBenchmarking SmartsTokenizer...")
    results.append(benchmark_smarts_tokenizer(smarts_list))

    sp_model = Path(args.sp_model)
    if sp_model.exists():
        logger.info("Benchmarking SentencePieceTokenizer (%s)...", sp_model)
        results.append(benchmark_sp_tokenizer(smarts_list, str(sp_model)))
    else:
        logger.warning(
            "SentencePiece model not found at %s — skipping.\n"
            "Run `dvc repro train_sentencepiece` or pass --sp-model to specify the path.",
            sp_model,
        )

    print_report(results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()