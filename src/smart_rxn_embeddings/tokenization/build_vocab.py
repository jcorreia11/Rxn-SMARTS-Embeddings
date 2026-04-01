"""Build a token vocabulary from validated SMARTS using the rule-based tokenizer."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from smart_rxn_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
INPUT_FILE = "data/processed/validated_smarts.csv"
OUTPUT_FILE = "data/processed/vocab.json"


def build_vocab(
    smarts_list: list[str],
    special_tokens: list[str] = SPECIAL_TOKENS,
) -> dict:
    """Tokenize *smarts_list* and build a frequency-sorted vocabulary.

    Parameters
    ----------
    smarts_list:
        Iterable of clean reaction SMARTS strings.
    special_tokens:
        Tokens prepended to the vocabulary with fixed IDs (in order).
        Defaults to ``["[PAD]", "[UNK]", "[BOS]", "[EOS]"]``.

    Returns
    -------
    dict with keys:
        - ``token_to_id``: mapping token → integer ID
        - ``id_to_token``:  mapping str(id) → token
        - ``frequencies``:  token → raw count, sorted most-common first
        - ``size``:         total vocabulary size
    """
    tokenizer = SmartsTokenizer()
    counter: Counter = Counter()
    n_errors = 0

    for smarts in smarts_list:
        try:
            tokens = tokenizer.tokenize(smarts)
            counter.update(tokens)
        except ValueError as exc:
            logger.debug("Tokenization failed for %r: %s", smarts[:60], exc)
            n_errors += 1

    logger.info(
        "Tokenized %d SMARTS (%d errors). Unique tokens: %d",
        len(smarts_list) - n_errors,
        n_errors,
        len(counter),
    )

    # Special tokens get the first IDs; regular tokens follow, frequency-sorted
    token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(special_tokens)}
    next_id = len(special_tokens)
    for tok, _ in counter.most_common():
        if tok not in token_to_id:
            token_to_id[tok] = next_id
            next_id += 1

    id_to_token = {str(v): k for k, v in token_to_id.items()}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "frequencies": dict(counter.most_common()),
        "size": len(token_to_id),
    }


def save_vocab(vocab: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)
    logger.info("Saved vocabulary (%d tokens) to %s", vocab["size"], output_path)


def main(input_file: str, output_file: str) -> None:
    df = pd.read_csv(input_file)
    valid_mask = df["valid"].astype(str).str.upper() == "TRUE"
    smarts_list = df.loc[valid_mask, "smarts"].tolist()
    logger.info("Loaded %d valid SMARTS from %s", len(smarts_list), input_file)

    vocab = build_vocab(smarts_list)
    save_vocab(vocab, output_file)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Build SMARTS token vocabulary.")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    main(args.input, args.output)