"""SentencePiece-based tokenizer for reaction SMARTS."""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd
import sentencepiece as spm

logger = logging.getLogger(__name__)

_DEFAULT_VOCAB_SIZE = 1000
_DEFAULT_MODEL_TYPE = "bpe"


class SentencePieceTokenizer:
    """Data-driven SMARTS tokenizer backed by a trained SentencePiece model.

    The model treats each SMARTS string as a raw character sequence and learns
    subword units from their co-occurrence statistics.  This is complementary
    to the rule-based :class:`SmartsTokenizer`: it requires no chemistry
    knowledge but needs a training corpus.

    Parameters
    ----------
    model_path:
        Path to an existing ``*.model`` file produced by :meth:`train`.

    Examples
    --------
    Train and immediately use::

        tok = SentencePieceTokenizer.train(smarts_list, "models/sp")
        tok.tokenize("[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]")
    """

    def __init__(self, model_path: str) -> None:
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self.model_path = Path(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        smarts_list: list[str],
        model_prefix: str,
        vocab_size: int = _DEFAULT_VOCAB_SIZE,
        model_type: str = _DEFAULT_MODEL_TYPE,
    ) -> "SentencePieceTokenizer":
        """Train a SentencePiece model on *smarts_list* and return a tokenizer.

        The trained model is written to ``{model_prefix}.model`` (and a
        matching ``{model_prefix}.vocab`` sidecar).

        Parameters
        ----------
        smarts_list:
            Raw reaction SMARTS strings used as the training corpus.
        model_prefix:
            File-system prefix for the output model (e.g. ``"data/processed/sp"``).
        vocab_size:
            Target vocabulary size.  Values in the range 500–2000 work well for
            typical metabolic SMARTS corpora.
        model_type:
            SentencePiece algorithm: ``"bpe"`` (default) or ``"unigram"``.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(smarts_list))
            tmpfile = f.name

        try:
            spm.SentencePieceTrainer.train(
                input=tmpfile,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                normalization_rule_name="identity",
            )
            logger.info(
                "SentencePiece model saved to %s.model (vocab_size=%d, type=%s)",
                model_prefix,
                vocab_size,
                model_type,
            )
        finally:
            Path(tmpfile).unlink(missing_ok=True)

        return cls(f"{model_prefix}.model")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the model vocabulary."""
        return self._sp.GetPieceSize()

    def tokenize(self, smarts: str) -> list[str]:
        """Return the subword pieces for *smarts*."""
        return self._sp.EncodeAsPieces(smarts)

    def encode(
        self, smarts: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Return integer token IDs for *smarts*."""
        return self._sp.Encode(smarts, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int]) -> str:
        """Reconstruct a SMARTS string from integer token IDs."""
        return self._sp.Decode(ids)


INPUT_FILE = "data/processed/validated_smarts.csv"
OUTPUT_PREFIX = "data/processed/sp_tokenizer"
DEFAULT_VOCAB_SIZE = _DEFAULT_VOCAB_SIZE
DEFAULT_MODEL_TYPE = _DEFAULT_MODEL_TYPE


def main(
    input_file: str,
    output_prefix: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> None:
    df = pd.read_csv(input_file)
    valid_mask = df["valid"].astype(str).str.upper() == "TRUE"
    smarts_list = df.loc[valid_mask, "smarts"].tolist()
    logger.info("Loaded %d valid SMARTS from %s", len(smarts_list), input_file)

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    SentencePieceTokenizer.train(smarts_list, output_prefix, vocab_size, model_type)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Train a SentencePiece model on SMARTS."
    )
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--model-type", default=DEFAULT_MODEL_TYPE, choices=["bpe", "unigram"]
    )
    args = parser.parse_args()

    main(args.input, args.output_prefix, args.vocab_size, args.model_type)
