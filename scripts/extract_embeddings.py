"""Extract reaction embeddings from a pretrained SmartsMLMModel."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rxn_smarts_embeddings.models.embed import SmartsEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-reaction embeddings from a pretrained transformer."
    )
    p.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pt)",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to model config (.json saved alongside weights)",
    )
    p.add_argument(
        "--vocab",
        default="data/processed/vocab.json",
        help="Path to vocab.json (default: data/processed/vocab.json)",
    )
    p.add_argument(
        "--data",
        default="data/processed/validated_smarts.csv",
        help="CSV with 'smarts' and 'valid' columns (default: data/processed/validated_smarts.csv)",
    )
    p.add_argument(
        "--output",
        default="data/embeddings/reaction_embeddings.npy",
        help="Output path for embeddings array (default: data/embeddings/reaction_embeddings.npy)",
    )
    p.add_argument(
        "--smarts-output",
        default="data/embeddings/reaction_smarts.txt",
        help="Output path for SMARTS list aligned with embeddings (default: data/embeddings/reaction_smarts.txt)",
    )
    p.add_argument(
        "--pooling",
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy: 'cls' (BOS token) or 'mean' (default: cls)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Sequences per forward pass (default: 256)",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override sequence length (default: from model config)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda' or 'cpu' (auto-detected if omitted)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    valid_mask = df["valid"].astype(str).str.upper() == "TRUE"
    smarts_list = df.loc[valid_mask, "smarts"].dropna().tolist()
    logger.info("Loaded %d valid SMARTS from %s", len(smarts_list), args.data)

    embedder = SmartsEmbedder.from_checkpoint(
        weights_path=args.weights,
        config_path=args.config,
        vocab_path=args.vocab,
        pooling=args.pooling,
        device=args.device,
    )

    t0 = time.perf_counter()
    embeddings = embedder.embed(
        smarts_list,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    elapsed_s = time.perf_counter() - t0

    assert embeddings.shape[0] == len(smarts_list), (
        f"Embedding count mismatch: {embeddings.shape[0]} embeddings for {len(smarts_list)} SMARTS"
    )
    assert embeddings.ndim == 2, f"Expected 2D array, got shape {embeddings.shape}"

    logger.info(
        "Extraction complete: %d sequences in %.1f s (%.0f sequences/s)",
        len(smarts_list),
        elapsed_s,
        len(smarts_list) / elapsed_s if elapsed_s > 0 else 0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info("Saved embeddings %s to %s", embeddings.shape, output_path)

    smarts_output = Path(args.smarts_output)
    smarts_output.write_text("\n".join(smarts_list))
    logger.info("Saved aligned SMARTS list to %s", smarts_output)


if __name__ == "__main__":
    main()
