"""Embedding–structural similarity correlation for a random-initialised model.

Identical to ``similarity_correlation.py`` in every respect (same SMARTS,
same sampled pairs, same Tanimoto computation) except that the model weights
are **never loaded** — the transformer is instantiated with random weights and
kept in eval mode.  This provides a lower-bound reference: if random
projections already correlate with structural similarity, the pretrained
model's correlation carries less interpretive weight.

Usage
-----
    python scripts/similarity_correlation_random_baseline.py \\
        --config  results/final/1-train-mlm/medium/smarts_transformer_20260428_160310.json \\
        --vocab   data/processed/vocab.json \\
        --smarts  data/embeddings/medium/smarts.txt \\
        --n-reactions 3000 \\
        --seed    42 \\
        --output  results/figures/sim_corr_random_baseline \\
        --save-json results/sim_corr_random_baseline.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ensure scripts/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from similarity_correlation import sample_and_compute, compute_stats, plot_correlation


# ---------------------------------------------------------------------------
# Random-init embedding
# ---------------------------------------------------------------------------


def build_random_embedder(config_path: str, vocab_path: str, device: str | None, seed: int = 42):
    """Return a SmartsEmbedder with random (untrained) weights."""
    import json as _json
    from smart_rxn_embeddings.models.embed import SmartsEmbedder
    from smart_rxn_embeddings.models.smarts_transformer import (
        SmartsMLMModel,
        TransformerConfig,
    )

    cfg = _json.loads(Path(config_path).read_text())
    model_config = TransformerConfig.from_dict(cfg["model_config"])
    torch.manual_seed(seed)  # fix init seed for reproducibility
    model = SmartsMLMModel(model_config)  # random weights — no checkpoint loaded
    model.eval()
    logger.info(
        "Built random-init model: d_model=%d, layers=%d, max_len=%d",
        model_config.d_model,
        model_config.num_encoder_layers,
        model_config.max_seq_len,
    )

    vocab = _json.loads(Path(vocab_path).read_text())
    token_to_id: dict[str, int] = vocab["token_to_id"]

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SmartsEmbedder(model, token_to_id, pooling="mean", device=_device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embedding–structural similarity correlation for a "
            "random-initialised (untrained) model."
        )
    )
    p.add_argument(
        "--config",
        required=True,
        help="Model config JSON (e.g. smarts_transformer_<ID>.json)",
    )
    p.add_argument(
        "--vocab",
        default="data/processed/vocab.json",
        help="Vocabulary JSON (default: data/processed/vocab.json)",
    )
    p.add_argument(
        "--smarts",
        required=True,
        help="Path to SMARTS .txt aligned with the pretrained embeddings (one per line)",
    )
    p.add_argument(
        "--n-reactions",
        type=int,
        default=3000,
        help="Reactions to sample (default: 3000 → 4,498,500 pairs)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reaction sampling seed — must match the pretrained run to compare the same pairs (default: 42)",
    )
    p.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Torch seed for random weight initialisation (default: 42)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Sequences per forward pass (default: 256)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="'cuda' or 'cpu' (auto-detected if omitted)",
    )
    p.add_argument(
        "--output",
        default="results/figures/sim_corr_random_baseline",
        help="Output path without extension",
    )
    p.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure format (default: pdf)",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--gridsize", type=int, default=50)
    p.add_argument(
        "--save-json",
        default=None,
        help="Save statistics and metadata to this JSON path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_smarts = Path(args.smarts).read_text().splitlines()
    n_total = len(all_smarts)
    logger.info("Loaded %d SMARTS from %s", n_total, args.smarts)

    # Pre-sample the same indices that sample_and_compute would select so we
    # only embed 3,000 reactions instead of all 361k (mirrors the sort in
    # sample_and_compute exactly, keeping Tanimoto pairs identical to pretrained runs).
    n_reactions = min(args.n_reactions, n_total)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=n_reactions, replace=False)
    idx.sort()
    sub_smarts = [all_smarts[i] for i in idx]
    logger.info("Pre-sampled %d reactions (seed=%d)", n_reactions, args.seed)

    embedder = build_random_embedder(args.config, args.vocab, args.device, seed=args.model_seed)

    logger.info("Extracting random-init embeddings for sampled reactions ...")
    t0_total = time.perf_counter()
    embeddings = embedder.embed(sub_smarts, batch_size=args.batch_size)
    logger.info("  Embeddings shape: %s", embeddings.shape)

    # Pass n_reactions == len(sub_smarts) so sample_and_compute uses all rows
    # in their existing order (same SMARTS order → same Tanimoto pairs as pretrained runs).
    cosine, tanimoto, n_reactions, n_pairs, _same_group = sample_and_compute(
        embeddings, sub_smarts, len(sub_smarts), args.seed
    )

    stats = compute_stats(cosine, tanimoto)
    total_time_s = time.perf_counter() - t0_total

    logger.info(
        "\nResults (random-init baseline):\n"
        "  Pearson  r = %.4f  (p = %.2e)\n"
        "  Spearman ρ = %.4f  (p = %.2e)\n"
        "  N pairs    = %d\n"
        "  Cosine  : mean=%.4f  std=%.4f\n"
        "  Tanimoto: mean=%.4f  std=%.4f",
        stats["pearson_r"], stats["pearson_p"],
        stats["spearman_r"], stats["spearman_p"],
        stats["n_pairs"],
        stats["cosine_mean"], stats["cosine_std"],
        stats["tanimoto_mean"], stats["tanimoto_std"],
    )

    plot_correlation(
        cosine, tanimoto, stats,
        output_path=Path(args.output),
        fmt=args.format,
        dpi=args.dpi,
        gridsize=args.gridsize,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "config_path": args.config,
                "smarts_path": args.smarts,
                "model": "random_init",
                "n_reactions_total": n_total,
                "n_reactions_sampled": n_reactions,
                "n_pairs": n_pairs,
                "seed": args.seed,
                "model_seed": args.model_seed,
                "fingerprint": "RDKit structural reaction fingerprint (4096 bits)",
                "total_time_s": round(total_time_s, 2),
            },
            "stats": stats,
        }
        out.write_text(json.dumps(payload, indent=2))
        logger.info("Statistics saved to %s", out)


if __name__ == "__main__":
    main()