"""CLI entry point for MLM pretraining on reaction SMARTS."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from smart_rxn_embeddings.datasets.smarts_dataset import SMARTSDataset
from smart_rxn_embeddings.models.smarts_transformer import (
    SmartsMLMModel,
    TransformerConfig,
)
from smart_rxn_embeddings.training.mlm_trainer import (
    MLMCollator,
    Trainer,
    TrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain a masked-language transformer on reaction SMARTS."
    )

    data = p.add_argument_group("data")
    data.add_argument(
        "--data",
        default="data/processed/validated_smarts.csv",
        help="CSV with 'smarts' and 'valid' columns",
    )
    data.add_argument(
        "--vocab",
        default="data/processed/vocab.json",
        help="Vocabulary JSON produced by build_vocab.py",
    )
    data.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Pad/truncate sequences to this length",
    )

    model = p.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--nhead", type=int, default=4)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--dim-feedforward", type=int, default=512)
    model.add_argument("--dropout", type=float, default=0.1)

    train = p.add_argument_group("training")
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--mask-prob", type=float, default=0.15)
    train.add_argument("--checkpoint-dir", default="models/checkpoints")
    train.add_argument("--output", default="models/smarts_transformer.pt")
    train.add_argument(
        "--device", default=None, help="'cuda' or 'cpu' (auto-detected if omitted)"
    )
    train.add_argument(
        "--log-every", type=int, default=10, help="Log batch loss every N steps"
    )
    train.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data held out for validation (default: 0.1)",
    )
    train.add_argument(
        "--warmup-steps", type=int, default=500,
        help="Linear LR warmup steps before cosine decay (default: 500)",
    )
    train.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Gradient clipping max norm, 0 to disable (default: 1.0)",
    )
    train.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    vocab = json.loads(Path(args.vocab).read_text())
    token_to_id: dict[str, int] = vocab["token_to_id"]

    df = pd.read_csv(args.data)
    valid_mask = df["valid"].astype(str).str.upper() == "TRUE"
    smarts_list = df.loc[valid_mask, "smarts"].tolist()
    logger.info("Loaded %d valid SMARTS from %s", len(smarts_list), args.data)

    dataset = SMARTSDataset(smarts_list, token_to_id, max_length=args.max_length)
    collator = MLMCollator(token_to_id=token_to_id, mask_prob=args.mask_prob)
    logger.info(
        "Vocabulary: %d base tokens + [MASK] → %d total (mask_id=%d)",
        len(token_to_id),
        collator.extended_vocab_size,
        collator.mask_id,
    )

    model_config = TransformerConfig(
        vocab_size=collator.extended_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_length,
    )
    model = SmartsMLMModel(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_params)

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        mask_prob=args.mask_prob,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        log_every=args.log_every,
        val_split=args.val_split,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
    )

    trainer = Trainer(model, dataset, collator, training_config, device=args.device)
    losses = trainer.train()
    logger.info(
        "Done. Losses: %s",
        " → ".join(f"{v:.4f}" for v in losses),
    )


if __name__ == "__main__":
    main()
