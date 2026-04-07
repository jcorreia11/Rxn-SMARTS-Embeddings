"""MLM collator and trainer for reaction SMARTS."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_MASK_TOKEN = "[MASK]"
_SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"}


@dataclass
class MLMCollator:
    """Collates a list of dataset items into a masked batch.

    Injects ``[MASK]`` into the vocabulary at collator level — existing saved
    vocabularies are not modified.  The injected ID is ``len(token_to_id)``
    unless ``[MASK]`` is already present.

    Parameters
    ----------
    token_to_id:
        Original vocabulary mapping (special tokens must be present).
    mask_prob:
        Fraction of real tokens selected for masking (default 0.15).

    Attributes
    ----------
    mask_id:
        Integer ID assigned to ``[MASK]``.
    extended_vocab_size:
        Vocabulary size seen by the model (original + 1 if ``[MASK]`` was injected).
    """

    token_to_id: dict[str, int]
    mask_prob: float = 0.15

    def __post_init__(self) -> None:
        if _MASK_TOKEN in self.token_to_id:
            self.mask_id = self.token_to_id[_MASK_TOKEN]
            self.extended_vocab_size = len(self.token_to_id)
        else:
            self.mask_id = len(self.token_to_id)
            self.extended_vocab_size = len(self.token_to_id) + 1

        # Pool of IDs eligible for random-token substitution (no specials)
        self._rand_ids = [
            v for k, v in self.token_to_id.items() if k not in _SPECIAL_TOKENS
        ]

    def __call__(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels, masked_ids = self._mask(input_ids, attention_mask)
        return {
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply BERT-style 80/10/10 masking to *input_ids*.

        Returns
        -------
        labels:
            Original IDs at masked positions; ``-100`` elsewhere (ignored by
            ``CrossEntropyLoss``).
        masked_ids:
            Input IDs with masking applied.
        """
        labels = input_ids.clone()
        masked_ids = input_ids.clone()

        real = attention_mask.bool()
        selected = real & (
            torch.rand_like(input_ids, dtype=torch.float) < self.mask_prob
        )

        labels[~selected] = -100

        r = torch.rand_like(input_ids, dtype=torch.float)
        replace_mask = selected & (r < 0.8)
        replace_rand = selected & (r >= 0.8) & (r < 0.9)
        # remaining 10 %: keep original — no action needed

        masked_ids[replace_mask] = self.mask_id

        n_rand = int(replace_rand.sum().item())
        if n_rand > 0:
            rand_ids = torch.tensor(
                [
                    self._rand_ids[i % len(self._rand_ids)]
                    for i in torch.randperm(n_rand).tolist()
                ],
                dtype=torch.long,
            )
            masked_ids[replace_rand] = rand_ids

        return labels, masked_ids


@dataclass
class TrainingConfig:
    """Training hyper-parameters.

    Parameters
    ----------
    learning_rate:  AdamW learning rate.
    batch_size:     Samples per gradient step.
    num_epochs:     Total training epochs.
    mask_prob:      Fraction of tokens masked per sequence.
    checkpoint_dir: Directory for per-epoch best checkpoints.
    output_path:    Final model weights path.
    log_every:      Log batch-level loss every N steps.
    """

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    mask_prob: float = 0.15
    checkpoint_dir: str = "models/checkpoints"
    output_path: str = "models/smarts_transformer.pt"
    log_every: int = 10


class Trainer:
    """Trains a :class:`SmartsMLMModel` with MLM objective.

    Saves the best checkpoint (by average epoch loss) to *checkpoint_dir* and
    the final weights + config to *output_path*.

    Parameters
    ----------
    model:    The model to train.
    dataset:  A ``torch.utils.data.Dataset`` returning ``input_ids`` /
              ``attention_mask`` dicts.
    collator: :class:`MLMCollator` instance (handles masking).
    config:   :class:`TrainingConfig` with all hyper-parameters.
    device:   ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        collator: MLMCollator,
        config: TrainingConfig,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.collator = collator
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self._ckpt_dir = Path(config.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_loss = float("inf")

    def train(self) -> list[float]:
        """Run the full training loop.

        Returns
        -------
        List of average losses, one per epoch.
        """
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        epoch_losses: list[float] = []

        for epoch in range(1, self.config.num_epochs + 1):
            avg_loss = self._run_epoch(loader, epoch)
            epoch_losses.append(avg_loss)

            if avg_loss < self._best_loss:
                self._best_loss = avg_loss
                self._save_weights(self._ckpt_dir / "best.pt")
                logger.info(
                    "Epoch %d/%d | new best loss %.4f — checkpoint saved",
                    epoch,
                    self.config.num_epochs,
                    avg_loss,
                )

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_weights(output_path)
        self._save_config(output_path.with_suffix(".json"))
        logger.info("Training complete. Final model saved to %s", output_path)

        return epoch_losses

    def _run_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)  # (B, L, V)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if step % self.config.log_every == 0:
                logger.info(
                    "Epoch %d/%d | step %4d | loss %.4f",
                    epoch,
                    self.config.num_epochs,
                    step,
                    loss.item(),
                )

        avg = total_loss / len(loader)
        logger.info("Epoch %d/%d | avg loss %.4f", epoch, self.config.num_epochs, avg)
        return avg

    def _save_weights(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def _save_config(self, path: Path) -> None:
        import json

        payload = {
            "model_config": self.model.config.to_dict(),
            "training_config": {k: v for k, v in self.config.__dict__.items()},
            "mask_id": self.collator.mask_id,
            "extended_vocab_size": self.collator.extended_vocab_size,
        }
        path.write_text(json.dumps(payload, indent=2))
