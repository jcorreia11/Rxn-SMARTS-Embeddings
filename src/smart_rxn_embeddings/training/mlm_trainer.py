"""MLM collator and trainer for reaction SMARTS."""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

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
    val_split:      Fraction of data held out for validation (0 = no split).
    warmup_steps:   Linear LR warmup steps; cosine decay for the remainder.
    max_grad_norm:  Gradient clipping max norm (0 = disabled).
    num_workers:    DataLoader worker processes.
    """

    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    mask_prob: float = 0.15
    checkpoint_dir: str = "models/checkpoints"
    output_path: str = "models/smarts_transformer.pt"
    log_every: int = 10
    val_split: float = 0.0
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    num_workers: int = 0


class Trainer:
    """Trains a :class:`SmartsMLMModel` with MLM objective.

    Saves the best checkpoint (by validation loss if ``val_split > 0``,
    otherwise by train loss) to *checkpoint_dir* and the final weights +
    config to *output_path*.

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
        self.collator = collator
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        if config.val_split > 0.0:
            n_val = max(1, int(len(dataset) * config.val_split))
            n_train = len(dataset) - n_val
            self.train_dataset, self.val_dataset = random_split(
                dataset, [n_train, n_val]
            )
            logger.info(
                "Train/val split: %d train | %d val (%.0f%%)",
                n_train,
                n_val,
                config.val_split * 100,
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = None

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self._ckpt_dir = Path(config.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_loss = float("inf")

    def train(self) -> list[float]:
        """Run the full training loop.

        Returns
        -------
        List of average *train* losses, one per epoch.
        """
        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=pin,
        )
        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=self.config.num_workers,
                pin_memory=pin,
            )

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = self._make_scheduler(total_steps)

        train_losses: list[float] = []
        val_losses: list[float] = []
        t_start = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            avg_train = self._run_epoch(train_loader, epoch, scheduler)
            train_losses.append(avg_train)

            monitor = avg_train
            if val_loader is not None:
                avg_val = self._run_val_epoch(val_loader, epoch)
                val_losses.append(avg_val)
                monitor = avg_val

            if monitor < self._best_loss:
                self._best_loss = monitor
                self._save_weights(self._ckpt_dir / "best.pt")
                logger.info(
                    "Epoch %d/%d | new best loss %.4f — checkpoint saved",
                    epoch,
                    self.config.num_epochs,
                    monitor,
                )

        elapsed = time.time() - t_start
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_weights(output_path)
        self._save_config(
            output_path.with_suffix(".json"),
            train_losses=train_losses,
            val_losses=val_losses,
            training_time_s=elapsed,
        )
        logger.info(
            "Training complete in %.0f s. Final model saved to %s", elapsed, output_path
        )

        return train_losses

    def _make_scheduler(
        self, total_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> float:
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
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % self.config.log_every == 0:
                logger.info(
                    "Epoch %d/%d | step %4d | loss %.4f | lr %.2e",
                    epoch,
                    self.config.num_epochs,
                    step,
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )

        avg = total_loss / len(loader)
        logger.info("Epoch %d/%d | avg train loss %.4f", epoch, self.config.num_epochs, avg)
        return avg

    def _run_val_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

        avg = total_loss / len(loader)
        logger.info("Epoch %d/%d | avg val   loss %.4f", epoch, self.config.num_epochs, avg)
        return avg

    def _save_weights(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def _save_config(
        self,
        path: Path,
        train_losses: list[float] | None = None,
        val_losses: list[float] | None = None,
        training_time_s: float | None = None,
    ) -> None:
        import json

        payload = {
            "model_config": self.model.config.to_dict(),
            "training_config": {k: v for k, v in self.config.__dict__.items()},
            "mask_id": self.collator.mask_id,
            "extended_vocab_size": self.collator.extended_vocab_size,
            "history": {
                "train_losses": train_losses or [],
                "val_losses": val_losses or [],
                "training_time_s": training_time_s,
            },
        }
        path.write_text(json.dumps(payload, indent=2))