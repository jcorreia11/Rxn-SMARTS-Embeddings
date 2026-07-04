"""MLM collator and trainer for reaction SMARTS."""

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from smart_rxn_embeddings.evaluation.splitting import group_holdout_split

logger = logging.getLogger(__name__)

_MASK_TOKEN = "[MASK]"
_SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"}
# Pure structural punctuation: syntactically necessary but carry no chemical
# information on their own.  Excluded from the content-accuracy metric so that
# trivially predictable bracket-matching tokens do not inflate the score.
_STRUCTURAL_TOKENS = {"(", ")", ".", ">>", ">"}


@dataclass
class MLMCollator:
    """Collates a list of dataset items into a masked batch.

    Injects ``[MASK]`` into the vocabulary at collator level — existing saved
    vocabularies are not modified.  The injected ID is ``len(token_to_id)``
    unless ``[MASK]`` is already present.

    Masking uses **contiguous span sampling** (SpanBERT-style): span lengths are
    drawn from a geometric distribution with mean ``mean_span``, capped at
    ``max_span``.  The 80/10/10 substitution decision (replace with ``[MASK]``,
    a random token, or leave unchanged) is made once per span so that every
    token in a span receives the same treatment.

    Parameters
    ----------
    token_to_id:
        Original vocabulary mapping (special tokens must be present).
    mask_prob:
        Fraction of real tokens selected for masking (default 0.25).
    mean_span:
        Mean span length for the geometric distribution (default 3.0).
    max_span:
        Hard cap on span length (default 10).

    Attributes
    ----------
    mask_id:
        Integer ID assigned to ``[MASK]``.
    extended_vocab_size:
        Vocabulary size seen by the model (original + 1 if ``[MASK]`` was injected).
    content_token_ids:
        Frozen set of token IDs that are chemically informative (atoms, bonds,
        ring closures, stereochemistry, …).  Excludes special tokens and pure
        structural punctuation ``( ) . >> >``.
    """

    token_to_id: dict[str, int]
    mask_prob: float = 0.25
    mean_span: float = 3.0
    max_span: int = 10

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

        # Content token IDs: chemically meaningful tokens only.
        self.content_token_ids: frozenset[int] = frozenset(
            v
            for k, v in self.token_to_id.items()
            if k not in _SPECIAL_TOKENS and k not in _STRUCTURAL_TOKENS
        )

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
        """Apply span-based 80/10/10 masking to *input_ids*.

        Span lengths are sampled from Geometric(p = 1/mean_span), capped at
        ``max_span``.  The 80/10/10 substitution is decided once per span.

        Returns
        -------
        labels:
            Original IDs at masked positions; ``-100`` elsewhere (ignored by
            ``CrossEntropyLoss``).
        masked_ids:
            Input IDs with masking applied.
        """
        B, L = input_ids.shape
        labels = torch.full_like(input_ids, -100)
        masked_ids = input_ids.clone()
        p_stop = (
            1.0 / self.mean_span
        )  # geometric: stop extending with probability p_stop

        for b in range(B):
            real_pos: list[int] = attention_mask[b].nonzero(as_tuple=True)[0].tolist()
            n_real = len(real_pos)
            if n_real == 0:
                continue

            budget = max(1, round(self.mask_prob * n_real))
            already_masked = [False] * L
            total_masked = 0

            for _ in range(budget * 20):  # cap attempts to avoid infinite loop
                if total_masked >= budget:
                    break

                # Sample span start from real, not-yet-masked positions
                candidates = [p for p in real_pos if not already_masked[p]]
                if not candidates:
                    break
                start = candidates[random.randrange(len(candidates))]

                # Sample span length: Geometric(p_stop), min 1, max max_span
                span_len = 1
                while span_len < self.max_span and random.random() > p_stop:
                    span_len += 1

                # Collect real, unmasked positions in [start, start + span_len)
                span_positions = [
                    pos
                    for pos in range(start, min(start + span_len, L))
                    if attention_mask[b, pos].item() == 1 and not already_masked[pos]
                ]
                if not span_positions:
                    continue

                # 80/10/10 decision at span level
                r = random.random()
                for pos in span_positions:
                    if total_masked >= budget:
                        break
                    labels[b, pos] = input_ids[b, pos]
                    already_masked[pos] = True
                    total_masked += 1
                    if r < 0.8:
                        masked_ids[b, pos] = self.mask_id
                    elif r < 0.9:
                        masked_ids[b, pos] = self._rand_ids[
                            random.randrange(len(self._rand_ids))
                        ]
                    # else: keep original (remaining 10 %)

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
    mask_prob: float = 0.25
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
    groups:   Optional per-row reaction-group ids (same order as *dataset*),
              e.g. RetroRules ``reaction_group``. When given, the train/val
              split keeps every group on one side (see
              ``evaluation.splitting.group_holdout_split``), preventing
              near-duplicate radius-siblings from leaking into validation.
              When ``None`` (default), falls back to plain ``random_split``.
    split_seed: Random seed for the group-aware split (ignored otherwise).
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        collator: MLMCollator,
        config: TrainingConfig,
        device: str | None = None,
        groups=None,
        split_seed: int = 42,
    ) -> None:
        self.model = model
        self.collator = collator
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = torch.compile(self.model)

        self.split_strategy = "group" if groups is not None else "random"

        if config.val_split > 0.0:
            if groups is not None:
                train_idx, val_idx = group_holdout_split(
                    len(dataset), groups, test_size=config.val_split, seed=split_seed
                )
                self.train_dataset = Subset(dataset, train_idx)
                self.val_dataset = Subset(dataset, val_idx)
                n_train, n_val = len(train_idx), len(val_idx)
            else:
                n_val = max(1, int(len(dataset) * config.val_split))
                n_train = len(dataset) - n_val
                self.train_dataset, self.val_dataset = random_split(
                    dataset, [n_train, n_val]
                )
            logger.info(
                "Train/val split (%s): %d train | %d val (%.0f%%)",
                self.split_strategy,
                n_train,
                n_val,
                config.val_split * 100,
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = None

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self._amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None

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
        persistent = self.config.num_workers > 0
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.config.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
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
                persistent_workers=persistent,
            )

        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = self._make_scheduler(total_steps)

        train_losses: list[float] = []
        val_losses: list[float] = []
        val_top1: list[float] = []
        val_top5: list[float] = []
        val_content_top1: list[float] = []
        val_content_top5: list[float] = []
        t_start = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            avg_train = self._run_epoch(train_loader, epoch, scheduler)
            train_losses.append(avg_train)

            monitor = avg_train
            if val_loader is not None:
                avg_val, top1, top5, content_top1, content_top5 = self._run_val_epoch(
                    val_loader, epoch
                )
                val_losses.append(avg_val)
                val_top1.append(top1)
                val_top5.append(top5)
                val_content_top1.append(content_top1)
                val_content_top5.append(content_top5)
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
            val_top1=val_top1,
            val_top5=val_top5,
            val_content_top1=val_content_top1,
            val_content_top5=val_content_top5,
            training_time_s=elapsed,
        )
        logger.info(
            "Training complete in %.0f s. Final model saved to %s", elapsed, output_path
        )

        return train_losses

    def _make_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
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

            with torch.autocast(
                "cuda", dtype=self._amp_dtype, enabled=self._amp_dtype is not None
            ):
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
        logger.info(
            "Epoch %d/%d | avg train loss %.4f", epoch, self.config.num_epochs, avg
        )
        return avg

    def _run_val_epoch(
        self, loader: DataLoader, epoch: int
    ) -> tuple[float, float, float, float, float]:
        """Returns (avg_loss, top1, top5, content_top1, content_top5).

        *top1* / *top5* are computed over **all** masked positions.
        *content_top1* / *content_top5* are restricted to positions whose
        original token is chemically informative (i.e. in
        ``collator.content_token_ids``), excluding pure structural punctuation
        such as ``( ) . >> >`` that is trivially predictable from context.
        """
        self.model.eval()
        total_loss = 0.0
        correct_top1 = correct_top5 = total_masked = 0
        content_correct_top1 = content_correct_top5 = content_total_masked = 0

        content_ids_t = torch.tensor(
            sorted(self.collator.content_token_ids), device=self.device
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(
                    "cuda", dtype=self._amp_dtype, enabled=self._amp_dtype is not None
                ):
                    logits = self.model(input_ids, attention_mask)  # (B, L, V)
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                total_loss += loss.item()

                flat_labels = labels.view(-1)  # (B*L,)
                flat_logits = logits.view(-1, logits.size(-1))  # (B*L, V)
                masked = flat_labels != -100

                if masked.any():
                    m_logits = flat_logits[masked]  # (M, V)
                    m_labels = flat_labels[masked]  # (M,)

                    top5_preds = m_logits.topk(5, dim=-1).indices  # (M, 5)
                    correct_top1 += (top5_preds[:, 0] == m_labels).sum().item()
                    correct_top5 += (
                        (top5_preds == m_labels.unsqueeze(1)).any(dim=1).sum().item()
                    )
                    total_masked += m_labels.size(0)

                    # Content-only accuracy
                    is_content = torch.isin(m_labels, content_ids_t)
                    if is_content.any():
                        c_logits = m_logits[is_content]
                        c_labels = m_labels[is_content]
                        c_top5 = c_logits.topk(5, dim=-1).indices
                        content_correct_top1 += (c_top5[:, 0] == c_labels).sum().item()
                        content_correct_top5 += (
                            (c_top5 == c_labels.unsqueeze(1)).any(dim=1).sum().item()
                        )
                        content_total_masked += c_labels.size(0)

        avg = total_loss / len(loader)
        top1 = correct_top1 / total_masked if total_masked > 0 else 0.0
        top5 = correct_top5 / total_masked if total_masked > 0 else 0.0
        content_top1 = (
            content_correct_top1 / content_total_masked
            if content_total_masked > 0
            else 0.0
        )
        content_top5 = (
            content_correct_top5 / content_total_masked
            if content_total_masked > 0
            else 0.0
        )

        logger.info(
            "Epoch %d/%d | avg val loss %.4f | top-1 %.4f | top-5 %.4f"
            " | content top-1 %.4f | content top-5 %.4f",
            epoch,
            self.config.num_epochs,
            avg,
            top1,
            top5,
            content_top1,
            content_top5,
        )
        return avg, top1, top5, content_top1, content_top5

    def _save_weights(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def _save_config(
        self,
        path: Path,
        train_losses: list[float] | None = None,
        val_losses: list[float] | None = None,
        val_top1: list[float] | None = None,
        val_top5: list[float] | None = None,
        val_content_top1: list[float] | None = None,
        val_content_top5: list[float] | None = None,
        training_time_s: float | None = None,
    ) -> None:
        import json

        payload = {
            "model_config": self.model.config.to_dict(),
            "training_config": {k: v for k, v in self.config.__dict__.items()},
            "split_strategy": self.split_strategy,
            "mask_id": self.collator.mask_id,
            "extended_vocab_size": self.collator.extended_vocab_size,
            "history": {
                "train_losses": train_losses or [],
                "val_losses": val_losses or [],
                "val_top1_accuracy": val_top1 or [],
                "val_top5_accuracy": val_top5 or [],
                "val_content_top1_accuracy": val_content_top1 or [],
                "val_content_top5_accuracy": val_content_top5 or [],
                "training_time_s": training_time_s,
            },
        }
        path.write_text(json.dumps(payload, indent=2))
