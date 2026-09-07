import random

import pytest
import torch

from rxn_smarts_embeddings.models.smarts_transformer import (
    SmartsMLMModel,
    TransformerConfig,
)
from rxn_smarts_embeddings.training.mlm_trainer import (
    MLMCollator,
    Trainer,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VOCAB: dict[str, int] = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "-": 4,
    "=": 5,
    ">>": 6,
    "[C:1]": 7,
    "[O:2]": 8,
    "c": 9,
    "n": 10,
    "1": 11,
}

MAX_LEN = 16
BATCH = 4


@pytest.fixture
def collator():
    return MLMCollator(token_to_id=_VOCAB, mask_prob=0.5)


def _make_batch(size: int = BATCH) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.randint(1, len(_VOCAB), (MAX_LEN,)),
            "attention_mask": torch.ones(MAX_LEN, dtype=torch.long),
        }
        for _ in range(size)
    ]


# ---------------------------------------------------------------------------
# MLMCollator — vocabulary injection
# ---------------------------------------------------------------------------


class TestMLMCollatorVocabInjection:
    def test_mask_id_is_len_vocab_when_absent(self):
        c = MLMCollator(token_to_id=_VOCAB)
        assert c.mask_id == len(_VOCAB)

    def test_extended_vocab_size_is_vocab_plus_one(self):
        c = MLMCollator(token_to_id=_VOCAB)
        assert c.extended_vocab_size == len(_VOCAB) + 1

    def test_mask_id_reused_when_already_present(self):
        vocab_with_mask = {**_VOCAB, "[MASK]": 99}
        c = MLMCollator(token_to_id=vocab_with_mask)
        assert c.mask_id == 99
        assert c.extended_vocab_size == len(vocab_with_mask)

    def test_content_token_ids_excludes_structural_tokens(self):
        # ">>" is in _VOCAB (id=6) and is a structural token — must be absent
        c = MLMCollator(token_to_id=_VOCAB)
        assert _VOCAB[">>"] not in c.content_token_ids

    def test_content_token_ids_excludes_special_tokens(self):
        c = MLMCollator(token_to_id=_VOCAB)
        for key in ("[PAD]", "[UNK]", "[BOS]", "[EOS]"):
            assert _VOCAB[key] not in c.content_token_ids

    def test_content_token_ids_includes_atoms_and_bonds(self):
        c = MLMCollator(token_to_id=_VOCAB)
        # "-", "=", "[C:1]", "[O:2]", "c", "n", "1" are all content
        for key in ("-", "=", "[C:1]", "[O:2]", "c", "n", "1"):
            assert _VOCAB[key] in c.content_token_ids


# ---------------------------------------------------------------------------
# MLMCollator — collation output
# ---------------------------------------------------------------------------


class TestMLMCollatorOutput:
    def test_returns_required_keys(self, collator):
        batch = collator(_make_batch())
        assert {"input_ids", "attention_mask", "labels"} == set(batch.keys())

    def test_output_shapes(self, collator):
        batch = collator(_make_batch())
        for key in ("input_ids", "attention_mask", "labels"):
            assert batch[key].shape == (BATCH, MAX_LEN)

    def test_attention_mask_unchanged(self, collator):
        items = _make_batch()
        original_mask = torch.stack([i["attention_mask"] for i in items])
        batch = collator(items)
        assert torch.equal(batch["attention_mask"], original_mask)


# ---------------------------------------------------------------------------
# MLMCollator — masking correctness
# ---------------------------------------------------------------------------


class TestMLMCollatorSpanMasking:
    def test_default_mean_span_and_max_span(self):
        c = MLMCollator(token_to_id=_VOCAB)
        assert c.mean_span == 3.0
        assert c.max_span == 10

    def test_custom_span_params_stored(self):
        c = MLMCollator(token_to_id=_VOCAB, mean_span=5.0, max_span=15)
        assert c.mean_span == 5.0
        assert c.max_span == 15

    def test_masked_positions_form_contiguous_spans(self):
        """With mean_span=MAX_LEN all real tokens masked in one span → contiguous block."""
        random.seed(0)
        c = MLMCollator(
            token_to_id=_VOCAB, mask_prob=1.0, mean_span=MAX_LEN, max_span=MAX_LEN
        )
        items = [
            {
                "input_ids": torch.arange(1, MAX_LEN + 1, dtype=torch.long),
                "attention_mask": torch.ones(MAX_LEN, dtype=torch.long),
            }
        ]
        batch = c(items)
        masked_positions = (
            (batch["labels"][0] != -100).nonzero(as_tuple=True)[0].tolist()
        )
        # All real positions should be masked when mask_prob=1.0
        assert len(masked_positions) == MAX_LEN

    def test_span_masking_satisfies_existing_invariants(self):
        """Span masking must not mask padding and must produce valid token IDs."""
        random.seed(42)
        c = MLMCollator(token_to_id=_VOCAB, mask_prob=0.5, mean_span=3.0)
        items = [
            {
                "input_ids": torch.cat(
                    [
                        torch.randint(1, len(_VOCAB), (8,)),
                        torch.zeros(8, dtype=torch.long),
                    ]
                ),
                "attention_mask": torch.cat(
                    [torch.ones(8, dtype=torch.long), torch.zeros(8, dtype=torch.long)]
                ),
            }
        ]
        batch = c(items)
        # Padding positions never masked
        assert (batch["labels"][0, 8:] == -100).all()
        # Changed IDs are within extended vocab
        changed = batch["input_ids"][0] != items[0]["input_ids"]
        if changed.any():
            assert (batch["input_ids"][0][changed] < c.extended_vocab_size).all()


class TestMLMCollatorMasking:
    def test_labels_minus100_at_unmasked_positions(self, collator):
        torch.manual_seed(0)
        batch = collator(_make_batch())
        unmasked = batch["labels"] == -100
        # At unmasked positions, input_ids must equal original (labels stores original)
        # We can't check original easily, but we can verify labels are binary: id or -100
        valid_ids = batch["labels"][~unmasked]
        assert (valid_ids >= 0).all()

    def test_masked_positions_have_mask_id_or_original(self, collator):
        """At selected positions, input_ids must be mask_id, a random token, or unchanged."""
        torch.manual_seed(42)
        items = _make_batch()
        original_ids = torch.stack([i["input_ids"] for i in items])
        batch = collator(items)

        selected = batch["labels"] != -100
        changed = batch["input_ids"][selected] != original_ids[selected]
        if changed.any():
            changed_vals = batch["input_ids"][selected][changed]
            assert (changed_vals < collator.extended_vocab_size).all()

    def test_padding_tokens_never_masked(self):
        c = MLMCollator(token_to_id=_VOCAB, mask_prob=1.0)
        items = [
            {
                "input_ids": torch.cat(
                    [
                        torch.randint(1, len(_VOCAB), (8,)),
                        torch.zeros(8, dtype=torch.long),
                    ]
                ),
                "attention_mask": torch.cat(
                    [torch.ones(8, dtype=torch.long), torch.zeros(8, dtype=torch.long)]
                ),
            }
        ]
        batch = c(items)
        pad_labels = batch["labels"][0, 8:]
        assert (pad_labels == -100).all()

    def test_masking_is_stochastic(self, collator):
        items = _make_batch(16)
        b1 = collator(items)
        b2 = collator(items)
        assert not torch.equal(b1["input_ids"], b2["input_ids"])


# ---------------------------------------------------------------------------
# Trainer — smoke test
# ---------------------------------------------------------------------------


class _TinyDataset(torch.utils.data.Dataset):
    """16 random items compatible with SMARTSDataset output format."""

    def __init__(self, vocab_size: int, seq_len: int, n: int = 16) -> None:
        self._ids = torch.randint(1, vocab_size, (n, seq_len))
        self._mask = torch.ones(n, seq_len, dtype=torch.long)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self._ids[idx], "attention_mask": self._mask[idx]}


@pytest.fixture
def tiny_setup():
    collator = MLMCollator(token_to_id=_VOCAB, mask_prob=0.15)
    model_config = TransformerConfig(
        vocab_size=collator.extended_vocab_size,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=MAX_LEN,
    )
    model = SmartsMLMModel(model_config)
    dataset = _TinyDataset(len(_VOCAB), MAX_LEN)
    return model, dataset, collator


class TestTrainer:
    def test_train_returns_one_loss_per_epoch(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        losses = trainer.train()
        assert len(losses) == 2

    def test_losses_are_finite(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        losses = trainer.train()
        assert all(isinstance(v, float) and torch.tensor(v).isfinite() for v in losses)

    def test_output_weights_saved(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        output = tmp_path / "model.pt"
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(output),
        )
        Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert output.exists()

    def test_config_json_saved_alongside_weights(self, tiny_setup, tmp_path):
        import json

        model, dataset, collator = tiny_setup
        output = tmp_path / "model.pt"
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(output),
        )
        Trainer(model, dataset, collator, cfg, device="cpu").train()
        config_path = output.with_suffix(".json")
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert "model_config" in saved
        assert "mask_id" in saved

    def test_best_checkpoint_saved(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        ckpt_dir = tmp_path / "ckpts"
        cfg = TrainingConfig(
            num_epochs=3,
            batch_size=4,
            checkpoint_dir=str(ckpt_dir),
            output_path=str(tmp_path / "model.pt"),
        )
        Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert (ckpt_dir / "best.pt").exists()

    def test_loss_decreases_over_epochs(self, tmp_path):
        """Loss should trend downward when the model can overfit a tiny dataset."""
        torch.manual_seed(0)
        vocab = {**_VOCAB}
        collator = MLMCollator(token_to_id=vocab, mask_prob=0.15)
        model_config = TransformerConfig(
            vocab_size=collator.extended_vocab_size,
            d_model=32,
            nhead=2,
            num_encoder_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            max_seq_len=MAX_LEN,
        )
        model = SmartsMLMModel(model_config)
        dataset = _TinyDataset(len(vocab), MAX_LEN, n=8)
        cfg = TrainingConfig(
            learning_rate=1e-2,
            batch_size=8,
            num_epochs=10,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        losses = Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestTrainerValSplit:
    def test_val_split_creates_separate_datasets(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        assert trainer.val_dataset is not None
        assert len(trainer.train_dataset) + len(trainer.val_dataset) == len(dataset)

    def test_val_split_zero_no_val_dataset(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.0,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        assert trainer.val_dataset is None

    def test_train_with_val_split_returns_train_losses(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        losses = Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert len(losses) == 2
        assert all(isinstance(v, float) and torch.tensor(v).isfinite() for v in losses)


class TestTrainerGroupSplit:
    """`groups=` keeps RetroRules radius-siblings out of both sides of the split."""

    def test_defaults_to_random_split_strategy(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        assert trainer.split_strategy == "random"

    def test_group_split_records_strategy(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        groups = [i // 4 for i in range(len(dataset))]  # 4 groups of 4
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu", groups=groups)
        assert trainer.split_strategy == "group"

    def test_no_group_straddles_train_val(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        groups = [i // 4 for i in range(len(dataset))]  # 4 groups of 4
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu", groups=groups)
        train_groups = {groups[i] for i in trainer.train_dataset.indices}
        val_groups = {groups[i] for i in trainer.val_dataset.indices}
        assert train_groups.isdisjoint(val_groups)
        assert len(trainer.train_dataset) + len(trainer.val_dataset) == len(dataset)

    def test_group_split_deterministic_given_seed(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        groups = [i // 4 for i in range(len(dataset))]
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        t1 = Trainer(
            model, dataset, collator, cfg, device="cpu", groups=groups, split_seed=7
        )
        t2 = Trainer(
            model, dataset, collator, cfg, device="cpu", groups=groups, split_seed=7
        )
        assert list(t1.train_dataset.indices) == list(t2.train_dataset.indices)
        assert list(t1.val_dataset.indices) == list(t2.val_dataset.indices)

    def test_group_split_saved_in_config_json(self, tiny_setup, tmp_path):
        import json

        model, dataset, collator = tiny_setup
        groups = [i // 4 for i in range(len(dataset))]
        output = tmp_path / "model.pt"
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(output),
        )
        Trainer(model, dataset, collator, cfg, device="cpu", groups=groups).train()
        saved = json.loads(output.with_suffix(".json").read_text())
        assert saved["split_strategy"] == "group"


class TestTrainerSchedulerAndClipping:
    def test_warmup_steps_runs_without_error(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            warmup_steps=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        losses = Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert len(losses) == 2

    def test_grad_clipping_disabled_with_zero(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            max_grad_norm=0.0,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        losses = Trainer(model, dataset, collator, cfg, device="cpu").train()
        assert all(torch.tensor(v).isfinite() for v in losses)


class TestTrainerAccuracy:
    def test_val_epoch_returns_loss_and_accuracies(self, tiny_setup, tmp_path):
        model, dataset, collator = tiny_setup
        cfg = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(tmp_path / "model.pt"),
        )
        trainer = Trainer(model, dataset, collator, cfg, device="cpu")
        from torch.utils.data import DataLoader

        val_loader = DataLoader(trainer.val_dataset, batch_size=4, collate_fn=collator)
        loss, top1, top5, content_top1, content_top5 = trainer._run_val_epoch(
            val_loader, epoch=1
        )
        assert isinstance(loss, float) and torch.tensor(loss).isfinite()
        assert 0.0 <= top1 <= 1.0
        assert 0.0 <= top5 <= 1.0
        assert top5 >= top1
        assert 0.0 <= content_top1 <= 1.0
        assert 0.0 <= content_top5 <= 1.0

    def test_accuracies_saved_in_config_json(self, tiny_setup, tmp_path):
        import json

        model, dataset, collator = tiny_setup
        output = tmp_path / "model.pt"
        cfg = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            val_split=0.25,
            checkpoint_dir=str(tmp_path / "ckpts"),
            output_path=str(output),
        )
        Trainer(model, dataset, collator, cfg, device="cpu").train()
        saved = json.loads(output.with_suffix(".json").read_text())
        history = saved["history"]
        assert len(history["val_top1_accuracy"]) == 2
        assert len(history["val_top5_accuracy"]) == 2
        assert all(0.0 <= v <= 1.0 for v in history["val_top1_accuracy"])
        assert all(0.0 <= v <= 1.0 for v in history["val_top5_accuracy"])
        assert len(history["val_content_top1_accuracy"]) == 2
        assert len(history["val_content_top5_accuracy"]) == 2
        assert all(0.0 <= v <= 1.0 for v in history["val_content_top1_accuracy"])
        assert all(0.0 <= v <= 1.0 for v in history["val_content_top5_accuracy"])
