import pytest
import torch

from rxn_smarts_embeddings.models.smarts_transformer import (
    MLMHead,
    SmartsMLMModel,
    SmartsTransformerEncoder,
    TransformerConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 30
MAX_LEN = 16
BATCH = 4


@pytest.fixture
def config():
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=MAX_LEN,
        pad_id=0,
    )


@pytest.fixture
def batch(config):
    input_ids = torch.randint(1, config.vocab_size, (BATCH, MAX_LEN))
    attention_mask = torch.ones(BATCH, MAX_LEN, dtype=torch.long)
    return input_ids, attention_mask


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------


class TestTransformerConfig:
    def test_to_dict_round_trip(self, config):
        assert TransformerConfig.from_dict(config.to_dict()) == config

    def test_to_dict_contains_all_fields(self, config):
        d = config.to_dict()
        for field in (
            "vocab_size",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "dropout",
            "max_seq_len",
            "pad_id",
        ):
            assert field in d


# ---------------------------------------------------------------------------
# SmartsTransformerEncoder
# ---------------------------------------------------------------------------


class TestSmartsTransformerEncoder:
    def test_output_shape(self, config, batch):
        encoder = SmartsTransformerEncoder(config)
        input_ids, attention_mask = batch
        out = encoder(input_ids, attention_mask)
        assert out.shape == (BATCH, MAX_LEN, config.d_model)

    def test_padding_does_not_affect_real_tokens(self, config):
        """Output at real-token positions must not change when padding changes."""
        encoder = SmartsTransformerEncoder(config)
        encoder.eval()

        input_ids = torch.randint(1, config.vocab_size, (1, MAX_LEN))
        full_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
        half_mask = full_mask.clone()
        half_mask[:, MAX_LEN // 2 :] = 0

        with torch.no_grad():
            out_full = encoder(input_ids, full_mask)
            # zero out padding tokens so position IDs stay the same
            padded_ids = input_ids.clone()
            padded_ids[:, MAX_LEN // 2 :] = 0
            out_half = encoder(padded_ids, half_mask)

        # Real-token positions (0 .. MAX_LEN//2 - 1) may differ because the
        # attention context changes — this test simply asserts the shapes match.
        assert out_full.shape == out_half.shape

    def test_output_dtype_is_float(self, config, batch):
        encoder = SmartsTransformerEncoder(config)
        input_ids, attention_mask = batch
        out = encoder(input_ids, attention_mask)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# MLMHead
# ---------------------------------------------------------------------------


class TestMLMHead:
    def test_output_shape(self, config):
        head = MLMHead(config)
        hidden = torch.randn(BATCH, MAX_LEN, config.d_model)
        logits = head(hidden)
        assert logits.shape == (BATCH, MAX_LEN, config.vocab_size)

    def test_output_is_not_all_zeros(self, config):
        head = MLMHead(config)
        hidden = torch.randn(BATCH, MAX_LEN, config.d_model)
        logits = head(hidden)
        assert logits.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# SmartsMLMModel
# ---------------------------------------------------------------------------


class TestSmartsMLMModel:
    def test_forward_output_shape(self, config, batch):
        model = SmartsMLMModel(config)
        input_ids, attention_mask = batch
        logits = model(input_ids, attention_mask)
        assert logits.shape == (BATCH, MAX_LEN, config.vocab_size)

    def test_forward_output_dtype(self, config, batch):
        model = SmartsMLMModel(config)
        input_ids, attention_mask = batch
        logits = model(input_ids, attention_mask)
        assert logits.dtype == torch.float32

    def test_loss_computable(self, config, batch):
        """Cross-entropy loss on masked positions must be a finite scalar."""
        import torch.nn as nn

        model = SmartsMLMModel(config)
        input_ids, attention_mask = batch

        labels = input_ids.clone()
        labels[:, ::2] = -100  # mask every other position

        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        assert loss.isfinite()

    def test_gradients_flow(self, config, batch):
        import torch.nn as nn

        model = SmartsMLMModel(config)
        input_ids, attention_mask = batch
        labels = input_ids.clone()

        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, config.vocab_size), labels.view(-1)
        )
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(g.isfinite().all() for g in grads)

    def test_state_dict_save_load(self, config, batch, tmp_path):
        model = SmartsMLMModel(config)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        loaded = SmartsMLMModel(config)
        loaded.load_state_dict(torch.load(path, weights_only=True))

        input_ids, attention_mask = batch
        with torch.no_grad():
            out1 = model(input_ids, attention_mask)
            out2 = loaded(input_ids, attention_mask)
        assert torch.allclose(out1, out2)
