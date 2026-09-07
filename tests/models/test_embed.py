import json

import numpy as np
import pytest
import torch

from rxn_smarts_embeddings.models.embed import SmartsEmbedder
from rxn_smarts_embeddings.models.smarts_transformer import (
    SmartsMLMModel,
    TransformerConfig,
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

_SMARTS = [
    "[C:1]-[O:2]>>[C:1]=[O:2]",
    "c1ccccc1>>c1cccnc1",
    "[C:1]=[O:2]>>[C:1]-[O:2]",
    "[O:2]-[C:1]>>[O:2]=[C:1]",
]

MAX_LEN = 16


@pytest.fixture
def config():
    return TransformerConfig(
        vocab_size=len(_VOCAB) + 1,  # +1 for [MASK]
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=MAX_LEN,
    )


@pytest.fixture
def embedder(config):
    model = SmartsMLMModel(config)
    return SmartsEmbedder(model, _VOCAB, pooling="cls", device="cpu")


@pytest.fixture
def embedder_mean(config):
    model = SmartsMLMModel(config)
    return SmartsEmbedder(model, _VOCAB, pooling="mean", device="cpu")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_invalid_pooling_raises(self, config):
        model = SmartsMLMModel(config)
        with pytest.raises(ValueError, match="pooling"):
            SmartsEmbedder(model, _VOCAB, pooling="bad")

    def test_model_set_to_eval(self, embedder):
        assert not embedder.model.training

    def test_model_on_correct_device(self, embedder):
        assert embedder.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


class TestEmbedOutputs:
    def test_shape_cls(self, embedder, config):
        out = embedder.embed(_SMARTS, batch_size=2)
        assert out.shape == (len(_SMARTS), config.d_model)

    def test_shape_mean(self, embedder_mean, config):
        out = embedder_mean.embed(_SMARTS, batch_size=2)
        assert out.shape == (len(_SMARTS), config.d_model)

    def test_dtype_is_float32(self, embedder):
        out = embedder.embed(_SMARTS)
        assert out.dtype == np.float32

    def test_count_equals_input(self, embedder):
        out = embedder.embed(_SMARTS)
        assert len(out) == len(_SMARTS)

    def test_single_smarts(self, embedder, config):
        out = embedder.embed([_SMARTS[0]])
        assert out.shape == (1, config.d_model)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self, embedder):
        out1 = embedder.embed(_SMARTS)
        out2 = embedder.embed(_SMARTS)
        np.testing.assert_array_equal(out1, out2)

    def test_batch_size_does_not_change_result(self, embedder):
        out1 = embedder.embed(_SMARTS, batch_size=1)
        out2 = embedder.embed(_SMARTS, batch_size=4)
        np.testing.assert_allclose(out1, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# Pooling correctness
# ---------------------------------------------------------------------------


class TestPooling:
    def test_cls_and_mean_differ(self, config):
        model = SmartsMLMModel(config)
        e_cls = SmartsEmbedder(model, _VOCAB, pooling="cls", device="cpu")
        e_mean = SmartsEmbedder(model, _VOCAB, pooling="mean", device="cpu")
        assert not np.allclose(e_cls.embed(_SMARTS), e_mean.embed(_SMARTS))

    def test_mean_pooling_not_all_zeros(self, embedder_mean):
        out = embedder_mean.embed(_SMARTS)
        assert not np.allclose(out, 0)


# ---------------------------------------------------------------------------
# from_checkpoint
# ---------------------------------------------------------------------------


def _write_checkpoint(config, tmp_path):
    model = SmartsMLMModel(config)
    weights_path = tmp_path / "model.pt"
    config_path = tmp_path / "model.json"
    vocab_path = tmp_path / "vocab.json"

    torch.save(model.state_dict(), weights_path)
    config_path.write_text(
        json.dumps({"model_config": config.to_dict(), "mask_id": len(_VOCAB)})
    )
    vocab_path.write_text(
        json.dumps(
            {
                "token_to_id": _VOCAB,
                "id_to_token": {},
                "frequencies": {},
                "size": len(_VOCAB),
            }
        )
    )
    return model, weights_path, config_path, vocab_path


class TestFromCheckpoint:
    def test_loads_and_embeds(self, config, tmp_path):
        _, weights, cfg, vocab = _write_checkpoint(config, tmp_path)
        embedder = SmartsEmbedder.from_checkpoint(
            str(weights), str(cfg), str(vocab), device="cpu"
        )
        out = embedder.embed(_SMARTS)
        assert out.shape == (len(_SMARTS), config.d_model)

    def test_loaded_weights_match_original(self, config, tmp_path):
        model, weights, cfg, vocab = _write_checkpoint(config, tmp_path)
        direct = SmartsEmbedder(model, _VOCAB, pooling="cls", device="cpu")
        loaded = SmartsEmbedder.from_checkpoint(
            str(weights), str(cfg), str(vocab), pooling="cls", device="cpu"
        )
        np.testing.assert_allclose(
            direct.embed(_SMARTS), loaded.embed(_SMARTS), atol=1e-5
        )
