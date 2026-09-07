import json

import pytest
import torch

from rxn_smarts_embeddings.datasets.smarts_dataset import SMARTSDataset

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMARTS = [
    "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]",
    "[O;H1:1]-[C;H2:2]>>[O;H0:1]=[C;H1:2]",
    "c1ccccc1>>c1cccnc1",
    "[C:1]-[Br:2]>>[C:1]-[OH:2]",
]

_VOCAB: dict[str, int] = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "-": 4,
    "=": 5,
    ">>": 6,
    "(": 7,
    ")": 8,
    ":": 9,
    "[C;H1:1]": 10,
    "[N;H0:2]": 11,
    "[O;H1:1]": 12,
    "[C;H2:2]": 13,
    "[O;H0:1]": 14,
    "[C;H1:2]": 15,
    "c": 16,
    "1": 17,
    "n": 18,
    "[C:1]": 19,
    "[Br:2]": 20,
    "[OH:2]": 21,
}


@pytest.fixture
def dataset():
    return SMARTSDataset(_SMARTS, _VOCAB, max_length=32)


@pytest.fixture
def vocab_file(tmp_path):
    data = {
        "token_to_id": _VOCAB,
        "id_to_token": {str(v): k for k, v in _VOCAB.items()},
        "frequencies": {},
        "size": len(_VOCAB),
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# Length and indexing
# ---------------------------------------------------------------------------


class TestLengthAndIndexing:
    def test_len_equals_input(self, dataset):
        assert len(dataset) == len(_SMARTS)

    def test_getitem_returns_dict(self, dataset):
        assert isinstance(dataset[0], dict)

    def test_getitem_has_required_keys(self, dataset):
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item


# ---------------------------------------------------------------------------
# Tensor shapes and dtypes
# ---------------------------------------------------------------------------


class TestTensors:
    def test_input_ids_shape(self, dataset):
        assert dataset[0]["input_ids"].shape == torch.Size([32])

    def test_attention_mask_shape(self, dataset):
        assert dataset[0]["attention_mask"].shape == torch.Size([32])

    def test_input_ids_dtype(self, dataset):
        assert dataset[0]["input_ids"].dtype == torch.long

    def test_attention_mask_dtype(self, dataset):
        assert dataset[0]["attention_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestPadding:
    def test_padded_positions_have_zero_id(self, dataset):
        item = dataset[0]
        pad_positions = item["attention_mask"] == 0
        assert (item["input_ids"][pad_positions] == 0).all()

    def test_mask_is_binary(self, dataset):
        for i in range(len(dataset)):
            assert set(dataset[i]["attention_mask"].tolist()).issubset({0, 1})

    def test_padding_contiguous_at_end(self, dataset):
        for i in range(len(dataset)):
            mask = dataset[i]["attention_mask"].tolist()
            found_pad = False
            for v in mask:
                if v == 0:
                    found_pad = True
                if found_pad:
                    assert v == 0

    def test_at_least_one_real_token(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["attention_mask"].sum().item() > 0


# ---------------------------------------------------------------------------
# max_length
# ---------------------------------------------------------------------------


class TestMaxLength:
    def test_explicit_max_length_shapes(self):
        ds = SMARTSDataset(_SMARTS, _VOCAB, max_length=16)
        for i in range(len(ds)):
            assert ds[i]["input_ids"].shape == torch.Size([16])

    def test_default_max_length_is_longest_sequence(self):
        ds = SMARTSDataset(_SMARTS, _VOCAB)
        expected = max(len(ds._encoded[i]) for i in range(len(ds)))
        assert ds.max_length == expected

    def test_truncation_to_max_length(self):
        ds = SMARTSDataset(_SMARTS, _VOCAB, max_length=3)
        for i in range(len(ds)):
            assert ds[i]["input_ids"].shape == torch.Size([3])


# ---------------------------------------------------------------------------
# BOS / EOS
# ---------------------------------------------------------------------------


class TestBosEos:
    def test_add_bos_prepends_bos_id(self):
        ds = SMARTSDataset(_SMARTS, _VOCAB, max_length=32, add_bos=True)
        assert ds[0]["input_ids"][0].item() == 2

    def test_add_eos_at_last_real_position(self):
        ds = SMARTSDataset(_SMARTS, _VOCAB, max_length=32, add_eos=True)
        item = ds[0]
        real_len = item["attention_mask"].sum().item()
        assert item["input_ids"][real_len - 1].item() == 3

    def test_bos_eos_add_two_real_tokens(self):
        ds_plain = SMARTSDataset(_SMARTS[:1], _VOCAB)
        ds_bos_eos = SMARTSDataset(_SMARTS[:1], _VOCAB, add_bos=True, add_eos=True)
        plain_real = ds_plain[0]["attention_mask"].sum().item()
        bos_eos_real = ds_bos_eos[0]["attention_mask"].sum().item()
        assert bos_eos_real == plain_real + 2


# ---------------------------------------------------------------------------
# Unknown tokens
# ---------------------------------------------------------------------------


class TestUnknownTokens:
    def test_unknown_token_mapped_to_unk_id(self):
        def tokenizer_with_unknown(smarts: str) -> list[str]:
            return ["NOT_IN_VOCAB", ">>"]

        ds = SMARTSDataset(
            ["anything"], _VOCAB, tokenizer=tokenizer_with_unknown, max_length=16
        )
        assert 1 in ds[0]["input_ids"].tolist()


# ---------------------------------------------------------------------------
# from_vocab_file
# ---------------------------------------------------------------------------


class TestFromVocabFile:
    def test_creates_dataset_of_correct_length(self, vocab_file):
        ds = SMARTSDataset.from_vocab_file(_SMARTS, vocab_file, max_length=32)
        assert len(ds) == len(_SMARTS)

    def test_output_shapes_match_max_length(self, vocab_file):
        ds = SMARTSDataset.from_vocab_file(_SMARTS, vocab_file, max_length=32)
        assert ds[0]["input_ids"].shape == torch.Size([32])

    def test_token_to_id_loaded_correctly(self, vocab_file):
        ds = SMARTSDataset.from_vocab_file(_SMARTS, vocab_file, max_length=32)
        assert ds.token_to_id == _VOCAB


# ---------------------------------------------------------------------------
# Custom tokenizer
# ---------------------------------------------------------------------------


class TestCustomTokenizer:
    def test_custom_tokenizer_is_called(self):
        calls = []

        def recording_tokenizer(smarts: str) -> list[str]:
            calls.append(smarts)
            return list(smarts)

        SMARTSDataset(_SMARTS, _VOCAB, tokenizer=recording_tokenizer, max_length=32)
        assert len(calls) == len(_SMARTS)

    def test_custom_tokenizer_affects_real_length(self):
        def char_tokenizer(smarts: str) -> list[str]:
            return list(smarts)

        ds = SMARTSDataset(_SMARTS[:1], _VOCAB, tokenizer=char_tokenizer, max_length=64)
        real_len = ds[0]["attention_mask"].sum().item()
        assert real_len == min(len(_SMARTS[0]), 64)
