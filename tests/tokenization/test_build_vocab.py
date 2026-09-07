import json

import pandas as pd

from rxn_smarts_embeddings.tokenization.build_vocab import (
    SPECIAL_TOKENS,
    build_vocab,
    main,
    save_vocab,
)

_SMARTS = [
    "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]",
    "[O;H1:1]-[C;H2:2]>>[O;H0:1]=[C;H1:2]",
    "[N;H2:1]-[C;H1:2]>>[N;H1:1]=[C;H0:2]",
]


# ---------------------------------------------------------------------------
# build_vocab
# ---------------------------------------------------------------------------


class TestBuildVocab:
    def test_returns_required_keys(self):
        vocab = build_vocab(_SMARTS)
        assert {"token_to_id", "id_to_token", "frequencies", "size"}.issubset(
            vocab.keys()
        )

    def test_special_tokens_have_fixed_ids(self):
        vocab = build_vocab(_SMARTS)
        t2i = vocab["token_to_id"]
        assert t2i["[PAD]"] == 0
        assert t2i["[UNK]"] == 1
        assert t2i["[BOS]"] == 2
        assert t2i["[EOS]"] == 3

    def test_token_to_id_and_id_to_token_are_inverse(self):
        vocab = build_vocab(_SMARTS)
        t2i = vocab["token_to_id"]
        i2t = vocab["id_to_token"]
        for tok, idx in t2i.items():
            assert i2t[str(idx)] == tok

    def test_size_equals_token_to_id_length(self):
        vocab = build_vocab(_SMARTS)
        assert vocab["size"] == len(vocab["token_to_id"])

    def test_most_frequent_token_gets_lowest_regular_id(self):
        # Repeat one SMARTS many times so its tokens dominate the frequency list
        smarts = ["[C:1]>>[N:1]"] * 20 + ["[O:1]>>[S:1]"]
        vocab = build_vocab(smarts)
        freqs = vocab["frequencies"]
        t2i = vocab["token_to_id"]
        most_common = next(iter(freqs))  # first key = highest frequency
        # Its ID must come right after the special tokens
        assert t2i[most_common] == len(SPECIAL_TOKENS)

    def test_frequencies_are_positive_integers(self):
        vocab = build_vocab(_SMARTS)
        for count in vocab["frequencies"].values():
            assert isinstance(count, int)
            assert count > 0

    def test_all_tokens_in_frequencies_are_in_token_to_id(self):
        vocab = build_vocab(_SMARTS)
        for tok in vocab["frequencies"]:
            assert tok in vocab["token_to_id"]

    def test_special_tokens_not_in_frequencies(self):
        vocab = build_vocab(_SMARTS)
        for sp in SPECIAL_TOKENS:
            assert sp not in vocab["frequencies"]

    def test_empty_input_returns_only_special_tokens(self):
        vocab = build_vocab([])
        assert vocab["size"] == len(SPECIAL_TOKENS)
        assert vocab["frequencies"] == {}

    def test_invalid_smarts_skipped_gracefully(self):
        smarts = ["[C:1]>>[N:1]", "not_valid_smarts_!!!"]
        vocab = build_vocab(smarts)
        # valid SMARTS should still produce tokens
        assert vocab["size"] > len(SPECIAL_TOKENS)

    def test_custom_special_tokens(self):
        custom = ["<PAD>", "<UNK>"]
        vocab = build_vocab(_SMARTS, special_tokens=custom)
        assert vocab["token_to_id"]["<PAD>"] == 0
        assert vocab["token_to_id"]["<UNK>"] == 1


# ---------------------------------------------------------------------------
# save_vocab
# ---------------------------------------------------------------------------


class TestSaveVocab:
    def test_creates_file(self, tmp_path):
        vocab = build_vocab(_SMARTS)
        out = tmp_path / "vocab.json"
        save_vocab(vocab, str(out))
        assert out.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        vocab = build_vocab(_SMARTS)
        out = tmp_path / "vocab.json"
        save_vocab(vocab, str(out))
        loaded = json.loads(out.read_text())
        assert "token_to_id" in loaded

    def test_creates_parent_directory(self, tmp_path):
        vocab = build_vocab(_SMARTS)
        out = tmp_path / "nested" / "dir" / "vocab.json"
        save_vocab(vocab, str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_produces_vocab_file(self, tmp_path):
        # Build a minimal validated_smarts.csv
        df = pd.DataFrame(
            {
                "smarts": _SMARTS + ["invalid???"],
                "valid": [True, True, True, False],
            }
        )
        input_path = tmp_path / "validated_smarts.csv"
        df.to_csv(input_path, index=False)
        output_path = tmp_path / "vocab.json"

        main(str(input_path), str(output_path))

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["size"] > len(SPECIAL_TOKENS)

    def test_main_excludes_invalid_smarts_rows(self, tmp_path):
        df = pd.DataFrame(
            {
                "smarts": ["[C:1]>>[N:1]", "[O:1]>>[S:1]"],
                "valid": [True, False],
            }
        )
        input_path = tmp_path / "validated_smarts.csv"
        df.to_csv(input_path, index=False)
        output_path = tmp_path / "vocab.json"

        main(str(input_path), str(output_path))

        loaded = json.loads(output_path.read_text())
        # Tokens from the False row ("[O:1]" and "[S:1]") should not appear
        assert "[O:1]" not in loaded["token_to_id"]
        assert "[S:1]" not in loaded["token_to_id"]
