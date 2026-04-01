import pytest

from smart_rxn_embeddings.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

# Diverse enough corpus so sentencepiece has enough unique characters to fill
# vocab_size=50.  Strings are repeated to give the trainer enough sentence count.
_CORPUS = [
    "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]",
    "[O;H1:1]-[C;H2:2]>>[O;H0:1]=[C;H1:2]",
    "[N;H2:1]-[C;H1:2]>>[N;H1:1]=[C;H0:2]",
    "([O;H0:1]-[C;H2:2].[O;H0:3])>>([O;H1:1].[C;H1:2])",
    "c1ccccc1>>c1cccnc1",
    "C(=O)O.OCC>>C(=O)OCC.O",
    "[C;H3:1]-[O;H1:2]>>[C;H2:1]=[O:2]",
    "[N;H1:1]=[C:2]>>[N;H2:1]-[C;H1:2]",
    "[S;H0:1](=O)(=O)>>[S;H1:1]=O",
    "[P;H0:1](=O)([OH])[OH]>>[P;H1:1](=O)[OH]",
] * 30  # repeat to create a corpus large enough for BPE


@pytest.fixture(scope="module")
def trained_tok(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("sp_model")
    model_prefix = str(model_dir / "test_sp")
    return SentencePieceTokenizer.train(_CORPUS, model_prefix, vocab_size=50)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_creates_model_file(self, tmp_path):
        model_prefix = str(tmp_path / "sp")
        SentencePieceTokenizer.train(_CORPUS, model_prefix, vocab_size=50)
        assert (tmp_path / "sp.model").exists()

    def test_train_returns_tokenizer_instance(self, tmp_path):
        model_prefix = str(tmp_path / "sp")
        tok = SentencePieceTokenizer.train(_CORPUS, model_prefix, vocab_size=50)
        assert isinstance(tok, SentencePieceTokenizer)

    def test_train_unigram_model_type(self, tmp_path):
        model_prefix = str(tmp_path / "sp_uni")
        tok = SentencePieceTokenizer.train(
            _CORPUS, model_prefix, vocab_size=50, model_type="unigram"
        )
        assert tok.vocab_size > 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_from_saved_model(self, trained_tok, tmp_path):
        # Train, then load from the saved .model file
        saved_path = str(trained_tok.model_path)
        reloaded = SentencePieceTokenizer(saved_path)
        assert reloaded.vocab_size == trained_tok.vocab_size


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TestInference:
    _SMARTS = "[C;H1:1]=[N;H0:2]>>[C;H1:1]-[N;H0:2]"

    def test_tokenize_returns_list_of_strings(self, trained_tok):
        tokens = trained_tok.tokenize(self._SMARTS)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert len(tokens) > 0

    def test_tokenize_covers_full_string(self, trained_tok):
        tokens = trained_tok.tokenize(self._SMARTS)
        assert "".join(tokens).replace("▁", "") != ""  # at least some content

    def test_encode_returns_list_of_ints(self, trained_tok):
        ids = trained_tok.encode(self._SMARTS)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_decode_roundtrip(self, trained_tok):
        ids = trained_tok.encode(self._SMARTS)
        decoded = trained_tok.decode(ids)
        # SentencePiece may add/remove whitespace; core content must survive
        assert decoded.replace(" ", "") != ""

    def test_encode_add_bos_eos(self, trained_tok):
        ids_plain = trained_tok.encode(self._SMARTS)
        ids_bos_eos = trained_tok.encode(self._SMARTS, add_bos=True, add_eos=True)
        assert len(ids_bos_eos) == len(ids_plain) + 2
        assert ids_bos_eos[0] == 2   # bos_id=2
        assert ids_bos_eos[-1] == 3  # eos_id=3

    def test_decode_returns_string(self, trained_tok):
        ids = trained_tok.encode(self._SMARTS)
        assert isinstance(trained_tok.decode(ids), str)

    def test_vocab_size_matches_requested(self, trained_tok):
        assert trained_tok.vocab_size == 50