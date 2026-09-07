"""Tests for smart_rxn_embeddings.predict."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smart_rxn_embeddings.predict import (
    _find_latest_checkpoint,
    _infer_format,
    _write_output,
    load_embedder,
    predict,
    main,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_D = 16  # small embedding dimension for tests
_SMARTS = ["[C:1]-[O:2]>>[C:1]=[O:2]", "c1ccccc1>>c1cccnc1"]


def _fake_embedder(n: int = len(_SMARTS), d: int = _D) -> MagicMock:
    """Return a mock embedder whose .embed() returns deterministic float32 data."""
    emb = MagicMock()
    emb.embed.return_value = np.ones((n, d), dtype=np.float32)
    return emb


# ---------------------------------------------------------------------------
# _find_latest_checkpoint
# ---------------------------------------------------------------------------


class TestFindLatestCheckpoint:
    def test_finds_most_recent(self, tmp_path):
        for name in (
            "smarts_transformer_20260101_000000",
            "smarts_transformer_20260201_000000",
        ):
            (tmp_path / f"{name}.pt").touch()
            (tmp_path / f"{name}.json").write_text("{}")
        weights, config = _find_latest_checkpoint(str(tmp_path))
        assert "20260201" in weights
        assert "20260201" in config

    def test_raises_when_no_checkpoints(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No checkpoints"):
            _find_latest_checkpoint(str(tmp_path))

    def test_raises_when_config_missing(self, tmp_path):
        (tmp_path / "smarts_transformer_20260101_000000.pt").touch()
        # no .json companion
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _find_latest_checkpoint(str(tmp_path))

    def test_returns_str_paths(self, tmp_path):
        (tmp_path / "smarts_transformer_20260101_000000.pt").touch()
        (tmp_path / "smarts_transformer_20260101_000000.json").write_text("{}")
        weights, config = _find_latest_checkpoint(str(tmp_path))
        assert isinstance(weights, str)
        assert isinstance(config, str)


# ---------------------------------------------------------------------------
# _infer_format
# ---------------------------------------------------------------------------


class TestInferFormat:
    def test_explicit_format_wins(self):
        assert _infer_format("out.json", "csv") == "csv"

    def test_infer_from_npy_extension(self):
        assert _infer_format("embeddings.npy", None) == "npy"

    def test_infer_from_csv_extension(self):
        assert _infer_format("out.csv", None) == "csv"

    def test_infer_from_json_extension(self):
        assert _infer_format("result.json", None) == "json"

    def test_defaults_to_json_with_no_output(self):
        assert _infer_format(None, None) == "json"

    def test_defaults_to_json_with_unknown_extension(self):
        assert _infer_format("out.txt", None) == "json"


# ---------------------------------------------------------------------------
# _write_output
# ---------------------------------------------------------------------------


class TestWriteOutputNpy:
    def test_saves_npy_file(self, tmp_path):
        out = tmp_path / "emb.npy"
        embs = np.zeros((2, _D), dtype=np.float32)
        _write_output(_SMARTS, embs, str(out), "npy")
        loaded = np.load(out)
        np.testing.assert_array_equal(loaded, embs)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "emb.npy"
        _write_output(_SMARTS, np.zeros((2, _D), dtype=np.float32), str(out), "npy")
        assert out.exists()


class TestWriteOutputJson:
    def _capture_stdout(self, smarts_list, embeddings):
        captured = StringIO()
        with patch("sys.stdout", captured):
            _write_output(smarts_list, embeddings, None, "json")
        return captured.getvalue()

    def test_stdout_is_valid_json(self):
        embs = np.ones((2, _D), dtype=np.float32)
        raw = self._capture_stdout(_SMARTS, embs)
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_json_structure(self):
        embs = np.ones((2, _D), dtype=np.float32)
        raw = self._capture_stdout(_SMARTS, embs)
        item = json.loads(raw)[0]
        assert set(item.keys()) == {"smarts", "embedding"}
        assert len(item["embedding"]) == _D

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "result.json"
        embs = np.ones((2, _D), dtype=np.float32)
        _write_output(_SMARTS, embs, str(out), "json")
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_smarts_aligned_with_embeddings(self):
        embs = np.array([[1.0] * _D, [2.0] * _D], dtype=np.float32)
        raw = self._capture_stdout(_SMARTS, embs)
        items = json.loads(raw)
        assert items[0]["smarts"] == _SMARTS[0]
        assert items[1]["smarts"] == _SMARTS[1]


class TestWriteOutputCsv:
    def _capture_stdout(self, smarts_list, embeddings):
        captured = StringIO()
        with patch("sys.stdout", captured):
            _write_output(smarts_list, embeddings, None, "csv")
        return captured.getvalue()

    def test_header_row(self):
        embs = np.zeros((2, _D), dtype=np.float32)
        csv_text = self._capture_stdout(_SMARTS, embs)
        header = csv_text.splitlines()[0]
        assert header.startswith("smarts,dim_0")

    def test_row_count(self):
        embs = np.zeros((2, _D), dtype=np.float32)
        csv_text = self._capture_stdout(_SMARTS, embs)
        lines = [line for line in csv_text.splitlines() if line]
        assert len(lines) == len(_SMARTS) + 1  # header + data rows

    def test_saves_to_file(self, tmp_path):
        out = tmp_path / "out.csv"
        embs = np.zeros((2, _D), dtype=np.float32)
        _write_output(_SMARTS, embs, str(out), "csv")
        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == len(_SMARTS) + 1

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            _write_output(_SMARTS, np.zeros((2, _D)), None, "parquet")


# ---------------------------------------------------------------------------
# load_embedder
# ---------------------------------------------------------------------------


class TestLoadEmbedder:
    @patch("smart_rxn_embeddings.predict._find_latest_checkpoint")
    @patch("smart_rxn_embeddings.predict.SmartsEmbedder.from_checkpoint")
    def test_auto_discovers_when_paths_none(self, mock_fc, mock_find, tmp_path):
        mock_find.return_value = ("w.pt", "c.json")
        mock_fc.return_value = MagicMock()
        load_embedder(weights=None, config=None, vocab=str(tmp_path / "v.json"))
        mock_find.assert_called_once()
        mock_fc.assert_called_once_with(
            "w.pt", "c.json", str(tmp_path / "v.json"), "cls", None
        )

    @patch("smart_rxn_embeddings.predict.SmartsEmbedder.from_checkpoint")
    def test_skips_autodiscovery_when_paths_given(self, mock_fc, tmp_path):
        mock_fc.return_value = MagicMock()
        load_embedder(weights="w.pt", config="c.json", vocab="v.json")
        mock_fc.assert_called_once_with("w.pt", "c.json", "v.json", "cls", None)

    @patch("smart_rxn_embeddings.predict.SmartsEmbedder.from_checkpoint")
    def test_uses_default_vocab_when_none(self, mock_fc):
        from smart_rxn_embeddings.predict import _DEFAULT_VOCAB

        mock_fc.return_value = MagicMock()
        load_embedder(weights="w.pt", config="c.json")
        assert mock_fc.call_args[0][2] == _DEFAULT_VOCAB


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_single_string_returns_1d(self, mock_le):
        mock_le.return_value = _fake_embedder(n=1, d=_D)
        result = predict(_SMARTS[0])
        assert result.ndim == 1
        assert result.shape == (_D,)

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_list_returns_2d(self, mock_le):
        mock_le.return_value = _fake_embedder(n=len(_SMARTS), d=_D)
        result = predict(_SMARTS)
        assert result.ndim == 2
        assert result.shape == (len(_SMARTS), _D)

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_passes_kwargs_to_embedder(self, mock_le):
        fake = _fake_embedder(n=1, d=_D)
        mock_le.return_value = fake
        predict(_SMARTS[0], batch_size=8, max_length=64)
        fake.embed.assert_called_once_with([_SMARTS[0]], batch_size=8, max_length=64)


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_positional_smarts(self, mock_le, capsys):
        mock_le.return_value = _fake_embedder()
        main([_SMARTS[0], _SMARTS[1]])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_file_input(self, mock_le, tmp_path, capsys):
        mock_le.return_value = _fake_embedder()
        f = tmp_path / "smarts.txt"
        f.write_text("\n".join(_SMARTS))
        main(["--file", str(f)])
        out = capsys.readouterr().out
        assert len(json.loads(out)) == len(_SMARTS)

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_stdin_input(self, mock_le, monkeypatch, capsys):
        mock_le.return_value = _fake_embedder(n=1)
        monkeypatch.setattr("sys.stdin", StringIO(_SMARTS[0] + "\n"))
        main([])
        out = capsys.readouterr().out
        assert len(json.loads(out)) == 1

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_output_npy(self, mock_le, tmp_path):
        mock_le.return_value = _fake_embedder()
        out = tmp_path / "emb.npy"
        main([*_SMARTS, "--output", str(out)])
        assert out.exists()
        arr = np.load(out)
        assert arr.shape == (len(_SMARTS), _D)

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_output_csv(self, mock_le, tmp_path):
        mock_le.return_value = _fake_embedder()
        out = tmp_path / "emb.csv"
        main([*_SMARTS, "--output", str(out)])
        lines = out.read_text().splitlines()
        assert lines[0].startswith("smarts,dim_0")
        assert len(lines) == len(_SMARTS) + 1

    @patch("smart_rxn_embeddings.predict.load_embedder")
    def test_output_json_file(self, mock_le, tmp_path):
        mock_le.return_value = _fake_embedder()
        out = tmp_path / "emb.json"
        main([*_SMARTS, "--output", str(out)])
        data = json.loads(out.read_text())
        assert len(data) == len(_SMARTS)

    def test_error_on_both_positional_and_file(self, tmp_path):
        f = tmp_path / "s.txt"
        f.write_text(_SMARTS[0])
        with pytest.raises(SystemExit):
            main([_SMARTS[0], "--file", str(f)])

    def test_error_npy_without_output(self):
        with pytest.raises(SystemExit):
            main([_SMARTS[0], "--format", "npy"])

    def test_error_no_input(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", StringIO(""))
        with pytest.raises(SystemExit):
            main([])
