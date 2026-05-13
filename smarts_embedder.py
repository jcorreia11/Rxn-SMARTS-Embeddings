"""Minimal self-contained embedder for reaction SMARTS.

Three files are required at runtime — upload them alongside this script:
    smarts_transformer_<RUN_ID>.pt    model weights (~28 MB)
    smarts_transformer_<RUN_ID>.json  model config
    vocab.json                        vocabulary

Dependencies: torch, numpy  (no RDKit, no sentencepiece)

Quick start
-----------
    embedder = SmartsEmbedder.from_checkpoint(
        weights_path="smarts_transformer_20260410_111830.pt",
        config_path="smarts_transformer_20260410_111830.json",
        vocab_path="vocab.json",
        pooling="mean",   # "mean" or "cls"
    )
    smarts = ["[C:1]>>[C:1]O", "[CH2:1][OH:2]>>[CH2:1]=[O:2]"]
    emb = embedder.embed(smarts)   # numpy array, shape (2, 256), dtype float32
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_REST_TOKEN_RE = re.compile(
    r">>"
    r"|Cl|Br"
    r"|[BCNOPSFIbcnopsAa*]"
    r"|/\?|\\\?"
    r"|[-=#~:@/\\]"
    r"|\(|\)"
    r"|\."
    r"|%\d{2}"
    r"|\d"
    r"|[!&,;]"
    r"|>"
)


def _tokenize(smarts: str) -> list[str]:
    """Split a reaction SMARTS string into chemically meaningful tokens."""
    tokens: list[str] = []
    i = 0
    n = len(smarts)
    while i < n:
        if smarts[i] == "[":
            # find matching closing bracket, tracking depth for recursive SMARTS
            depth = 0
            for j in range(i, n):
                if smarts[j] == "[":
                    depth += 1
                elif smarts[j] == "]":
                    depth -= 1
                    if depth == 0:
                        tokens.append(smarts[i : j + 1])
                        i = j + 1
                        break
            else:
                raise ValueError(f"Unmatched '[' at position {i} in {smarts!r}")
        else:
            m = _REST_TOKEN_RE.match(smarts, i)
            if m is None:
                raise ValueError(
                    f"Unexpected character {smarts[i]!r} at position {i} in {smarts!r}"
                )
            tokens.append(m.group())
            i = m.end()
    return tokens


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

_PAD_ID = 0
_UNK_ID = 1
_BOS_ID = 2


@dataclass
class _Config:
    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    pad_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "_Config":
        return cls(**d)


class _Encoder(nn.Module):
    def __init__(self, cfg: _Config) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.num_encoder_layers,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids) + self.pos_embedding(positions))
        return self.encoder(x, src_key_padding_mask=(attention_mask == 0))


class _MLMHead(nn.Module):
    def __init__(self, cfg: _Config) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.d_model, cfg.d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.decoder(self.layer_norm(self.act(self.dense(hidden))))


class _Model(nn.Module):
    def __init__(self, cfg: _Config) -> None:
        super().__init__()
        self.config = cfg
        self.encoder = _Encoder(cfg)
        self.mlm_head = _MLMHead(cfg)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.mlm_head(self.encoder(input_ids, attention_mask))


# ---------------------------------------------------------------------------
# Dataset (encoding + padding, no PyTorch dependency beyond Dataset/DataLoader)
# ---------------------------------------------------------------------------

class _SMARTSDataset(Dataset):
    def __init__(
        self,
        smarts_list: list[str],
        token_to_id: dict[str, int],
        max_length: int,
        add_bos: bool,
    ) -> None:
        self._max_length = max_length
        self._encoded: list[list[int]] = []
        for smarts in smarts_list:
            try:
                ids = [token_to_id.get(t, _UNK_ID) for t in _tokenize(smarts)]
                if add_bos:
                    ids = [_BOS_ID] + ids
            except Exception:
                ids = []
            self._encoded.append(ids)

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = self._encoded[idx][: self._max_length]
        pad = self._max_length - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [_PAD_ID] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SmartsEmbedder:
    """Embed reaction SMARTS strings using a pretrained transformer.

    Parameters
    ----------
    pooling:
        ``"mean"`` (recommended) — average of non-padding hidden states.
        ``"cls"``  — BOS token at position 0 (BERT-style).
    device:
        ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.
    """

    def __init__(
        self,
        model: _Model,
        token_to_id: dict[str, int],
        pooling: str = "mean",
        device: str | None = None,
    ) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        self._model = model
        self._token_to_id = token_to_id
        self._pooling = pooling
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model.to(self._device).eval()

    @classmethod
    def from_checkpoint(
        cls,
        weights_path: str,
        config_path: str,
        vocab_path: str,
        pooling: str = "mean",
        device: str | None = None,
    ) -> "SmartsEmbedder":
        """Load from the three artifact files."""
        cfg_raw = json.loads(Path(config_path).read_text())
        model = _Model(_Config.from_dict(cfg_raw["model_config"]))
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)

        vocab = json.loads(Path(vocab_path).read_text())
        return cls(model, vocab["token_to_id"], pooling, device)

    def embed(
        self,
        smarts_list: list[str],
        batch_size: int = 64,
        max_length: int | None = None,
    ) -> np.ndarray:
        """Return a ``(N, d_model)`` float32 array of embeddings.

        Parameters
        ----------
        smarts_list:
            Reaction SMARTS strings to embed.
        batch_size:
            Sequences per forward pass.
        max_length:
            Pad/truncate to this length. Defaults to the model's ``max_seq_len``.
        """
        max_len = max_length or self._model.config.max_seq_len
        add_bos = self._pooling == "cls"
        dataset = _SMARTSDataset(smarts_list, self._token_to_id, max_len, add_bos)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                hidden = self._model.encoder(input_ids, attention_mask)  # (B, L, d)
                emb = self._pool(hidden, attention_mask)                  # (B, d)
                chunks.append(emb.cpu().float().numpy())

        return np.concatenate(chunks, axis=0)

    def _pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        if self._pooling == "cls":
            return hidden[:, 0, :]
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)