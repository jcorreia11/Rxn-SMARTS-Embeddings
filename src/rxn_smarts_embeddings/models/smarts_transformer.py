"""Transformer encoder + MLM head for reaction SMARTS."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TransformerConfig:
    """Hyper-parameters for :class:`SmartsMLMModel`.

    Parameters
    ----------
    vocab_size:
        Total vocabulary size, **including** the injected ``[MASK]`` token.
    d_model:
        Token embedding / hidden dimension.
    nhead:
        Number of attention heads (must divide *d_model*).
    num_encoder_layers:
        Depth of the transformer encoder stack.
    dim_feedforward:
        Inner dimension of each encoder feed-forward block.
    dropout:
        Dropout probability applied to embeddings and encoder layers.
    max_seq_len:
        Maximum sequence length; determines positional embedding table size.
    pad_id:
        ID of the ``[PAD]`` token (embedding is zeroed, positions ignored).
    """

    vocab_size: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    pad_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**d)


class SmartsTransformerEncoder(nn.Module):
    """Token + positional embeddings followed by a stack of transformer encoder layers."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input_ids:      (B, L) long tensor of token IDs.
        attention_mask: (B, L) long tensor — 1 for real tokens, 0 for padding.

        Returns
        -------
        (B, L, d_model) hidden states.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        # TransformerEncoder expects True where tokens should be *ignored*
        padding_mask = attention_mask == 0
        return self.encoder(x, src_key_padding_mask=padding_mask)


class MLMHead(nn.Module):
    """Dense → GELU → LayerNorm → linear projection to vocabulary logits."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.decoder = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        """(B, L, d_model) → (B, L, vocab_size) logits."""
        return self.decoder(self.layer_norm(self.act(self.dense(hidden))))


class SmartsMLMModel(nn.Module):
    """Full masked-language model: encoder + MLM head."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SmartsTransformerEncoder(config)
        self.mlm_head = MLMHead(config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Returns (B, L, vocab_size) logits."""
        hidden = self.encoder(input_ids, attention_mask)
        return self.mlm_head(hidden)
