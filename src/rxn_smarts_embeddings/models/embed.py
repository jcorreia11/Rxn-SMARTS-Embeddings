"""Extract fixed-size reaction embeddings from a pretrained SmartsMLMModel."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from rxn_smarts_embeddings.datasets.smarts_dataset import SMARTSDataset
from rxn_smarts_embeddings.models.smarts_transformer import (
    SmartsMLMModel,
    TransformerConfig,
)

logger = logging.getLogger(__name__)

_POOLING_OPTIONS = ("cls", "mean")


class SmartsEmbedder:
    """Extract fixed-size embeddings from a pretrained :class:`SmartsMLMModel`.

    Two pooling strategies are supported:

    ``"mean"`` (default, recommended)
        Mean of all non-padding token hidden states. Validated as strictly
        better than CLS pooling across all model sizes and classifier heads
        (span-masked pretraining imposes no sequence-level aggregation loss
        on the ``[BOS]`` position).
    ``"cls"``
        Uses the ``[BOS]`` token at position 0 as a sequence-level representation
        (BERT-style CLS pooling).  ``add_bos=True`` is set automatically.

    Parameters
    ----------
    model:
        Pretrained :class:`SmartsMLMModel`.
    token_to_id:
        Vocabulary mapping (as loaded from ``vocab.json``).
    pooling:
        ``"cls"`` or ``"mean"`` (default ``"mean"`` — validated as strictly
        better than CLS pooling across all model sizes and classifier heads).
    device:
        ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.

    Examples
    --------
    >>> embedder = SmartsEmbedder.from_checkpoint(
    ...     "models/smarts_transformer.pt",
    ...     "models/smarts_transformer.json",
    ...     "data/processed/vocab.json",
    ... )
    >>> embeddings = embedder.embed(smarts_list)  # (N, d_model) float32
    """

    def __init__(
        self,
        model: SmartsMLMModel,
        token_to_id: dict[str, int],
        pooling: str = "mean",
        device: str | None = None,
    ) -> None:
        if pooling not in _POOLING_OPTIONS:
            raise ValueError(
                f"pooling must be one of {_POOLING_OPTIONS}, got {pooling!r}"
            )

        self.model = model
        self.token_to_id = token_to_id
        self.pooling = pooling
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        weights_path: str,
        config_path: str,
        vocab_path: str,
        pooling: str = "mean",
        device: str | None = None,
    ) -> "SmartsEmbedder":
        """Load a :class:`SmartsEmbedder` from saved checkpoint files.

        Parameters
        ----------
        weights_path:
            Path to ``smarts_transformer_<RUN_ID>.pt``.
        config_path:
            Path to the companion ``smarts_transformer_<RUN_ID>.json``.
        vocab_path:
            Path to ``vocab.json`` produced by ``build_vocab.py``.
        pooling:
            Pooling strategy — ``"cls"`` or ``"mean"``.
        device:
            ``"cuda"`` / ``"cpu"`` — auto-detected when ``None``.
        """
        cfg = json.loads(Path(config_path).read_text())
        model_config = TransformerConfig.from_dict(cfg["model_config"])
        model = SmartsMLMModel(model_config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        logger.info(
            "Loaded model from %s (d_model=%d, layers=%d)",
            weights_path,
            model_config.d_model,
            model_config.num_encoder_layers,
        )

        vocab = json.loads(Path(vocab_path).read_text())
        token_to_id: dict[str, int] = vocab["token_to_id"]

        return cls(model, token_to_id, pooling, device)

    def embed(
        self,
        smarts_list: list[str],
        batch_size: int = 64,
        max_length: int | None = None,
    ) -> np.ndarray:
        """Embed *smarts_list* and return an ``(N, d_model)`` float32 array.

        Parameters
        ----------
        smarts_list:
            Reaction SMARTS strings to embed.
        batch_size:
            Number of sequences per forward pass.
        max_length:
            Pad/truncate to this length.  Defaults to ``model.config.max_seq_len``.

        Returns
        -------
        numpy.ndarray of shape ``(len(smarts_list), d_model)``, dtype float32.
        """
        add_bos = self.pooling == "cls"
        dataset = SMARTSDataset(
            smarts_list,
            self.token_to_id,
            max_length=max_length or self.model.config.max_seq_len,
            add_bos=add_bos,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                hidden = self.model.encoder(
                    input_ids, attention_mask
                )  # (B, L, d_model)
                emb = self._pool(hidden, attention_mask)  # (B, d_model)
                chunks.append(emb.cpu().float().numpy())

        result = np.concatenate(chunks, axis=0)
        logger.info(
            "Extracted %d embeddings | dim=%d | pooling=%s | device=%s",
            result.shape[0],
            result.shape[1],
            self.pooling,
            self.device,
        )
        return result

    def _pool(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        # mean pooling over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
