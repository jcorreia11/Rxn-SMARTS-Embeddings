"""PyTorch dataset for reaction SMARTS."""

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from rxn_smarts_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer

logger = logging.getLogger(__name__)

_PAD_ID = 0
_UNK_ID = 1
_BOS_ID = 2
_EOS_ID = 3


class SMARTSDataset(Dataset):
    """PyTorch dataset that tokenizes reaction SMARTS and returns padded tensors.

    Each item is a dict with:
        - ``input_ids``      (LongTensor [max_length]): token IDs, padded with 0
        - ``attention_mask`` (LongTensor [max_length]): 1 for real tokens, 0 for padding

    Parameters
    ----------
    smarts_list:
        Raw reaction SMARTS strings.
    token_to_id:
        Mapping from token string to integer ID.  Must contain the four
        special tokens produced by :func:`build_vocab`:
        ``[PAD]``=0, ``[UNK]``=1, ``[BOS]``=2, ``[EOS]``=3.
    tokenizer:
        Callable that maps a SMARTS string to a list of token strings.
        Defaults to :class:`SmartsTokenizer`.
    max_length:
        All sequences are truncated or padded to this length.
        If ``None``, uses the longest sequence in *smarts_list*.
    add_bos:
        Prepend the ``[BOS]`` token to every sequence.
    add_eos:
        Append the ``[EOS]`` token to every sequence.

    Examples
    --------
    >>> from rxn_smarts_embeddings.datasets.smarts_dataset import SMARTSDataset
    >>> dataset = SMARTSDataset.from_vocab_file(smarts, "data/processed/vocab.json")
    >>> item = dataset[0]
    >>> item["input_ids"].shape, item["attention_mask"].shape
    (torch.Size([128]), torch.Size([128]))
    """

    def __init__(
        self,
        smarts_list: list[str],
        token_to_id: dict[str, int],
        tokenizer: Callable[[str], list[str]] | None = None,
        max_length: int | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> None:
        self.token_to_id = token_to_id
        self.add_bos = add_bos
        self.add_eos = add_eos

        _tokenize = tokenizer if tokenizer is not None else SmartsTokenizer().tokenize

        # Encode all sequences up front
        self._encoded: list[list[int]] = []
        n_errors = 0
        for smarts in smarts_list:
            try:
                ids = self._encode(smarts, _tokenize)
                self._encoded.append(ids)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping %r: %s", smarts[:60], exc)
                self._encoded.append([])
                n_errors += 1

        if n_errors:
            logger.warning(
                "%d / %d sequences failed tokenization.", n_errors, len(smarts_list)
            )

        seq_lengths = [len(ids) for ids in self._encoded]
        self.max_length: int = (
            max_length if max_length is not None else max(seq_lengths, default=1)
        )

        logger.info(
            "SMARTSDataset: %d sequences | max_length=%d | "
            "avg_len=%.1f | vocab_size=%d",
            len(self._encoded),
            self.max_length,
            sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0,
            len(token_to_id),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_vocab_file(
        cls,
        smarts_list: list[str],
        vocab_path: str,
        tokenizer: Callable[[str], list[str]] | None = None,
        max_length: int | None = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> "SMARTSDataset":
        """Construct a dataset by loading *token_to_id* from a ``vocab.json`` file."""
        vocab = json.loads(Path(vocab_path).read_text())
        token_to_id: dict[str, int] = vocab["token_to_id"]
        return cls(smarts_list, token_to_id, tokenizer, max_length, add_bos, add_eos)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, smarts: str, tokenize: Callable[[str], list[str]]) -> list[int]:
        """Tokenize *smarts* and map each token to its integer ID."""
        tokens = tokenize(smarts)
        ids = [self.token_to_id.get(tok, _UNK_ID) for tok in tokens]
        if self.add_bos:
            ids = [_BOS_ID] + ids
        if self.add_eos:
            ids = ids + [_EOS_ID]
        return ids

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ids = self._encoded[idx]

        # Truncate
        ids = ids[: self.max_length]

        # Pad
        pad_len = self.max_length - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [_PAD_ID] * pad_len

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
