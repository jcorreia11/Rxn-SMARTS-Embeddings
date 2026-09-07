"""Predict reaction embeddings from SMARTS strings."""

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np

from rxn_smarts_embeddings.models.embed import SmartsEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_VOCAB = "data/processed/vocab.json"
_MODELS_DIR = "models"

# Fallback source when no local checkpoint/vocab is found: the released
# weights for all three model sizes, expected under a `<size>/` prefix
# (e.g. `medium/smarts_transformer_medium.pt`), plus a shared `vocab.json`
# at the repo root (mirrored from the Zenodo data deposit for convenience).
_HF_REPO_ID = "jcorreia11/Rxn-SMARTS-Embeddings"
_HF_SIZES = ("small", "medium", "large")
_DEFAULT_HF_SIZE = "medium"


def _find_latest_checkpoint(models_dir: str = _MODELS_DIR) -> tuple[str, str]:
    """Return ``(weights_path, config_path)`` for the most recent checkpoint.

    Checkpoints are named ``smarts_transformer_<YYYYMMDD_HHMMSS>.pt``; the
    lexicographic sort on the timestamp suffix gives the newest file last.
    """
    pts = sorted(Path(models_dir).glob("smarts_transformer_*.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No checkpoints found in {models_dir!r}. "
            "Pass --weights and --config explicitly, or run from the project root."
        )
    latest = pts[-1]
    config = latest.with_suffix(".json")
    if not config.exists():
        raise FileNotFoundError(
            f"Config file not found for checkpoint {latest}. Expected {config}."
        )
    return str(latest), str(config)


def _download_checkpoint_from_hub(size: str) -> tuple[str, str]:
    """Download and cache the ``(weights, config)`` pair for *size* from the Hub."""
    if size not in _HF_SIZES:
        raise ValueError(f"size must be one of {_HF_SIZES}, got {size!r}")
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(_HF_REPO_ID, f"{size}/smarts_transformer_{size}.pt")
    config = hf_hub_download(_HF_REPO_ID, f"{size}/smarts_transformer_{size}.json")
    return weights, config


def _download_vocab_from_hub() -> str:
    """Download and cache ``vocab.json`` from the Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_HF_REPO_ID, "vocab.json")


def _resolve_vocab(vocab: str | None) -> str:
    if vocab is not None:
        return vocab
    if Path(_DEFAULT_VOCAB).exists():
        return _DEFAULT_VOCAB
    logger.info(
        "No local vocab.json found — downloading from the Hugging Face Hub (%s)...",
        _HF_REPO_ID,
    )
    return _download_vocab_from_hub()


def load_embedder(
    weights: str | None = None,
    config: str | None = None,
    vocab: str | None = None,
    pooling: str = "mean",
    device: str | None = None,
    hf_size: str = _DEFAULT_HF_SIZE,
) -> SmartsEmbedder:
    """Load a :class:`~rxn_smarts_embeddings.models.embed.SmartsEmbedder`.

    When *weights* or *config* are ``None``, the most recent checkpoint in
    ``models/`` is discovered automatically; if none is found, the *hf_size*
    checkpoint is downloaded (and cached) from the Hugging Face Hub. *vocab*
    falls back the same way, to ``data/processed/vocab.json`` and then the Hub.

    Parameters
    ----------
    weights:
        Path to ``smarts_transformer_*.pt``.
    config:
        Path to the companion ``smarts_transformer_*.json``.
    vocab:
        Path to ``vocab.json``. Defaults to ``data/processed/vocab.json``,
        then the Hugging Face Hub.
    pooling:
        ``"mean"`` (default, recommended) or ``"cls"``.
    device:
        ``"cuda"`` or ``"cpu"``. Auto-detected when ``None``.
    hf_size:
        ``"small"``, ``"medium"`` (default), or ``"large"`` — which released
        model to download when no local checkpoint is found. Ignored when
        *weights*/*config* are given explicitly or auto-discovered locally.
    """
    if weights is None or config is None:
        try:
            weights, config = _find_latest_checkpoint()
            logger.info("Auto-discovered checkpoint: %s", weights)
        except FileNotFoundError:
            logger.info(
                "No local checkpoint in %r — downloading the %r model from the "
                "Hugging Face Hub (%s)...",
                _MODELS_DIR,
                hf_size,
                _HF_REPO_ID,
            )
            weights, config = _download_checkpoint_from_hub(hf_size)
    vocab = _resolve_vocab(vocab)
    return SmartsEmbedder.from_checkpoint(weights, config, vocab, pooling, device)


def predict(
    smarts: "str | list[str]",
    *,
    weights: str | None = None,
    config: str | None = None,
    vocab: str | None = None,
    pooling: str = "mean",
    device: str | None = None,
    batch_size: int = 64,
    max_length: int | None = None,
    hf_size: str = _DEFAULT_HF_SIZE,
) -> np.ndarray:
    """Embed one or more reaction SMARTS strings.

    Parameters
    ----------
    smarts:
        A single SMARTS string or a list of SMARTS strings.
    weights, config, vocab:
        Paths to model files. Latest checkpoint is auto-discovered if omitted,
        falling back to a download from the Hugging Face Hub.
    pooling:
        ``"mean"`` (default, recommended — average non-padding tokens) or
        ``"cls"`` (BOS token).
    device:
        ``"cuda"`` or ``"cpu"``. Auto-detected when ``None``.
    batch_size:
        Number of sequences per forward pass.
    max_length:
        Pad/truncate to this length. Defaults to the model's ``max_seq_len``.
    hf_size:
        ``"small"``, ``"medium"`` (default), or ``"large"`` — which released
        model to download when no local checkpoint is found.

    Returns
    -------
    numpy.ndarray
        Shape ``(d_model,)`` for a single SMARTS string, ``(N, d_model)``
        for a list.

    Examples
    --------
    >>> from rxn_smarts_embeddings.predict import predict
    >>> emb = predict("[C:1]-[O:2]>>[C:1]=[O:2]")   # shape (256,)
    >>> embs = predict(["[C:1]-[O:2]>>[C:1]=[O:2]", "c1ccccc1>>c1cccnc1"])  # (2, 256)
    """
    single = isinstance(smarts, str)
    smarts_list = [smarts] if single else list(smarts)
    embedder = load_embedder(weights, config, vocab, pooling, device, hf_size)
    result = embedder.embed(smarts_list, batch_size=batch_size, max_length=max_length)
    return result[0] if single else result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _infer_format(output: str | None, fmt: str | None) -> str:
    if fmt:
        return fmt
    if output:
        suffix = Path(output).suffix.lstrip(".")
        if suffix in ("npy", "json", "csv"):
            return suffix
    return "json"


def _write_output(
    smarts_list: list[str],
    embeddings: np.ndarray,
    output: str | None,
    fmt: str,
) -> None:
    if fmt == "npy":
        path = Path(output)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info("Saved %s embeddings to %s", embeddings.shape, path)

    elif fmt == "json":
        payload = json.dumps(
            [
                {"smarts": s, "embedding": emb.tolist()}
                for s, emb in zip(smarts_list, embeddings)
            ],
            indent=2,
        )
        if output:
            Path(output).write_text(payload)
            logger.info("Saved JSON to %s", output)
        else:
            print(payload)

    elif fmt == "csv":
        header = ["smarts"] + [f"dim_{i}" for i in range(embeddings.shape[1])]
        rows = [[s] + emb.tolist() for s, emb in zip(smarts_list, embeddings)]
        if output:
            with open(output, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            logger.info("Saved CSV to %s", output)
        else:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(header)
            w.writerows(rows)
            print(buf.getvalue(), end="")

    else:
        raise ValueError(f"Unknown format {fmt!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smarts-embed",
        description="Predict reaction embeddings from SMARTS strings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single SMARTS — prints JSON to stdout
  smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]"

  # Multiple SMARTS as positional args
  smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" "c1ccccc1>>c1cccnc1"

  # Batch from file, save as numpy array
  smarts-embed --file smarts.txt --output embeddings.npy

  # Pipe from stdin, output CSV
  echo "[C:1]-[O:2]>>[C:1]=[O:2]" | smarts-embed --format csv

  # Explicit model paths
  smarts-embed "[C:1]-[O:2]>>[C:1]=[O:2]" \\
      --weights models/smarts_transformer_20260715_111009.pt \\
      --config  models/smarts_transformer_20260715_111009.json
""",
    )

    # -- Input ---------------------------------------------------------------
    p.add_argument(
        "smarts",
        nargs="*",
        metavar="SMARTS",
        help=(
            "One or more reaction SMARTS strings (positional). "
            "Mutually exclusive with --file. "
            "If neither is given, SMARTS are read from stdin (one per line)."
        ),
    )
    p.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        help="Text file with one SMARTS per line (mutually exclusive with positional args).",
    )

    # -- Model ---------------------------------------------------------------
    model_grp = p.add_argument_group("model")
    model_grp.add_argument(
        "--weights",
        metavar="PATH",
        default=None,
        help="Model weights (.pt). Auto-discovers the latest local checkpoint, "
        "falling back to a Hugging Face Hub download, when omitted.",
    )
    model_grp.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Model config (.json). Auto-discovers alongside --weights when omitted.",
    )
    model_grp.add_argument(
        "--vocab",
        metavar="PATH",
        default=None,
        help=f"Vocabulary file (default: {_DEFAULT_VOCAB}, "
        "falling back to a Hugging Face Hub download).",
    )
    model_grp.add_argument(
        "--size",
        choices=list(_HF_SIZES),
        default=_DEFAULT_HF_SIZE,
        help="Model size to download from the Hugging Face Hub when no local "
        f"checkpoint is found (default: {_DEFAULT_HF_SIZE}). Ignored when "
        "--weights/--config are given or a local checkpoint is auto-discovered.",
    )
    model_grp.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="mean",
        help="Pooling strategy: 'mean' (average non-padding tokens) or 'cls' "
        "(BOS token) (default: mean).",
    )
    model_grp.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="'cuda' or 'cpu'. Auto-detected when omitted.",
    )
    model_grp.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Sequences per forward pass (default: 64).",
    )
    model_grp.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="N",
        help="Pad/truncate length. Defaults to model's max_seq_len.",
    )

    # -- Output --------------------------------------------------------------
    out_grp = p.add_argument_group("output")
    out_grp.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=None,
        help=(
            "Write results to file. Format is inferred from the extension "
            "(.npy, .json, .csv). When omitted, results go to stdout as JSON."
        ),
    )
    out_grp.add_argument(
        "--format",
        dest="fmt",
        choices=["npy", "json", "csv"],
        default=None,
        help="Override output format (default: inferred from --output, else json).",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``smarts-embed``."""
    p = _build_parser()
    args = p.parse_args(argv)

    if args.smarts and args.file:
        p.error("Provide either positional SMARTS or --file, not both.")

    fmt = _infer_format(args.output, args.fmt)
    if fmt == "npy" and args.output is None:
        p.error("--output is required when the format is 'npy'.")

    # Resolve SMARTS input
    if args.file:
        text = Path(args.file).read_text()
        smarts_list = [s.strip() for s in text.splitlines() if s.strip()]
    elif args.smarts:
        smarts_list = list(args.smarts)
    else:
        smarts_list = [s.strip() for s in sys.stdin if s.strip()]

    if not smarts_list:
        p.error("No SMARTS input provided.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    embedder = load_embedder(
        weights=args.weights,
        config=args.config,
        vocab=args.vocab,
        pooling=args.pooling,
        device=args.device,
        hf_size=args.size,
    )
    embeddings = embedder.embed(
        smarts_list, batch_size=args.batch_size, max_length=args.max_length
    )
    _write_output(smarts_list, embeddings, args.output, fmt)


if __name__ == "__main__":
    main()
