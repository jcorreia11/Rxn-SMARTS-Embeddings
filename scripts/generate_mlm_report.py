"""Generate a hardware/software/training report for a research article.

Reads the model config JSON produced by train_mlm.py and collects
environment information to produce a self-contained Markdown report.

Usage
-----
    python scripts/generate_mlm_report.py \\
        --config models/smarts_transformer_<RUN_ID>.json \\
        --output results/mlm_training_report.md

    # On the HPC (after the training job), pass the env JSON collected
    # by the sbatch to avoid needing GPU access at report time:
    python scripts/generate_mlm_report.py \\
        --config models/smarts_transformer_<RUN_ID>.json \\
        --env    models/smarts_transformer_<RUN_ID>_env.json \\
        --output results/mlm_training_report.md
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def collect_env() -> dict:
    """Collect hardware and software info — delegates to collect_env.py."""
    # Import from the sibling script to avoid duplication.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "collect_env",
        Path(__file__).parent / "collect_env.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.collect_env()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _opt(v, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "N/A"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _hardware_section(env: dict) -> str:
    lines = [
        "## Hardware",
        "",
        "| Component | Details |",
        "|-----------|---------|",
        f"| GPU | {_opt(env.get('gpu_name'))} |",
        f"| GPU Memory | {_opt(env.get('gpu_memory_gb'), ' GB')} |",
        f"| GPU Count | {_opt(env.get('gpu_count'))} |",
        f"| CUDA Version | {_opt(env.get('cuda_version'))} |",
        f"| Driver Version | {_opt(env.get('driver_version'))} |",
        f"| CPU | {_opt(env.get('cpu_model'))} |",
        f"| CPU Cores | {_opt(env.get('cpu_cores'))} |",
        f"| RAM | {_opt(env.get('ram_gb'), ' GB')} |",
        f"| OS | {_opt(env.get('platform'))} |",
    ]
    return "\n".join(lines)


def _software_section(env: dict) -> str:
    lines = [
        "## Software",
        "",
        "| Package | Version |",
        "|---------|---------|",
        f"| Python | {_opt(env.get('python_version'))} |",
        f"| PyTorch | {_opt(env.get('torch_version'))} |",
        f"| NumPy | {_opt(env.get('pkg_numpy'))} |",
        f"| pandas | {_opt(env.get('pkg_pandas'))} |",
        f"| scikit-learn | {_opt(env.get('pkg_scikit_learn'))} |",
        f"| SentencePiece | {_opt(env.get('pkg_sentencepiece'))} |",
        f"| RDKit | {_opt(env.get('pkg_rdkit'))} |",
    ]
    return "\n".join(lines)


def _model_section(cfg: dict) -> str:
    mc = cfg.get("model_config", {})
    tc = cfg.get("training_config", {})

    n_params = _count_params(cfg)

    lines = [
        "## Model Architecture",
        "",
        "Transformer encoder pre-trained with a masked language modelling (MLM) "
        "objective (BERT-style, 80/10/10 masking strategy).",
        "",
        "| Hyper-parameter | Value |",
        "|-----------------|-------|",
        f"| Hidden dimension (`d_model`) | {mc.get('d_model', 'N/A')} |",
        f"| Attention heads (`nhead`) | {mc.get('nhead', 'N/A')} |",
        f"| Encoder layers | {mc.get('num_encoder_layers', 'N/A')} |",
        f"| Feed-forward dimension | {mc.get('dim_feedforward', 'N/A')} |",
        f"| Dropout | {mc.get('dropout', 'N/A')} |",
        f"| Max sequence length | {mc.get('max_seq_len', 'N/A')} |",
        f"| Vocabulary size | {mc.get('vocab_size', 'N/A'):,} |"
        if isinstance(mc.get('vocab_size'), int) else
        f"| Vocabulary size | {mc.get('vocab_size', 'N/A')} |",
        f"| Total parameters | {n_params} |",
        f"| Masking probability | {tc.get('mask_prob', 0.15):.0%} |",
    ]
    return "\n".join(lines)


def _training_section(cfg: dict) -> str:
    tc = cfg.get("training_config", {})
    history = cfg.get("history", {})

    warmup = tc.get("warmup_steps", 0)
    scheduler_desc = (
        f"Linear warmup for {warmup} steps, then cosine decay to 0"
        if warmup > 0
        else "Cosine decay (no warmup)"
    )
    grad_clip = tc.get("max_grad_norm", 0)
    clip_desc = f"Max norm {grad_clip}" if grad_clip > 0 else "Disabled"

    val_split = tc.get("val_split", 0.0)
    split_desc = f"{val_split:.0%} held out" if val_split > 0 else "None (full dataset used for training)"

    lines = [
        "## Training Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Optimizer | AdamW |",
        f"| Learning rate | {tc.get('learning_rate', 'N/A')} |",
        f"| LR schedule | {scheduler_desc} |",
        f"| Batch size | {tc.get('batch_size', 'N/A')} |",
        f"| Epochs | {tc.get('num_epochs', 'N/A')} |",
        f"| Gradient clipping | {clip_desc} |",
        f"| Validation split | {split_desc} |",
        f"| DataLoader workers | {tc.get('num_workers', 0)} |",
        f"| Training time | {_fmt_time(history.get('training_time_s'))} |",
    ]
    return "\n".join(lines)


def _results_section(cfg: dict, figure_path: str | None) -> str:
    history = cfg.get("history", {})
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])

    lines = ["## Results", ""]

    val_top1 = history.get("val_top1_accuracy", [])
    val_top5 = history.get("val_top5_accuracy", [])

    if train_losses:
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Initial train loss | {train_losses[0]:.4f} |",
            f"| Final train loss | {train_losses[-1]:.4f} |",
            f"| Best train loss | {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses)) + 1}) |",
        ]
        if val_losses:
            lines += [
                f"| Final val loss | {val_losses[-1]:.4f} |",
                f"| Best val loss | {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1}) |",
            ]
        if val_top1:
            lines += [
                f"| Final top-1 accuracy | {val_top1[-1]*100:.2f}% |",
                f"| Best top-1 accuracy | {max(val_top1)*100:.2f}% (epoch {val_top1.index(max(val_top1)) + 1}) |",
            ]
        if val_top5:
            lines += [
                f"| Final top-5 accuracy | {val_top5[-1]*100:.2f}% |",
                f"| Best top-5 accuracy | {max(val_top5)*100:.2f}% (epoch {val_top5.index(max(val_top5)) + 1}) |",
            ]
        lines.append("")

    if figure_path:
        lines += [
            f"![Training loss curve]({figure_path})",
            "",
            f"*Figure: MLM training loss over epochs.*",
        ]

    return "\n".join(lines)


def _count_params(cfg: dict) -> str:
    """Estimate parameter count from config without loading the model."""
    mc = cfg.get("model_config", {})
    try:
        import torch
        from smart_rxn_embeddings.models.smarts_transformer import (
            SmartsMLMModel,
            TransformerConfig,
        )
        model_cfg = TransformerConfig.from_dict(mc)
        model = SmartsMLMModel(model_cfg)
        n = sum(p.numel() for p in model.parameters())
        return f"{n:,}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_report(cfg: dict, env: dict, figure_path: str | None) -> str:
    today = date.today().isoformat()
    run_id = Path(cfg.get("training_config", {}).get("output_path", "")).stem or "unknown"

    sections = [
        "# MLM Pre-training Report",
        "",
        f"**Date:** {today}  ",
        f"**Run ID:** `{run_id}`",
        "",
        "---",
        "",
        _hardware_section(env),
        "",
        "---",
        "",
        _software_section(env),
        "",
        "---",
        "",
        _model_section(cfg),
        "",
        "---",
        "",
        _training_section(cfg),
        "",
        "---",
        "",
        _results_section(cfg, figure_path),
        "",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate hardware/software/training report for a research article."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to smarts_transformer_<RUN_ID>.json",
    )
    p.add_argument(
        "--env",
        default=None,
        help="Pre-collected environment JSON (optional; collected live if omitted)",
    )
    p.add_argument(
        "--figure",
        default=None,
        help="Relative path to the training loss figure for embedding in the report",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output Markdown path (default: <config_dir>/mlm_training_report.md)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    cfg = json.loads(config_path.read_text())

    if args.env:
        env = json.loads(Path(args.env).read_text())
        logger.info("Loaded environment from %s", args.env)
    else:
        logger.info("Collecting environment info from current machine...")
        env = collect_env()

    output_path = Path(args.output) if args.output else config_path.parent / "mlm_training_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(cfg, env, args.figure)
    output_path.write_text(report)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()