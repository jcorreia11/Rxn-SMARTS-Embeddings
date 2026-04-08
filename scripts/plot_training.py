"""Generate training loss plots from a saved model config JSON.

Usage
-----
    python scripts/plot_training.py --config models/smarts_transformer_<RUN_ID>.json
    python scripts/plot_training.py --config models/smarts_transformer_<RUN_ID>.json \\
        --output results/figures/training_loss.png --format pdf
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot MLM training loss curves from a model config JSON."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to smarts_transformer_<RUN_ID>.json",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output file path (default: same directory as config, .png)",
    )
    p.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format (default: png)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for raster formats (default: 300)",
    )
    return p.parse_args()


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_losses, label="Train loss", linewidth=2, color="#2563eb")
    if val_losses:
        ax.plot(
            list(range(1, len(val_losses) + 1)),
            val_losses,
            label="Validation loss",
            linewidth=2,
            linestyle="--",
            color="#dc2626",
        )

        best_val_epoch = val_losses.index(min(val_losses)) + 1
        best_val = min(val_losses)
        ax.axvline(
            best_val_epoch,
            color="#dc2626",
            linestyle=":",
            alpha=0.5,
            linewidth=1,
        )
        ax.annotate(
            f"Best val: {best_val:.4f}\n(epoch {best_val_epoch})",
            xy=(best_val_epoch, best_val),
            xytext=(best_val_epoch + max(1, len(epochs) * 0.05), best_val * 1.15),
            fontsize=8,
            color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MLM Loss", fontsize=12)
    ax.set_title("MLM Pre-training Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = output_path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", out)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    data = json.loads(config_path.read_text())
    history = data.get("history", {})
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])

    if not train_losses:
        logger.error("No training history found in %s", config_path)
        raise SystemExit(1)

    logger.info(
        "Loaded %d train epochs, %d val epochs from %s",
        len(train_losses),
        len(val_losses),
        config_path,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = config_path.with_name(
            config_path.stem.replace("smarts_transformer", "training_loss")
        )

    plot_losses(train_losses, val_losses, output_path, args.format, args.dpi)


if __name__ == "__main__":
    main()