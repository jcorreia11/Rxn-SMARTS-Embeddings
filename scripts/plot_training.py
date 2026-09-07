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
    val_top1: list[float],
    val_top5: list[float],
    output_path: Path,
    fmt: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    has_acc = bool(val_top1 and val_top5)
    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # --- Loss panel ---
    ax = axes[0]
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=2, color="#2563eb")
    if val_losses:
        val_epochs = list(range(1, len(val_losses) + 1))
        ax.plot(
            val_epochs,
            val_losses,
            label="Val loss",
            linewidth=2,
            linestyle="--",
            color="#dc2626",
        )
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_val = min(val_losses)
        ax.axvline(best_epoch, color="#dc2626", linestyle=":", alpha=0.5, linewidth=1)
        ax.annotate(
            f"Best: {best_val:.4f}\n(epoch {best_epoch})",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + max(1, len(epochs) * 0.05), best_val * 1.15),
            fontsize=8,
            color="#dc2626",
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1),
        )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MLM Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Accuracy panel ---
    if has_acc:
        ax2 = axes[1]
        acc_epochs = list(range(1, len(val_top1) + 1))
        ax2.plot(
            acc_epochs,
            [v * 100 for v in val_top1],
            label="Top-1 accuracy",
            linewidth=2,
            color="#16a34a",
        )
        ax2.plot(
            acc_epochs,
            [v * 100 for v in val_top5],
            label="Top-5 accuracy",
            linewidth=2,
            linestyle="--",
            color="#d97706",
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title(
            "Masked Token Prediction Accuracy", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=11)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.spines[["top", "right"]].set_visible(False)
        # annotate final values
        ax2.annotate(
            f"{val_top1[-1] * 100:.1f}%",
            xy=(acc_epochs[-1], val_top1[-1] * 100),
            xytext=(-30, 8),
            textcoords="offset points",
            fontsize=9,
            color="#16a34a",
        )
        ax2.annotate(
            f"{val_top5[-1] * 100:.1f}%",
            xy=(acc_epochs[-1], val_top5[-1] * 100),
            xytext=(-30, 8),
            textcoords="offset points",
            fontsize=9,
            color="#d97706",
        )

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
    val_top1 = history.get("val_top1_accuracy", [])
    val_top5 = history.get("val_top5_accuracy", [])

    if not train_losses:
        logger.error("No training history found in %s", config_path)
        raise SystemExit(1)

    logger.info(
        "Loaded %d train epochs, %d val epochs (top-1: %s, top-5: %s) from %s",
        len(train_losses),
        len(val_losses),
        f"{val_top1[-1] * 100:.1f}%" if val_top1 else "N/A",
        f"{val_top5[-1] * 100:.1f}%" if val_top5 else "N/A",
        config_path,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = config_path.with_name(
            config_path.stem.replace("smarts_transformer", "training_loss")
        )

    plot_losses(
        train_losses, val_losses, val_top1, val_top5, output_path, args.format, args.dpi
    )


if __name__ == "__main__":
    main()
