"""
Generate sequence length distribution figures for the paper (Methods section 2.1).

Produces three panels saved as a combined PDF + individual SVG/PNG:
  Panel A — SMARTS character-length distribution
  Panel B — Token-length distribution: rule-based SMARTS tokenizer
  Panel C — Token-length distribution: SentencePiece tokenizer

All three share the same x-axis scale for direct visual comparison.
Vertical lines mark key MAX_LENGTH thresholds (256, 512) and the median.

Usage
-----
    uv run python paper/plot_length_distributions.py
    uv run python paper/plot_length_distributions.py \\
        --validated data/processed/validated_smarts.csv \\
        --vocab     data/processed/vocab.json \\
        --sp-model  data/processed/sp_tokenizer.model \\
        --output    results/figures/length_distributions \\
        --dpi       300
"""

import argparse
import json
import logging
import sys
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.smart_rxn_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer
from src.smart_rxn_embeddings.tokenization.sentencepiece_tokenizer import SentencePieceTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Plot style constants (matches project conventions) ─────────────────────
_FS_TITLE = 14
_FS_LABEL = 12
_FS_TICK  = 10
_FS_ANNOT =  9

_COLOR_CHAR   = "#2563eb"   # blue  — character lengths
_COLOR_SMARTS = "#16a34a"   # green — rule-based tokenizer
_COLOR_SP     = "#dc2626"   # red   — SentencePiece tokenizer
_COLOR_MEDIAN = "#1e1e1e"   # near-black
_COLOR_256    = "#d97706"   # amber — MAX_LENGTH=256
_COLOR_512    = "#7c3aed"   # purple — MAX_LENGTH=512


# ── Data loading ──────────────────────────────────────────────────────────

def load_smarts(validated_path: str) -> list[str]:
    df = pd.read_csv(validated_path)
    df = df[df["valid"].astype(str).str.upper() == "TRUE"].dropna(subset=["smarts"])
    logger.info("Loaded %d validated SMARTS", len(df))
    return df["smarts"].tolist()


def compute_lengths(
    smarts_list: list[str],
    smarts_tok: SmartsTokenizer,
    sp_tok: SentencePieceTokenizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    char_lengths = np.array([len(s) for s in smarts_list])

    logger.info("Tokenizing with rule-based SMARTS tokenizer (%d sequences)...", len(smarts_list))
    rule_lengths = np.array([len(smarts_tok.tokenize(s)) for s in smarts_list])

    logger.info("Tokenizing with SentencePiece tokenizer (%d sequences)...", len(smarts_list))
    sp_lengths = np.array([len(sp_tok.tokenize(s)) for s in smarts_list])

    return char_lengths, rule_lengths, sp_lengths


# ── Stats helper ──────────────────────────────────────────────────────────

def _stats_text(arr: np.ndarray) -> str:
    return (
        f"N = {len(arr):,}\n"
        f"Median = {int(np.median(arr))}\n"
        f"Mean = {arr.mean():.0f}\n"
        f"Max = {int(arr.max()):,}"
    )


# ── Individual panels ─────────────────────────────────────────────────────

def _panel(
    arr: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
    x_max: int,
    bins: int = 80,
    vlines_token: bool = False,
) -> plt.Figure:
    """Single histogram panel with median and optional MAX_LENGTH vlines."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Clip display range but keep all data in stats
    plot_arr = arr[arr <= x_max]
    n_clipped = (arr > x_max).sum()

    ax.hist(plot_arr, bins=bins, color=color, alpha=0.80,
            edgecolor="white", linewidth=0.3)

    median = int(np.median(arr))
    ax.axvline(median, color=_COLOR_MEDIAN, linestyle="--", linewidth=1.5,
               label=f"Median = {median}")

    if vlines_token:
        cov_256 = (arr <= 256).mean() * 100
        cov_512 = (arr <= 512).mean() * 100
        ax.axvline(256, color=_COLOR_256, linestyle=":", linewidth=1.8,
                   label=f"MAX_LENGTH=256 ({cov_256:.1f}% covered)")
        ax.axvline(512, color=_COLOR_512, linestyle="-.", linewidth=1.8,
                   label=f"MAX_LENGTH=512 ({cov_512:.1f}% covered)")

    ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
    ax.set_ylabel("Count", fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_TITLE, fontweight="bold", pad=10)
    ax.set_xlim(0, x_max)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=_FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(fontsize=_FS_TICK, loc="upper right", framealpha=0.9)

    # Stats box (lower right, below legend)
    extra = f"\nShown: ≤{x_max:,} ({len(plot_arr)/len(arr)*100:.1f}%)" if n_clipped else ""
    ax.text(
        0.97, 0.62,
        _stats_text(arr) + extra,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=_FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    return fig


# ── Save helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Saved %s", out)


def _save_combined(
    fig_a: plt.Figure,
    fig_b: plt.Figure,
    fig_c: plt.Figure,
    output_path: Path,
    dpi: int,
) -> None:
    """Assemble three panels side-by-side into a single PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    for src_fig, ax in zip([fig_a, fig_b, fig_c], axes):
        src_fig.canvas.draw()
        buf = np.frombuffer(src_fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = src_fig.canvas.get_width_height()
        img = buf.reshape(h, w, 4)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    out = output_path.with_suffix(".pdf")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined figure: %s", out)


# ── Main ──────────────────────────────────────────────────────────────────

def plot_length_distributions(
    smarts_list: list[str],
    smarts_tok: SmartsTokenizer,
    sp_tok: SentencePieceTokenizer,
    output_path: Path,
    dpi: int,
) -> None:
    char_lengths, rule_lengths, sp_lengths = compute_lengths(
        smarts_list, smarts_tok, sp_tok
    )

    # Shared token x-axis: 95th percentile of the wider SP distribution
    token_x_max = int(np.percentile(sp_lengths, 99)) + 20

    fig_a = _panel(
        arr=char_lengths,
        title="(A) Character Length Distribution",
        xlabel="SMARTS length (characters)",
        color=_COLOR_CHAR,
        x_max=int(np.percentile(char_lengths, 99)) + 50,
        bins=80,
        vlines_token=False,
    )
    fig_b = _panel(
        arr=rule_lengths,
        title="(B) Token Length — Rule-Based Tokenizer",
        xlabel="Sequence length (tokens)",
        color=_COLOR_SMARTS,
        x_max=token_x_max,
        bins=80,
        vlines_token=True,
    )
    fig_c = _panel(
        arr=sp_lengths,
        title="(C) Token Length — SentencePiece Tokenizer",
        xlabel="Sequence length (tokens)",
        color=_COLOR_SP,
        x_max=token_x_max,
        bins=80,
        vlines_token=True,
    )

    _save(fig_a, output_path.parent / "length_dist_char",    dpi)
    _save(fig_b, output_path.parent / "length_dist_rule_tok", dpi)
    _save(fig_c, output_path.parent / "length_dist_sp_tok",   dpi)
    _save_combined(fig_a, fig_b, fig_c, output_path, dpi)

    for f in [fig_a, fig_b, fig_c]:
        plt.close(f)

    # ── Save JSON ──────────────────────────────────────────────────────────
    def _arr_stats(arr: np.ndarray) -> dict:
        return {
            'n': int(len(arr)),
            'min': int(arr.min()),
            'median': int(np.median(arr)),
            'mean': round(float(arr.mean()), 1),
            'p75': int(np.percentile(arr, 75)),
            'p90': int(np.percentile(arr, 90)),
            'p95': int(np.percentile(arr, 95)),
            'p99': int(np.percentile(arr, 99)),
            'max': int(arr.max()),
        }

    json_out = output_path.parent / "length_distributions.json"
    results = {
        'character_length': _arr_stats(char_lengths),
        'token_length_rule_based': {
            **_arr_stats(rule_lengths),
            'fertility': round(float(rule_lengths.mean() / char_lengths.mean()), 4),
            'n_leq_256': int((rule_lengths <= 256).sum()),
            'pct_leq_256': round(float((rule_lengths <= 256).mean() * 100), 1),
            'n_leq_512': int((rule_lengths <= 512).sum()),
            'pct_leq_512': round(float((rule_lengths <= 512).mean() * 100), 1),
            'n_gt_512': int((rule_lengths > 512).sum()),
            'pct_gt_512': round(float((rule_lengths > 512).mean() * 100), 1),
        },
        'token_length_sentencepiece': {
            **_arr_stats(sp_lengths),
            'fertility': round(float(sp_lengths.mean() / char_lengths.mean()), 4),
            'n_leq_256': int((sp_lengths <= 256).sum()),
            'pct_leq_256': round(float((sp_lengths <= 256).mean() * 100), 1),
            'n_leq_512': int((sp_lengths <= 512).sum()),
            'pct_leq_512': round(float((sp_lengths <= 512).mean() * 100), 1),
            'n_gt_512': int((sp_lengths > 512).sum()),
            'pct_gt_512': round(float((sp_lengths > 512).mean() * 100), 1),
        },
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", json_out)

    # Print summary table
    print("\n=== Length Distribution Summary ===")
    print(f"{'Metric':<28} {'Characters':>14} {'Rule-Based (tokens)':>20} {'SentencePiece (tokens)':>22}")
    print("-" * 86)
    for label, arr in [("Characters", char_lengths),
                        ("Rule-Based tokens", rule_lengths),
                        ("SentencePiece tokens", sp_lengths)]:
        pass  # printed below

    rows = [
        ("N",        char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("Min",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("Median",   char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("Mean",     char_lengths, rule_lengths, sp_lengths, "{:>14,.1f}", "{:>20,.1f}", "{:>22,.1f}"),
        ("p75",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("p90",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("p95",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("p99",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
        ("Max",      char_lengths, rule_lengths, sp_lengths, "{:>14,.0f}", "{:>20,.0f}", "{:>22,.0f}"),
    ]
    fns = {
        "N": len, "Min": np.min, "Median": np.median, "Mean": np.mean,
        "p75": lambda a: np.percentile(a, 75),
        "p90": lambda a: np.percentile(a, 90),
        "p95": lambda a: np.percentile(a, 95),
        "p99": lambda a: np.percentile(a, 99),
        "Max": np.max,
    }
    fmt_c  = "{:>14,.0f}"
    fmt_r  = "{:>20,.0f}"
    fmt_sp = "{:>22,.0f}"
    fmt_f  = "{:>14,.1f}"
    for name, fn in fns.items():
        c  = fn(char_lengths)
        r  = fn(rule_lengths)
        s  = fn(sp_lengths)
        ff = fmt_f if name == "Mean" else fmt_c
        print(f"{name:<28}" + ff.format(c) + fmt_r.format(r) + fmt_sp.format(s))

    print()
    print(f"{'≤256 tokens (rule-based)':<28}{'—':>14} {(rule_lengths<=256).mean()*100:>19.1f}% {'—':>22}")
    print(f"{'≤512 tokens (rule-based)':<28}{'—':>14} {(rule_lengths<=512).mean()*100:>19.1f}% {'—':>22}")
    print(f"{'≤256 tokens (SentencePiece)':<28}{'—':>14} {'—':>20} {(sp_lengths<=256).mean()*100:>21.1f}%")
    print(f"{'≤512 tokens (SentencePiece)':<28}{'—':>14} {'—':>20} {(sp_lengths<=512).mean()*100:>21.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SMARTS sequence length distributions.")
    p.add_argument("--validated", default="data/processed/validated_smarts.csv")
    p.add_argument("--vocab",     default="data/processed/vocab.json")
    p.add_argument("--sp-model",  default="data/processed/sp_tokenizer.model")
    p.add_argument("--output",    default="results/figures/length_distributions")
    p.add_argument("--dpi",       type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    smarts_list = load_smarts(args.validated)
    smarts_tok  = SmartsTokenizer()
    sp_tok      = SentencePieceTokenizer(args.sp_model)

    plot_length_distributions(
        smarts_list=smarts_list,
        smarts_tok=smarts_tok,
        sp_tok=sp_tok,
        output_path=Path(args.output),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()