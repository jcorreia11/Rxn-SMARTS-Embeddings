"""Generate a BERT-style architecture diagram for Rxn-SMARTS-Embeddings."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── colours ──────────────────────────────────────────────────────────────────
C_TOKEN = "#AED6F1"  # light blue  – token boxes
C_EMBED = "#A9DFBF"  # light green – embedding block
C_LAYER = "#D2B4DE"  # light purple – transformer layers
C_HEAD = "#FAD7A0"  # light orange – MLM head
C_POOL = "#F9E79F"  # light yellow – pooling block
C_EDGE = "#2C3E50"  # dark        – text / arrows

FIG_W, FIG_H = 10, 12


def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9, radius=0.05):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=C_EDGE,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    dy = 0.05 if sublabel else 0
    ax.text(
        x,
        y + dy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=C_EDGE,
        zorder=4,
    )
    if sublabel:
        ax.text(
            x,
            y - 0.13,
            sublabel,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#555555",
            zorder=4,
        )


def arrow(ax, x1, y1, x2, y2, color=C_EDGE):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        zorder=5,
    )


def token_box(ax, x, y, label, masked=False, special=False):
    color = "#F1948A" if masked else ("#D5D8DC" if special else C_TOKEN)
    w, h = 0.72, 0.38
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0,
        edgecolor=C_EDGE,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        color=C_EDGE,
        family="monospace",
        zorder=4,
    )


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── title ────────────────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2,
    11.6,
    "Rxn-SMARTS-Embeddings — Model Architecture",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=C_EDGE,
)

# ── 1. Input SMARTS string ───────────────────────────────────────────────────
ax.text(
    FIG_W / 2,
    11.15,
    "Input Reaction SMARTS",
    ha="center",
    va="center",
    fontsize=9,
    color="#555555",
)
smarts_str = "[C;H0:1](=[O:2])-[OH]>>[C;H0:1](=[O:2])-[O-]"
ax.text(
    FIG_W / 2,
    10.78,
    smarts_str,
    ha="center",
    va="center",
    fontsize=8,
    color=C_EDGE,
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDFEFE", edgecolor="#BDC3C7", lw=1),
)

arrow(ax, FIG_W / 2, 10.55, FIG_W / 2, 10.15)

# ── 2. Token sequence ────────────────────────────────────────────────────────
tokens = [
    "[BOS]",
    "[C;H0:1]",
    "(",
    "=[O:2]",
    ")",
    "-",
    "[MASK]",
    ">>",
    "[C;H0:1]",
    "=[O:2]",
    "-",
    "[O-]",
]
specials = {0}
masked = {6}
n = len(tokens)
xs = [FIG_W / 2 + (i - (n - 1) / 2) * 0.81 for i in range(n)]
y_tok = 9.82

ax.text(
    FIG_W / 2,
    10.12,
    "Tokenizer  (rule-based, 4 465 types)",
    ha="center",
    va="center",
    fontsize=8.5,
    color="#555555",
    style="italic",
)

for i, (xi, tok) in enumerate(zip(xs, tokens)):
    token_box(ax, xi, y_tok, tok, masked=(i in masked), special=(i in specials))

# ── 3. Embedding block ───────────────────────────────────────────────────────
arrow(ax, FIG_W / 2, y_tok - 0.2, FIG_W / 2, 9.18)
box(
    ax,
    FIG_W / 2,
    8.88,
    5.6,
    0.52,
    C_EMBED,
    "Token Embedding  +  Positional Embedding",
    sublabel="d_model = 256  |  learned positional  |  dropout = 0.1",
    fontsize=9,
)

# ── 4. Transformer encoder layers ────────────────────────────────────────────
arrow(ax, FIG_W / 2, 8.62, FIG_W / 2, 8.22)

layer_y_start = 8.0
layer_h = 0.78
layer_gap = 0.16
n_layers = 6

for i in range(n_layers):
    y = layer_y_start - i * (layer_h + layer_gap)
    box(
        ax,
        FIG_W / 2,
        y,
        5.6,
        layer_h,
        C_LAYER,
        f"Transformer Encoder Layer {i + 1}",
        sublabel="Multi-Head Self-Attention (8 heads)  ·  Add & Norm  ·  FFN (1 024)  ·  Add & Norm",
        fontsize=9,
    )
    if i < n_layers - 1:
        arrow(ax, FIG_W / 2, y - layer_h / 2, FIG_W / 2, y - layer_h / 2 - layer_gap)

# ── 5. Hidden states ─────────────────────────────────────────────────────────
y_after_layers = layer_y_start - (n_layers - 1) * (layer_h + layer_gap) - layer_h / 2
arrow(ax, FIG_W / 2, y_after_layers, FIG_W / 2, y_after_layers - 0.22)

y_hidden = y_after_layers - 0.49
box(
    ax,
    FIG_W / 2,
    y_hidden,
    5.6,
    0.44,
    "#D5D8DC",
    "Hidden States  (seq_len × 256)",
    fontsize=9,
)

# ── 6. Two output branches ────────────────────────────────────────────────────
y_fork = y_hidden - 0.22
x_left = 2.5
x_right = 7.5
y_branch = y_fork - 0.44

# fork arrows
ax.annotate(
    "",
    xy=(x_left, y_branch + 0.18),
    xytext=(FIG_W / 2, y_fork),
    arrowprops=dict(
        arrowstyle="-|>", color="#E74C3C", lw=1.3, connectionstyle="arc3,rad=0.15"
    ),
    zorder=5,
)
ax.annotate(
    "",
    xy=(x_right, y_branch + 0.18),
    xytext=(FIG_W / 2, y_fork),
    arrowprops=dict(
        arrowstyle="-|>", color="#1A5276", lw=1.3, connectionstyle="arc3,rad=-0.15"
    ),
    zorder=5,
)

# branch labels
ax.text(
    x_left,
    y_branch + 0.54,
    "Pre-training",
    ha="center",
    fontsize=8,
    color="#E74C3C",
    fontstyle="italic",
)
ax.text(
    x_right,
    y_branch + 0.54,
    "Inference",
    ha="center",
    fontsize=8,
    color="#1A5276",
    fontstyle="italic",
)

# MLM head
box(
    ax,
    x_left,
    y_branch,
    3.6,
    0.52,
    C_HEAD,
    "MLM Head",
    sublabel="Linear → GELU → LayerNorm → Linear (4 465)",
    fontsize=9,
)

# Pooling block
box(
    ax,
    x_right,
    y_branch,
    3.6,
    0.52,
    C_POOL,
    "Pooling",
    sublabel="[BOS] token (CLS)  or  mean over non-pad tokens",
    fontsize=9,
)

# ── 7. Outputs ────────────────────────────────────────────────────────────────
y_out = y_branch - 0.58
arrow(ax, x_left, y_branch - 0.26, x_left, y_out + 0.15)
arrow(ax, x_right, y_branch - 0.26, x_right, y_out + 0.15)

box(ax, x_left, y_out, 3.0, 0.3, "#FADBD8", "Vocab logits  (4 465)", fontsize=8.5)
box(
    ax, x_right, y_out, 3.0, 0.3, "#FEF9E7", "Reaction embedding  (256-d)", fontsize=8.5
)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_TOKEN, edgecolor=C_EDGE, label="Normal token"),
    mpatches.Patch(facecolor="#F1948A", edgecolor=C_EDGE, label="Masked token"),
    mpatches.Patch(
        facecolor="#D5D8DC", edgecolor=C_EDGE, label="Special token ([BOS])"
    ),
    mpatches.Patch(facecolor=C_LAYER, edgecolor=C_EDGE, label="Transformer layer"),
    mpatches.Patch(facecolor=C_HEAD, edgecolor=C_EDGE, label="MLM head (training)"),
    mpatches.Patch(facecolor=C_POOL, edgecolor=C_EDGE, label="Pooling (inference)"),
]
ax.legend(
    handles=legend_items,
    loc="lower right",
    fontsize=7.5,
    framealpha=0.9,
    edgecolor="#BDC3C7",
    bbox_to_anchor=(0.99, 0.01),
)

plt.tight_layout()
out = "figures/architecture.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved → {out.replace('.pdf', '.png')}")
