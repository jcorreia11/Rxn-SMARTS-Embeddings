"""
Generate tokenization example figure for Methods section 2.2.1.

Shows a real RetroRules SMARTS tokenized by the rule-based tokenizer,
with tokens colour-coded by grammatical type. The reaction arrow splits
the tokens into a reactant row and a product row.

Output: paper/figures/tokenizer_example.{pdf,png}
Run from project root: python paper/plot_tokenizer_example.py
"""

import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from src.rxn_smarts_embeddings.tokenization.smarts_tokenizer import SmartsTokenizer

# ── Example reaction (aromatic decarboxylation from RetroRules v3.0) ───────
SMARTS = (
    "[c;H1:1]:[c;H0:2](:[c;H1:3])-[C;H0:4](=[O;H0:5])-[O;H1:6]"
    ">>"
    "[O;H0:5]=[C;H0:4]=[O;H0:6].[c;H1:1]:[c;H1:2]:[c;H1:3]"
)
REACTION_LABEL = "Aromatic decarboxylation (RetroRules v3.0)"

OUTPUT_DIR = Path("paper/figures")

# ── Token type colour palette (colorblind-safe, dark-bg text contrast) ─────
PALETTE = {
    "bracket_atom":   ("#2166AC", "white"),   # blue
    "reaction_arrow": ("#B2182B", "white"),   # red
    "aromatic_bond":  ("#1B7837", "white"),   # dark green
    "single_bond":    ("#E08214", "white"),   # amber
    "double_bond":    ("#762A83", "white"),   # purple
    "triple_bond":    ("#3288BD", "white"),   # sky blue
    "branch":         ("#777777", "white"),   # grey
    "disconnection":  ("#D6604D", "white"),   # salmon
    "ring_closure":   ("#8C6D3F", "white"),   # brown
    "atom":           ("#4DAC26", "black"),   # lime
    "other_bond":     ("#80B1D3", "black"),   # light blue
    "logical_op":     ("#D1B000", "black"),   # gold
    "agent_sep":      ("#66BD63", "black"),   # mint
}

LABELS = {
    "bracket_atom":   "Bracket atom expression",
    "reaction_arrow": "Reaction arrow",
    "aromatic_bond":  "Aromatic bond  (:)",
    "single_bond":    "Single bond  (−)",
    "double_bond":    "Double bond  (=)",
    "triple_bond":    "Triple bond  (#)",
    "branch":         "Branch / grouping",
    "disconnection":  "Disconnection  (.)",
    "ring_closure":   "Ring closure",
    "atom":           "Non-bracket atom",
    "other_bond":     "Other bond / stereo",
    "logical_op":     "Logical operator",
    "agent_sep":      "Agent separator  (>)",
}


def categorize(tok: str) -> str:
    if tok.startswith("["):                           return "bracket_atom"
    if tok == ">>":                                   return "reaction_arrow"
    if tok == ">":                                    return "agent_sep"
    if tok == ":":                                    return "aromatic_bond"
    if tok == "-":                                    return "single_bond"
    if tok == "=":                                    return "double_bond"
    if tok == "#":                                    return "triple_bond"
    if tok in ("~", "@", "/", "\\", "/?", "\\?"):    return "other_bond"
    if tok in ("(", ")"):                             return "branch"
    if tok == ".":                                    return "disconnection"
    if tok.isdigit() or tok.startswith("%"):          return "ring_closure"
    if tok in ("!", "&", ",", ";"):                   return "logical_op"
    return "atom"


# ── Layout constants (all in inches; 1 data unit = 1 inch) ─────────────────
FONT_SIZE   = 8.5        # pt — monospace token labels
CHAR_W      = FONT_SIZE * 0.60 / 72   # inch per character (monospace)
BOX_PAD_X   = 0.14       # horizontal padding inside each box (each side)
BOX_H       = 0.42       # box height
ROW_GAP     = 0.38       # vertical gap between token rows
GAP_X       = 0.09       # horizontal gap between boxes
MARGIN_X    = 0.35       # left/right figure margin
MARGIN_TOP  = 0.70       # above SMARTS label
SMARTS_H    = 0.42       # SMARTS string box height
ARROW_H     = 0.28       # tokenizer arrow section height
LEG_GAP     = 0.30       # gap between last token row and legend
LEG_ITEM_H  = 0.22       # legend item height
LEG_COLS    = 4          # legend columns


def box_w(tok: str) -> float:
    return len(tok) * CHAR_W + 2 * BOX_PAD_X


# ── Tokenize and categorise ────────────────────────────────────────────────
tokenizer = SmartsTokenizer()
tokens = tokenizer.tokenize(SMARTS)
types  = [categorize(t) for t in tokens]

# Split into reactant row / product row at >>
arr_idx   = tokens.index(">>")
reactants = list(zip(tokens[:arr_idx], types[:arr_idx]))
products  = list(zip(tokens[arr_idx:], types[arr_idx:]))
rows      = [reactants, products]
row_labels = ["Reactant", "Product"]

# ── Compute row widths and total figure width ──────────────────────────────
def row_width(row):
    return sum(box_w(t) for t, _ in row) + GAP_X * (len(row) - 1)

max_row_w = max(row_width(r) for r in rows)
fig_w = max_row_w + 2 * MARGIN_X

# ── Compute figure height ──────────────────────────────────────────────────
n_rows    = len(rows)
used_types = sorted(set(types), key=lambda t: list(PALETTE.keys()).index(t))
n_leg_rows = -(-len(used_types) // LEG_COLS)   # ceiling division

fig_h = (
    MARGIN_TOP
    + SMARTS_H
    + ARROW_H
    + n_rows * BOX_H
    + (n_rows - 1) * ROW_GAP
    + LEG_GAP
    + n_leg_rows * LEG_ITEM_H
    + 0.30   # bottom margin
)

# ── Draw ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

y = fig_h  # current drawing cursor (top-down)

# -- SMARTS string -----------------------------------------------------------
y -= MARGIN_TOP
smarts_cx = fig_w / 2
smarts_cy = y - SMARTS_H / 2
ax.text(
    smarts_cx, smarts_cy, SMARTS,
    ha="center", va="center",
    fontsize=7.5, fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="#F3F3F3",
        edgecolor="#AAAAAA",
        lw=0.8,
    ),
    clip_on=False,
    zorder=3,
)
# small caption above
ax.text(
    smarts_cx, y - 0.10,
    f"Input SMARTS  —  {REACTION_LABEL}",
    ha="center", va="bottom",
    fontsize=7, color="#555555",
    clip_on=False,
)
y -= SMARTS_H

# -- Downward arrow with label -----------------------------------------------
arrow_top = y
arrow_bot = y - ARROW_H
ax.annotate(
    "", xy=(smarts_cx, arrow_bot + 0.04),
    xytext=(smarts_cx, arrow_top),
    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2),
)
ax.text(
    smarts_cx + 0.12, (arrow_top + arrow_bot) / 2,
    f"SmartsTokenizer  →  {len(tokens)} tokens",
    ha="left", va="center",
    fontsize=7, fontstyle="italic", color="#444444",
)
y -= ARROW_H

# -- Token rows --------------------------------------------------------------
for row_idx, row in enumerate(rows):
    if row_idx > 0:
        y -= ROW_GAP

    rw = row_width(row)
    x  = (fig_w - rw) / 2   # centre the row
    cy = y - BOX_H / 2       # vertical centre of this row

    # Row label (left margin)
    ax.text(
        MARGIN_X - 0.12, cy,
        row_labels[row_idx],
        ha="right", va="center",
        fontsize=7, color="#555555", fontstyle="italic",
    )

    for tok_str, tok_type in row:
        w = box_w(tok_str)
        color, text_color = PALETTE[tok_type]

        patch = FancyBboxPatch(
            (x, cy - BOX_H / 2), w, BOX_H,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white",
            lw=0.6, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, cy, tok_str,
            ha="center", va="center",
            fontsize=FONT_SIZE, fontfamily="monospace",
            color=text_color,
            zorder=3,
        )
        x += w + GAP_X

    y -= BOX_H

# -- Legend ------------------------------------------------------------------
leg_y = y - LEG_GAP
handles = [
    mpatches.Patch(
        facecolor=PALETTE[t][0],
        edgecolor="#AAAAAA",
        lw=0.4,
        label=LABELS[t],
    )
    for t in used_types
]
legend = ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, leg_y / fig_h),
    ncol=LEG_COLS,
    fontsize=7,
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    handlelength=1.4,
    handleheight=0.95,
    columnspacing=1.0,
    handletextpad=0.5,
)
legend.get_frame().set_linewidth(0.6)


# ── Save ────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    out = OUTPUT_DIR / f"tokenizer_example.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved  {out}")

plt.close(fig)
n_tok = len(tokens)
print(f"\nTokens ({n_tok}):")
for i, (t, ty) in enumerate(zip(tokens, types)):
    print(f"  {i+1:2d}. {t!r:14s}  [{ty}]")