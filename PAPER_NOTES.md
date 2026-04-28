# Paper Notes

Design decisions, analysis findings, and rationale for choices made during experiments.
Used as a reference when writing the Methods and Results sections.

---

## Dataset

### SMARTS Length Distribution

- **N = 361,751** validated SMARTS sequences
- Median length: **368 characters** | Mean: **428** | Max: **3,073**
- Distribution is right-skewed with a long tail: only 5% of sequences exceed 954 characters

**Decision: keep all sequences, truncate at tokenisation time.**

- Very long SMARTS (>954 chars) are chemically meaningful multi-step reaction templates — not noise
- They are not removed from the dataset
- `MAX_LENGTH=512` is chosen for both training and embedding extraction:
  - Covers ~98% of the distribution
  - Consistent between pretraining and inference
  - Manageable memory cost on A100 (attention is O(n²) in sequence length)
- The current default `MAX_LENGTH=256` covers only ~75% — insufficient for publication
- ~2% of sequences are truncated at 512 tokens; this should be stated in the paper

**To mention in Methods:** sequences longer than 512 tokens are truncated; 98% of the dataset
is retained in full.

---

## MLM Pre-training — Improved Masking Strategy

### 1. Contiguous Span Masking (SpanBERT-style)

**Change:** Replace independent per-token Bernoulli masking with contiguous span masking.
Span lengths are drawn from a Geometric distribution with mean ≈ 3 tokens, capped at 10.
The 80/10/10 substitution decision (replace with `[MASK]`, a random token, or leave unchanged)
is made once per span so that all tokens in a span receive the same treatment.

**Rationale:** Independent token masking lets the model exploit local cues — e.g., a masked
position between `[` and `]` is almost certainly an atom map or charge symbol, and its
identity can be recovered from bracket-matching alone without any understanding of reaction
chemistry. Masking contiguous spans forces the model to infer entire sub-structures from
longer-range context, requiring genuine understanding of SMARTS syntax and chemical
patterns rather than surface-level pattern matching.  This follows the finding in
SpanBERT (Joshi et al., 2020) that span masking consistently outperforms token masking on
tasks requiring structural understanding.

**Implementation:** `MLMCollator._mask()` — per-batch Python loop; span start sampled
uniformly from un-masked real positions, span length from `Geometric(p = 1/mean_span)`.

### 2. Increased Mask Probability (0.15 → 0.25)

**Change:** Default `mask_prob` raised from 0.15 to 0.25.

**Rationale:** Span masking introduces positive correlation between masked positions, which
reduces the effective number of independent training signals per sequence compared to
independent masking at the same nominal rate.  Raising the mask probability compensates
for this, ensuring that a comparable fraction of the chemical information is withheld at
each step.  A value of 0.25 is consistent with what SpanBERT and subsequent span-masking
works report as optimal in structured-language domains.

### 3. Content-Token Accuracy Metrics

**Change:** Added `content_top1` and `content_top5` validation accuracy metrics alongside
the existing overall `top1` / `top5` metrics.  "Content tokens" are all tokens that are
not special tokens (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`) and not pure structural
punctuation (`(`, `)`, `.`, `>>`, `>`).

**Rationale:** Structural punctuation tokens are trivially predictable from syntactic
context (e.g., the model can learn to close brackets without understanding chemistry).
Including them in accuracy inflates the reported number and obscures genuine chemical
understanding.  The content-token accuracy reflects how well the model predicts
chemically informative tokens — atom symbols, bond types, ring closures, stereo
descriptors, atom maps — which is the quantity we actually care about.  Both metrics are
retained in the logs and saved JSON so that reviewers can see the raw vs. filtered
numbers side by side.

**To mention in Methods:** We report both overall masked-token accuracy and
content-token accuracy (restricted to chemically informative tokens, excluding structural
punctuation) to provide a more faithful measure of chemical understanding.