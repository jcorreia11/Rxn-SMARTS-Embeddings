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