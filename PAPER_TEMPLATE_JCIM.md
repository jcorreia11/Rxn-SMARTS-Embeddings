# JCIM Paper Template — SmartRxnEmbeddings

> **How to use this template**
> Text in `[square brackets]` = instructions to the author (delete before submitting).
> Text in `**bold**` = already-known values you can paste in directly.
> `[PENDING]` = results not yet available (requires MAX_LENGTH=512 retraining).
> JCIM word limits: Abstract ≤ 250 words; full article typically 6,000–10,000 words.

---

## Title

**Self-Supervised Transformer Embeddings for Biochemical Reaction SMARTS via Masked Language Modeling**

Alternative framings to consider:
- "Learning Chemical Reaction Representations from SMARTS with Span-Masked Language Modeling"
- "Reaction Rule Embeddings from Unlabeled SMARTS: A BERT-Style Pretraining Approach"
Pick a title that highlights the novelty (self-supervised + SMARTS domain) and the application.]

---

## Table of Contents (TOC) Graphic

[Required by JCIM. 1 image, ~3.25 × 1.75 inches.
Suggested content: a schematic showing SMARTS string → Transformer → embedding space colored by EC class (use your UMAP figure). Should be self-explanatory without reading the paper.]

---

## Abstract

[~200–250 words. Structure: (1) motivation/problem, (2) what you did, (3) how you did it, (4) key results, (5) what this enables. Do NOT use references or acronyms that need defining. Write last.]

Reaction templates expressed in SMARTS notation encode rich chemical knowledge about
bond-breaking and bond-forming events, yet remain largely unexploited as a source of
learned molecular representations. [SENTENCE 1 — Expand on why this matters: retrosynthesis
planning, enzyme function prediction, reaction retrieval all rely on meaningful reaction
similarity measures.]

Here we present **SmartRxnEmbeddings**, a self-supervised framework that pretrain a
BERT-style transformer encoder directly on reaction SMARTS using a masked language
modeling (MLM) objective with contiguous span masking. The model is trained on
**361,751** validated reaction templates from the RetroRules v3.0 database without any
labeled supervision.

[SENTENCES 3–5 — Briefly describe: rule-based SMARTS tokenizer, transformer architecture
(6 layers, d_model=256, ~5M parameters), SpanBERT-style span masking (mean span=3,
mask_prob=0.25), mean pooling to obtain fixed-size reaction embeddings.]

We evaluate the resulting embeddings on three downstream tasks. For enzyme class (EC)
prediction across **[PENDING — update after MAX_LENGTH=512 run]** EC classes, pretrained
embeddings with a shallow MLP achieve **[PENDING]%** accuracy at depth=1 and
**[PENDING]%** at depth=2, outperforming TF-IDF baselines by **[PENDING]** percentage
points and randomly initialized embeddings by **[PENDING]** points. Pairwise cosine
similarity in embedding space correlates with structural Tanimoto similarity at
**r = 0.591** (Spearman ρ = 0.575) across **4.5 million** reaction pairs, demonstrating
that the model encodes chemical information beyond syntactic pattern matching. UMAP
visualization and nearest-neighbor retrieval further confirm that the embedding space
organizes reactions by chemical function.

[LAST SENTENCE — Broader implication: these embeddings provide a label-free foundation
for reaction retrieval, clustering, and transfer learning in retrosynthesis and enzyme
engineering.]

---

## 1. Introduction

[~600–900 words, 4–5 paragraphs. Goal: establish the problem, survey prior work, identify
the gap, and state your contributions clearly.]

### Paragraph 1 — Context and Motivation

[Why do reaction representations matter?
- Retrosynthesis planning (e.g., ASKCOS, AIZYNTHFINDER) relies on reaction template matching
- Enzyme engineering and metabolic pathway design need meaningful reaction similarity
- RetroRules encodes generalized reaction logic as SMARTS — millions of templates available
Key message: there is a large unlabeled corpus of reaction SMARTS, and no good way to
embed it into a continuous space that captures chemical semantics.]

### Paragraph 2 — Prior Work on Molecular Representations

[Survey what exists:
- SMILES-based molecule encoders: ChemBERTa, MolBERT, GROVER (graph-based)
- Reaction-level representations: RXNMapper (attention-based atom mapping),
  Schwaller et al. (BERT on reaction SMILES), rxnfp
- Template/SMARTS-specific: what has been done? Probably very little — this is your gap
- TF-IDF and fingerprint baselines for reaction templates
Key message: existing methods operate on SMILES reactions (product + reactants), not on
generalized SMARTS templates. The SMARTS domain has distinct tokenization challenges and
encodes *reaction rules*, not specific reactions.]

### Paragraph 3 — The Gap

[Articulate clearly what is missing:
- No transformer model trained specifically on SMARTS syntax
- SMARTS tokens are structurally richer and more diverse than SMILES tokens (atom maps,
  recursion, reaction arrows, stereo descriptors in template form)
- Independent token masking in MLM is insufficient for SMARTS: bracket structure lets
  the model recover masked tokens from local context without understanding chemistry
  (motivate span masking here)
- No systematic benchmark of SMARTS embedding quality against labeled tasks]

### Paragraph 4 — Our Contributions

[List 3–4 concrete contributions, numbered or bulleted:
1. A rule-based SMARTS tokenizer that correctly handles Daylight SMARTS grammar
   (atom expressions, bond types, ring closures, reaction arrows, atom maps)
2. A pretrained transformer encoder for reaction SMARTS with SpanBERT-style span masking,
   trained on 361,751 templates from RetroRules v3.0
3. A systematic evaluation suite: EC classification (depth 1–3), embedding–Tanimoto
   similarity correlation, UMAP visualization, nearest-neighbor retrieval
4. Empirical evidence that MLM pretraining encodes enzyme-class information well beyond
   TF-IDF baselines and random initialization, establishing a strong foundation for
   transfer learning on reaction templates]

### Paragraph 5 — Paper Outline

[Optional for JCIM but common. One sentence per section:
"The remainder of the paper is organized as follows: Section 2 describes..."]

---

## 2. Methods

[JCIM calls this section "Methods" or "Computational Details". Be precise enough to
reproduce. Target ~1,500–2,000 words.]

### 2.1 Dataset

Reaction templates were obtained from RetroRules v3.0 [CITE RetroRules paper], a
database of enzymatic reaction rules derived from the MetaNetX [CITE] and Rhea [CITE]
metabolic databases. RetroRules encodes generalized enzymatic transformations as
reaction SMARTS — a superset of the Daylight SMARTS notation extended with reaction
arrows and atom-map numbers — at multiple levels of chemical specificity (diameters 2,
4, 6, 8, 10, 12, and 16). Each template describes the local chemical environment of a
reaction centre for a given enzyme class, with atoms annotated by hybridization,
hydrogen count, charge, and isotope constraints. The two source CSV files
(retrorules-v3.0-metanetx.csv and retrorules-v3.0-rhea.csv) together contain a total
of 428,721 entries, all of which carry a VALID flag from the original database curation.

**Deduplication and validation.** Since the two sources partially overlap, all entries
were pooled and deduplicated by exact SMARTS string, yielding 361,751 unique templates
(66,970 cross-database duplicates removed). Each template was then parsed with RDKit
(AllChem.ReactionFromSmarts followed by Reaction.Initialize) to confirm syntactic and
chemical validity; all 361,751 templates passed this check without exception. The
resulting dataset contains exclusively single-reactant rules (n_reactant = 1 for all
entries) with between 1 and 11 product templates per rule (modal value: 2; 58.4% of
templates). Each validated template was characterised structurally: the mean number
of heavy atoms across reactant and product fragments is 45.6 (median 39, max 293),
and the mean number of bonds is 45.5 (median 38, max 296). The full data provenance
is summarised in Table 1.

**Table 1. Dataset construction summary.**

| Step | Source / Filter | Entries |
|---|---|---|
| Raw entries — MetaNetX | retrorules-v3.0-metanetx.csv | 302,879 |
| Raw entries — Rhea | retrorules-v3.0-rhea.csv | 125,842 |
| Combined (before dedup) | MetaNetX ∪ Rhea | 428,721 |
| After cross-database deduplication | exact SMARTS string match | 361,751 |
| After RDKit syntactic validation | AllChem.ReactionFromSmarts + Initialize | 361,751 (0 rejected) |
| **Final corpus (pretraining)** | | **361,751** |
| — with EC annotation | at least one EC number | 212,952 (58.9%) |
| — without EC annotation | no EC number (pretraining only) | 148,799 (41.1%) |
| Training split (90%) | random split, no stratification | 325,576 |
| Validation split (10%) | random split, no stratification | 36,175 |

**Sequence length distribution.** SMARTS character lengths are right-skewed
(median 368, mean 428, max 3,073 characters; 5th–95th percentile range: 109–954).
After tokenization, token-level lengths depend strongly on the tokenizer: the
rule-based SMARTS tokenizer yields median 101 tokens (mean 120, max 763; fertility
0.28 tokens per character), whereas the SentencePiece BPE tokenizer yields median 161
tokens (mean 190, max 1,433; fertility 0.44 tokens per character). At a maximum
sequence length of 256 tokens, the rule-based tokenizer retains 93.5% of the corpus
without truncation (SentencePiece: 77.0%); at 512 tokens, 99.8% is retained
(SentencePiece: 97.4%), with only 553 rule-based-tokenized templates (0.2%) requiring
truncation. All templates were kept in the training corpus; truncation occurs at
tokenization time during dataset collation. Length distributions for both tokenizers
are shown in Figure~\ref{fig:length_dist}; a full tokenizer comparison including
downstream task performance is presented in Section 3.1.

**Train/validation split.** The 361,751 templates were split randomly into a training
set (90%, 325,576 sequences) and a held-out validation set (10%, 36,175 sequences)
for monitoring MLM training loss and masked-token accuracy. No label information was
used at any stage of pretraining; the split is purely for convergence diagnostics.

**EC number annotations.** RetroRules associates each template with one or more Enzyme
Commission (EC) numbers encoding the catalytic function of the underlying enzyme. Of
the 361,751 validated templates, 212,952 (58.9%) carry at least one EC annotation;
the remaining 148,799 (41.1%) are unannotated and participate exclusively in
self-supervised pretraining. EC annotations are semicolon-separated strings that may
encode multiple enzyme classes per template; in such cases, all listed classes are
retained for downstream evaluation.

At the top-level (depth=1), 7 enzyme classes are represented (Table 2). The
distribution is highly imbalanced: Transferases (EC 2) account for 43.7% of annotated
templates (92,980), followed by Oxidoreductases (EC 1, 26.3%; 56,018) and Hydrolases
(EC 3, 17.0%; 36,260), while Translocases (EC 7) represent only 0.1% (218 templates).
At depth=2 (sub-subclass level), 72 distinct classes are present; the imbalance is more
pronounced, with EC 2.4 (Glycosyltransferases, 31,081) and EC 2.3 (Acyltransferases,
19,674) jointly accounting for 24% of annotated templates, whereas 14 classes contain
fewer than 50 examples. At depth=3 (sub-sub-subclass), 249 classes are represented,
with the distribution ranging from EC 2.4.1 (Hexosyltransferases, 27,916) down to
singletons and classes with fewer than 10 templates. This progressive imbalance is
an inherent property of enzymatic databases and has direct implications for the
interpretation of macro-averaged metrics in Section 3.4. An overview of annotation
coverage and label granularity across the three classification depths is given in
Table 3; the full per-class distributions at depth=2 and depth=3 are provided in
Supporting Information (Tables S1–S2).

**Table 2. EC class distribution at depth=1 (templates with EC annotation, N=212,952).**
Values indicate the number of template–class associations; templates with multiple EC
annotations contribute to each listed class.

| EC class | Name | Templates | % of annotated |
|---|---|---|---|
| EC 1 | Oxidoreductases | 56,018 | 26.3% |
| EC 2 | Transferases | 92,980 | 43.7% |
| EC 3 | Hydrolases | 36,260 | 17.0% |
| EC 4 | Lyases | 20,072 | 9.4% |
| EC 5 | Isomerases | 5,362 | 2.5% |
| EC 6 | Ligases | 9,338 | 4.4% |
| EC 7 | Translocases | 218 | 0.1% |

**Table 3. EC annotation coverage and label granularity used in downstream classification.**
"Templates with label" counts unique SMARTS carrying at least one EC annotation at that
depth; a template with a 3-level EC number (e.g., 1.1.1) contributes to depth=1, 2, and 3.

| EC depth | Label level | Templates with label | Unique classes | Most populated class | Least populated class |
|---|---|---|---|---|---|
| 1 | Class | 212,952 (58.9%) | 7 | EC 2 — Transferases (92,980) | EC 7 — Translocases (218) |
| 2 | Sub-class | 212,596 (58.8%) | 72 | EC 2.4 — Glycosyltransferases (31,081) | EC 1.9 (2) |
| 3 | Sub-sub-class | 211,162 (58.4%) | 249 | EC 2.4.1 — Hexosyltransferases (27,916) | EC 1.9.6 (2) |

> **Note for writing:** Insert the RetroRules, MetaNetX, and Rhea citations once you
> have the BibTeX keys. Standard references are: Duignan et al. (RetroRules v3),
> Moretti et al. (MetaNetX), and Lombardot et al. / Nucleic Acids Res. (Rhea).
> The depth=2 and depth=3 full distributions are large — consider moving them to
> Supporting Information and summarising only the top-5 most and least populated
> classes in the main text.

### 2.2 Tokenization

Two tokenizers were evaluated on the RetroRules corpus. The rule-based SMARTS tokenizer
was selected for pretraining; the SentencePiece model was retained as a data-driven
baseline for the TF-IDF downstream experiments. A quantitative comparison is presented
in Section~\ref{sec:tokenizer_comparison}.

#### 2.2.1 Rule-Based SMARTS Tokenizer

Reaction SMARTS strings were tokenized with a deterministic, grammar-aware tokenizer
designed to respect the Daylight SMARTS specification. Tokenization proceeds left-to-right
in two passes. First, any substring opening with `[` is consumed as a single token by
tracking bracket depth, so that arbitrarily nested recursive SMARTS
(e.g.\ `[$(C([OH]))]`) are returned as one indivisible unit; this ensures that all
atom primitives, logical operators, and atom-map numbers enclosed within a bracket
expression are preserved together. Second, all remaining characters are matched by a
priority-ordered regular expression covering: the reaction arrow (`>>`), two-letter
aliphatic atoms (`Cl`, `Br`), one-letter atoms and SMARTS primitives (`B`, `C`, `N`,
`O`, `P`, `S`, `F`, `I`; aromatic analogues `b`, `c`, `n`, `o`, `p`, `s`; `A`, `a`,
`*`), directional bonds with the "or unspecified" qualifier (`/?`, `\?`), bond symbols
(`-`, `=`, `#`, `~`, `:`, `@`, `/`, `\`), branch and component-grouping parentheses,
the disconnection dot, two-digit ring-closure labels (`%nn`), single-digit ring
closures, the logical operators `!`, `&`, `,`, and `;`, and the lone agent-separator
`>`. Priority ordering ensures that longer alternatives (e.g.\ `>>` before `>`, `Cl`
before `C`) are matched first, preventing ambiguous splits. Any unrecognized character
raises a parse error, guaranteeing that every token in the vocabulary is chemically
interpretable.

The vocabulary was built by tokenizing all 361,751 validated templates in the training
corpus, collecting token frequencies, and assigning integer IDs in frequency-descending
order. Four special tokens were prepended at fixed positions: `[PAD]` (ID 0), `[UNK]`
(ID 1), `[BOS]` (ID 2), and `[EOS]` (ID 3). A fifth special token, `[MASK]`, is
injected dynamically at training time and does not appear in the static vocabulary file.
The resulting vocabulary contains **4,465** tokens. The tokenizer achieves 100\% 
round-trip fidelity and a 0\% error rate on the full corpus.

#### 2.2.2 SentencePiece Baseline

As a data-driven alternative, a byte-pair encoding (BPE) model [CITE Sennrich et al.,
2016] was trained on the same 361,751 SMARTS strings using the SentencePiece library
[CITE Kudo and Richardson, 2018]. The corpus was written to a plain-text file (one
SMARTS per line) and passed to `SentencePieceTrainer.train` with
`character_coverage=1.0` and `normalization_rule_name="identity"` to prevent any
character normalization that could alter SMARTS syntax. A target vocabulary of
**1,000** subword units was selected as a practical size for a corpus of this scale;
the trained model achieves 98.8\% vocabulary utilization on the full dataset. Special
tokens are assigned built-in IDs: `[PAD]`=0, `[UNK]`=1, `[BOS]`=2, `[EOS]`=3,
consistent with the rule-based tokenizer. Unlike the rule-based tokenizer, the
SentencePiece model requires no chemistry knowledge and derives its units purely from
character co-occurrence statistics in the training corpus. It was used exclusively in
the TF-IDF baseline experiments and was not employed for transformer pretraining.

#### 2.2.3 Tokenizer Selection

Both tokenizers were applied to the complete corpus to characterise their intrinsic
properties (sequence length distribution, fertility, coverage at fixed maximum lengths,
and throughput) and their extrinsic utility as feature extractors. As an extrinsic
probe, TF-IDF vectors derived from each tokenizer were evaluated on EC depth=1
classification via logistic regression. Full intrinsic metrics and downstream
classification results are presented in Section~\ref{sec:tokenizer_comparison}. The
rule-based SMARTS tokenizer was selected for transformer pretraining on three grounds:
(i) its vocabulary maps one-to-one onto chemically defined token classes, enabling
interpretation of model attention and token-level loss; (ii) it produces more compact
sequences (median 101 tokens vs.\ 161 for SentencePiece), which directly reduces
truncation at fixed MAX\_LENGTH and lowers the computational cost of pretraining; and
(iii) its throughput is 3.1-fold higher, a practical consideration when tokenizing
hundreds of thousands of templates. The SentencePiece model was retained as a baseline
for the TF-IDF experiments.

### 2.3 Model Architecture

We adopt an encoder-only transformer [CITE Vaswani et al., 2017] following the
pretraining architecture of BERT [CITE Devlin et al., 2019], adapted to the SMARTS
domain. The model consists of two components: a sequence encoder used for both
pretraining and downstream inference, and a masked-language-modeling (MLM) head
attached only during pretraining and discarded thereafter.

**Sequence encoder.** Token indices produced by the rule-based SMARTS tokenizer are
mapped to a continuous representation by summing a token embedding and a learned
positional embedding, both of dimension \(d_\text{model} = 256\). Learned positional
embeddings were preferred over fixed sinusoidal encodings because the token length
distribution of SMARTS is heavily right-skewed and the model is expected to attend to
short-range chemical substructures at varying absolute positions. The summed
representation is passed through \(L = 6\) identical transformer encoder layers
[CITE Vaswani et al., 2017]. Each layer applies multi-head self-attention with
\(H = 8\) heads (head dimension 32), followed by a position-wise feed-forward network
(FFN) with inner dimension 1,024 (4× expansion), ReLU activation, residual connections,
and post-layer normalisation (Post-LN). A dropout probability of 0.1 is applied to
embeddings and within each encoder layer. The maximum supported sequence length is 512
tokens, consistent with the positional embedding table size.

**MLM head.** A lightweight projection head maps the encoder hidden states to
per-token vocabulary logits during pretraining. It consists of a dense linear layer
(\(d_\text{model} \to d_\text{model}\)), GELU activation, layer normalisation, and a
final linear projection (\(d_\text{model} \to |\mathcal{V}|\)), where
\(|\mathcal{V}| = 4\text{,}466\) is the extended vocabulary size including the
\texttt{[MASK]} token. The head is not used during embedding extraction or downstream
evaluation.

**Parameter count.** Table~\ref{tab:arch} summarises the architecture and parameter
budget. The encoder contains **6.01M** trainable parameters; adding the MLM head
yields **7.23M** parameters for the pretraining model. The full model occupies 28.9 MB
in single-precision floating point. The model was implemented using
\texttt{nn.TransformerEncoder} from PyTorch [CITE Paszke et al., 2019].

**Table~\ref{tab:arch}. Model architecture and parameter budget.**

| Component | Specification | Parameters |
|---|---|---|
| Token embedding | 4,466 × 256 | 1,143,296 |
| Positional embedding | 512 × 256 (learned) | 131,072 |
| Transformer encoder × 6 | — | 4,738,560 |
| — Self-attention / layer | 8 heads, head dim 32 | 263,168 |
| — FFN / layer | 256 → 1,024 → 256, ReLU, Post-LN | 525,568 |
| — LayerNorms / layer (×2) | 256 | 1,024 |
| **Encoder total** | | **6,012,928 (6.01M)** |
| MLM head (pretraining only) | Linear → GELU → LN → Linear | 1,214,066 |
| **Full model (pretraining)** | | **7,226,994 (7.23M)** |

### 2.4 Masked Language Modeling with Span Masking

**Objective.** The model is pretrained with the masked language modeling (MLM)
objective introduced by Devlin et al. [CITE BERT]: a fraction of tokens in each
input sequence are replaced by a special \texttt{[MASK]} token, and the model is
trained to recover the original tokens from context. The pretraining loss is the
mean cross-entropy over all masked positions, computed against the original token
identities; unmasked positions are excluded from the loss via a label-ignore index
($-100$) in PyTorch's \texttt{CrossEntropyLoss}.

**Motivation for span masking.** Standard MLM samples each token for masking
independently with a fixed probability. Applied to SMARTS, independent masking
disproportionately selects short structural tokens — branch parentheses \texttt{(},
\texttt{)}, the disconnection dot \texttt{.}, and the reaction arrow \texttt{>>} —
which together account for 4 of the 4,465 vocabulary types but appear at high
frequency in every sequence. These tokens are locally deterministic: a closed
parenthesis is uniquely determined by the matching open parenthesis, and a reaction
arrow is recoverable from the positional context alone. A model can achieve high
token-recovery accuracy on independently masked sequences by exploiting syntactic
co-occurrence patterns, without ever learning to distinguish chemically distinct
bracket atom expressions. Contiguous span masking [CITE SpanBERT, Joshi et al.,
2020] addresses this by jointly masking entire sub-phrases, forcing the model to
predict a local chemical context — an atom with its primitives and bond type, or a
reaction-centre fragment — from the surrounding sequence.

**Span sampling procedure.** At each training step, the collator applies span-based
masking independently to each sequence in the batch. A masking budget of
$B = \lfloor 0.25 \cdot L_\text{real} \rceil$ token positions is allocated per
sequence, where $L_\text{real}$ is the number of non-padding tokens ($B = 25$ at
the median sequence length of 101 tokens; $B = 30$ at the mean length of 120). Spans
are sampled iteratively until the budget is exhausted: at each iteration, a span start
is drawn uniformly from the set of unmasked real-token positions, and a span length
$\ell$ is drawn from a truncated geometric distribution with stop probability
$p = \nicefrac{1}{3}$ (mean $= 3.0$, hard cap $= 10$):

$$P(\ell = k) = \left(1 - \tfrac{1}{3}\right)^{k-1}\!\tfrac{1}{3}, \quad k = 1,\ldots,9;
\qquad P(\ell = 10) = \left(\tfrac{2}{3}\right)^{9}.$$

The expected span length under this distribution is 2.95 tokens; on a median-length
sequence approximately 8–9 spans are selected per masking pass. The mask probability
of 0.25 is elevated relative to the standard 0.15 used in BERT because contiguous
spans introduce positive autocorrelation among selected positions: each span
``spends'' multiple budget tokens in a single draw, so a higher nominal rate is needed
to achieve the same effective coverage of chemically diverse positions [CITE SpanBERT].

**Substitution rule.** The 80/10/10 substitution rule [CITE BERT] is applied once per
span rather than per token, so that all positions within a span receive the same
treatment: 80\% of spans have all their tokens replaced by \texttt{[MASK]}, 10\% have
all their tokens replaced by uniformly sampled non-special vocabulary tokens, and the
remaining 10\% are left unchanged. Applying the decision at span granularity preserves
contiguous surface-form cues, encouraging the model to use long-range chemical context
rather than local syntactic clues to reconstruct masked positions.

**Content-token accuracy.** Standard masked-token accuracy — the fraction of masked
positions predicted correctly — is inflated by the structural tokens noted above, which
are trivially predictable even without chemical understanding. We therefore report a
complementary metric, \emph{content-token accuracy}, restricted to the 4,457
chemically informative token types: bracket atom expressions, bond symbols
(\texttt{-}, \texttt{=}, \texttt{:}, \texttt{\#}, \texttt{\textasciitilde},
\texttt{@}, \texttt{/}, \texttt{\textbackslash}), ring-closure labels, directional
bond qualifiers, and logical operators. The four structural tokens
(\texttt{(}, \texttt{)}, \texttt{.}, \texttt{>>}) and the four fixed special tokens
are excluded from both the numerator and denominator. Content-token accuracy measures
the model's ability to recover chemical information, not syntactic form, and is the
primary intrinsic metric reported in Section~\ref{sec:pretraining_results}.

### 2.5 Training

The model was trained on the 325,576-template training split (Section~\ref{sec:dataset})
for 100 epochs using the MLM objective described in Section~\ref{sec:mlm}. A held-out
validation set of 36,175 templates (10\%) was used exclusively for monitoring
convergence; no validation-set information influenced model parameters or
hyperparameter selection.

**Optimiser and learning-rate schedule.** Parameters were updated with AdamW
[CITE Loshchilov \& Hutter, 2019] ($\beta_1 = 0.9$, $\beta_2 = 0.999$,
$\varepsilon = 10^{-8}$, weight decay $= 0.01$) at a peak learning rate of
$10^{-4}$. The schedule consisted of a linear warmup from 0 to $10^{-4}$ over the
first 2,000 steps (0.39\% of training), followed by cosine annealing to 0 over the
remaining 506,800 steps. Gradient norms were clipped to a maximum of 1.0 before each
parameter update.

**Implementation.** Training was carried out in PyTorch 2.6 on a single NVIDIA
A100-SXM4-40GB GPU (40 GB HBM2e) with CUDA 12.4. Mixed-precision training used the
\texttt{bfloat16} data type via PyTorch's \texttt{torch.autocast}, and the model was
compiled with \texttt{torch.compile} prior to training for additional throughput. The
DataLoader used 4 persistent worker processes with pinned memory. All other
hyperparameters are summarised in Table~\ref{tab:training}.

**Table~\ref{tab:training}. Pretraining hyperparameters.**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, wd=0.01) |
| Peak learning rate | $10^{-4}$ |
| LR schedule | Linear warmup (2,000 steps) → cosine annealing to 0 |
| Epochs | 100 (5,088 steps/epoch; 508,800 total) |
| Batch size | 64 sequences |
| MAX\_LENGTH | 512 tokens |
| Gradient clipping | max norm 1.0 |
| AMP dtype | bfloat16 |
| Training set size | 325,576 templates |
| Validation set size | 36,175 templates (10%) |
| Hardware | NVIDIA A100-SXM4-40GB (39.4 GB) |
| Training time | 6.19 h |

**Convergence.** Validation loss decreased from 1.882 at epoch 1 to 0.121 at epoch
100, with the sharpest drop in the first 10 epochs (1.882 → 0.314) as the model
learned the basic syntactic structure of SMARTS. Improvement continued progressively
but with diminishing returns: the final 10 epochs reduced validation loss by only
0.0008 (0.7\% relative), indicating near-convergence. Content-token top-1 accuracy on
the validation set reached 96.0\% at epoch 100 (overall top-1: 96.4\%; top-5: 99.5\%).
Training and validation loss curves are shown in Figure~\ref{fig:training}.

### 2.6 Embedding Extraction and Pooling

The encoder maps a variable-length token sequence to a sequence of $d$-dimensional
hidden states.  To obtain a fixed-size reaction embedding, these per-token vectors must
be pooled.  We compared two strategies.  **CLS pooling** takes the hidden state at
position 0, which corresponds to the `[BOS]` token prepended to every sequence; this
mirrors the CLS convention in BERT~\cite{Devlin2019}.  **Mean pooling** averages the
hidden states of all non-padding positions:

$$
\mathbf{e} = \frac{\sum_{t=1}^{L} m_t \, \mathbf{h}_t}{\sum_{t=1}^{L} m_t},
$$

\noindent where $\mathbf{h}_t \in \mathbb{R}^{256}$ is the hidden state at position $t$,
$m_t \in \{0,1\}$ is the attention mask, and $L$ is the padded sequence length.  Unlike
BERT, our model was pretrained with a span-masked language modelling objective without
any pooling loss; consequently, the `[BOS]` token is not explicitly trained to
accumulate sequence-level information, making mean pooling the more natural aggregate.

To validate this hypothesis, we conducted a controlled ablation before committing to a
pooling strategy.  A stratified sample of 19,996 reaction SMARTS with EC depth=1 labels
(7 classes, drawn from RetroRules v3.0) was split 80/20 into train and test sets.
Both CLS and mean embeddings were extracted from the same pretrained model in a single
forward pass.  Each embedding set was evaluated with two classifiers — an $\ell_2$-regularised
logistic regression and a two-hidden-layer MLP (256–128 units, ReLU, early stopping)
— under 10-fold stratified cross-validation, with the full training set used for the
final held-out evaluation.  Embeddings were standardised to zero mean and unit variance
before classification.  Mean pooling was selected based on this ablation; results are
reported in Section~3.3.

**Embedding extraction.** All 361,751 validated reaction SMARTS were embedded using the
final trained model (run \texttt{20260428\_160310}, max\_seq\_len = 512) with mean
pooling.  Inference was performed on a single GPU in evaluation mode (no gradient
computation), processing sequences in batches of 256.  Each SMARTS was padded or
truncated to 512 tokens.  The resulting embedding matrix has shape
$361{,}751 \times 256$ (float32, 370.4 MB) and is stored as a NumPy \texttt{.npy}
archive alongside the aligned list of SMARTS strings, version-controlled with DVC.

### 2.7 Downstream Evaluation

#### 2.7.1 EC Number Classification

Enzyme Commission (EC) numbers encode reaction chemistry in a four-level hierarchy: the
first digit identifies the reaction class (e.g., EC 1: Oxidoreductases), the second the
sub-class, and the third the sub-sub-class.  We evaluate embeddings as features for
predicting EC numbers at three levels of granularity — depth=1 (7 classes), depth=2
(72 classes), and depth=3 (237 classes) — to assess how well pretraining captures
biochemical specificity at increasing resolution.

**Dataset.** EC annotations were drawn from both RetroRules v3.0 source files
(Section~2.1).  For each template, the first EC number in the \texttt{ECS} field was
taken, truncated to $k$ components to form depth-$k$ labels.  Templates lacking a
valid EC annotation and classes with too few representatives for reliable stratified
splitting were excluded.  Joining with the validated SMARTS corpus (Section~2.1) yielded
approximately 50,000 labelled templates at each depth: 49,996 (depth=1), 49,962
(depth=2), and 49,884 (depth=3).

**Experimental protocol.** Each labelled set was partitioned into an 80\% training set
and a 20\% held-out test set using stratified random sampling (seed=42).  Performance
was estimated by 10-fold stratified cross-validation on the training partition, with
mean accuracy and F1-macro reported across folds; final per-class metrics were obtained
from the held-out test set.  Embedding features were standardised to zero mean and unit
variance before training any linear classifier.

**Methods and baselines.** Six conditions were evaluated at each EC depth:

\begin{enumerate}
\item \textbf{TF-IDF + LR (SMARTS tokenizer)} — a TF-IDF matrix (unigrams, sublinear TF,
  no lowercasing, L2-normalised) built from rule-based tokenization (Section~2.2.1),
  with a logistic regression head ($\ell_2$, $C=1.0$, balanced class weights).
\item \textbf{TF-IDF + LR (SentencePiece)} — same pipeline with SentencePiece tokens
  (Section~2.2.2) as the featuriser.
\item \textbf{Random-init + LR} — 256-dimensional embeddings from a randomly
  initialised transformer of the same architecture (Section~2.3), with no pretrained
  weights.  This control isolates the contribution of pretraining.
\item \textbf{Random-init + MLP} — same random embeddings, with a two-hidden-layer MLP
  classifier (256–128 units, ReLU, early stopping).
\item \textbf{Pretrained + LR} — mean-pooled embeddings from the pretrained model
  (Section~2.6), with a logistic regression head.
\item \textbf{Pretrained + MLP} — same pretrained embeddings with the MLP classifier.
\end{enumerate}

All embedding-based classifiers used the same MLP configuration: hidden layers of 256
and 128 units, ReLU activations, \texttt{max\_iter}=300, and early stopping with a
10\% held-out validation fraction.  Performance is reported as accuracy, F1-macro, and
F1-weighted across the 10-fold CV (mean $\pm$ standard deviation), with results
discussed in Section~3.4.

#### 2.7.2 Embedding–Structural Similarity Correlation

A well-structured embedding space should place chemically similar reactions closer
together.  To test this geometrically, we measure how well pairwise cosine similarity
in embedding space correlates with structural similarity computed independently from
reaction graph topology.

**Structural similarity.** Each reaction SMARTS was parsed into an RDKit reaction
object and fingerprinted using \texttt{rdChemReactions.CreateStructuralFingerprintForReaction}
(4,096-bit vector).  Structural similarity between two reactions was quantified by the
Tanimoto coefficient between their fingerprint vectors
(\texttt{DataStructs.TanimotoSimilarity}).

**Embedding similarity.** Embeddings were L2-normalised and pairwise cosine
similarities were computed as the upper-triangle entries of the resulting Gram matrix.

**Sampling.** Computing all $\binom{361{,}751}{2} \approx 65 \times 10^9$ pairs is
intractable.  We instead drew a random sample of 3,000 reactions (seed=42), yielding
$\binom{3000}{2} = 4{,}498{,}500$ unique pairs.  All sampled SMARTS were parsed
successfully, so no pairs were excluded.

**Statistics and visualisation.** Pearson $r$ and Spearman $\rho$ were computed
between the two similarity distributions using \texttt{scipy.stats}.  At this sample
size, $p$-values are effectively zero by construction and are reported as $p < 10^{-300}$.
Results are visualised as a hexbin density scatter plot (gridsize=50) with an ordinary
least-squares regression overlay (Figure~\ref{fig:sim_corr}), and are reported in
Section~3.5.

#### 2.7.3 UMAP Visualization

To qualitatively assess whether the embedding space exhibits a coherent biochemical
organisation, we projected all 361,751 reaction embeddings to two dimensions using
UMAP~\cite{McInnes2018}.  The reduction was performed with the following
hyperparameters: \texttt{n\_neighbors=15}, \texttt{min\_dist=0.1},
\texttt{metric=cosine}, \texttt{random\_state=42}.  UMAP was run in
\texttt{low\_memory=False} mode to maximise speed, and optimization was carried out
for 200 epochs.

The 2-D coordinates were computed once and cached as a NumPy array to avoid repeating
the expensive reduction.  Three figures were then generated from the same coordinates,
colouring each point by its EC label at depth=1, depth=2, and depth=3 respectively.
EC annotations cover 212,952 of the 361,751 embedded templates (58.9\%); templates
without an EC annotation are rendered in light grey in the background.  For depth=1,
each of the seven EC classes is assigned a distinct, colour-blind-friendly hue.  For
depth=2 and depth=3, sub-classes are shaded using sequential colormaps anchored to
their parent EC class colour (e.g., all EC 1.x sub-classes in shades of blue), so
both fine-grained and coarse structure remain legible in the same figure.  At depth=2
and depth=3, the legend is capped at the 20 most frequent sub-classes to preserve
readability.  Figures are shown in Figure~\ref{fig:umap}.

#### 2.7.4 Nearest-Neighbor Retrieval

As a qualitative probe of retrieval coherence, we retrieved the top-$K$ nearest
neighbors in embedding space for randomly selected query reactions.  Similarity was
measured by cosine distance: embeddings were L2-normalised and neighbor scores were
computed as dot products against the query vector, with the query itself excluded.
Retrieval was performed exhaustively over all 361,751 embeddings ($K=10$, seed=42).

Each retrieved neighbor was annotated with its EC label (depth=1) from RetroRules v3.0
where available, and its SMARTS string was inspected manually.  The qualitative
evaluation considers three aspects: (i) EC class agreement between the query and its
neighbors, (ii) shared reaction-centre motifs visible in the SMARTS, and (iii)
substrate-type consistency.  Results and illustrative examples are discussed in
Section~3.6.

---

## 3. Results and Discussion

[~2,000–3,000 words. Present results in the order of the Methods subsections.
Each subsection should: show the result, interpret it, compare to baseline, discuss
limitations.]

### 3.1 Dataset and Tokenization

**Dataset.** The final corpus was assembled from two RetroRules v3.0 release files:
302,879 templates from the MetaNetX source and 125,842 from the Rhea source,
giving 428,721 records in total.  After deduplication on the TEMPLATE field,
66,970 duplicates were removed, leaving **361,751 unique reaction SMARTS**.
All templates carried the \texttt{VALID=TRUE} flag in the source files; no additional
filtering was required.  Of the 361,751 templates, 212,952 (58.9\%) are annotated
with at least one EC number and are used in the supervised downstream experiments.
Dataset statistics — sequence-length distribution, EC class distribution, and the
90/10 pretraining split — are shown in Figure~\ref{fig:dataset_stats}.

**Tokenizer comparison — intrinsic metrics.** Both tokenizers were applied to all
361,751 templates and their properties are summarised in
Table~\ref{tab:tokenizer}.  The rule-based SMARTS tokenizer produces markedly more
compact sequences: median length 101 tokens versus 161 for SentencePiece BPE
(0.63× ratio), and a fertility of 0.28 tokens per character versus 0.45.  The
practical consequence for training is substantial: at \texttt{MAX\_LENGTH=512} the
rule-based tokenizer retains 99.8\% of templates intact, truncating only 553
sequences (0.2\%), whereas SentencePiece requires truncation for 2.6\% of the
corpus.  The advantage is even larger at \texttt{MAX\_LENGTH=256}, where coverage
drops to 77.0\% for SentencePiece but remains 93.5\% for the rule-based tokenizer
— a gap of 16.5 percentage points.  Throughput is 3.1$\times$ higher for the
rule-based tokenizer (9,002 vs.\ 2,940 SMARTS/s).  Both tokenizers achieve 100\%
round-trip fidelity and zero tokenization errors on the full corpus.
Token-length distributions for both tokenizers are shown in
Figure~\ref{fig:length_dist}.

\begin{table}[h]
\centering
\caption{Intrinsic tokenizer properties on the full RetroRules v3.0 corpus
($N = 361{,}751$). Coverage: fraction of templates not requiring truncation at the
given \texttt{MAX\_LENGTH}.}
\label{tab:tokenizer}
\begin{tabular}{lrr}
\hline
\textbf{Metric} & \textbf{Rule-based} & \textbf{SentencePiece (BPE)} \\
\hline
Vocabulary size             & 4,465   & 1,000  \\
Median sequence length (tokens)  & 101    & 161    \\
Mean sequence length (tokens)    & 119.8  & 190.4  \\
Max sequence length (tokens)     & 763    & 1,433  \\
Fertility (tokens / character)   & 0.276  & 0.447  \\
Coverage @ \texttt{MAX\_LENGTH=256} & 93.5\% & 77.0\% \\
Coverage @ \texttt{MAX\_LENGTH=512} & 99.8\% & 97.4\% \\
Round-trip fidelity              & 100\%  & 100\%  \\
Throughput (SMARTS/s)            & 9,002  & 2,940  \\
\hline
\end{tabular}
\end{table}

**Tokenizer comparison — extrinsic probe.** To assess whether each tokenizer
preserves biochemically discriminative information, TF-IDF vectors were constructed
from each tokenized corpus and used to train an $\ell_2$-regularised logistic
regression classifier for EC class prediction (depth=1, 7 classes, 214,007 labelled
templates, 5-fold stratified CV; Table~\ref{tab:tokenizer_ec}).  SentencePiece
achieves higher accuracy ($0.643 \pm 0.003$ vs.\ $0.598 \pm 0.001$) and F1-macro
($0.508 \pm 0.004$ vs.\ $0.448 \pm 0.003$).  This reversal relative to the
intrinsic metrics is interpretable: BPE merges frequently co-occurring character
sequences — recurring atom-map patterns such as \texttt{[C;H} or \texttt{;H0:]} —
into shared subword units, creating a TF-IDF feature space that implicitly captures
recurring reaction-centre motifs correlated with enzyme class.  The rule-based
tokenizer segments these patterns into their atomic grammatical constituents, yielding
a sparser but more granular vocabulary.

\begin{table}[h]
\centering
\caption{TF-IDF + logistic regression EC depth=1 classification (214,007 templates,
7 classes, 5-fold stratified CV, 80/20 train/test split). Mean $\pm$ std across folds.
Bold: best per metric.}
\label{tab:tokenizer_ec}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Rule-based} & \textbf{SentencePiece (BPE)} \\
\hline
Accuracy   & $0.598 \pm 0.001$ & $\mathbf{0.643 \pm 0.003}$ \\
F1-macro   & $0.448 \pm 0.003$ & $\mathbf{0.508 \pm 0.004}$ \\
F1-weighted & $0.627 \pm 0.001$ & $\mathbf{0.666 \pm 0.002}$ \\
\hline
\end{tabular}
\end{table}

The TF-IDF comparison evaluates token identity in isolation, without any contextual
representation.  As shown in Section~3.4, pretrained transformer embeddings built on
the rule-based tokenizer substantially outperform both TF-IDF baselines on all EC
depths, confirming that the tokenizer choice is superseded by the quality of the
learned contextual representation.

### 3.2 MLM Pretraining

**Convergence.** Training proceeded for 100 epochs (508,800 gradient steps) on a
single NVIDIA A100-SXM4-40GB GPU and completed in 6.19 hours.  The model converged
rapidly in the early phase: validation loss fell from 1.882 at epoch 1 to 0.314 by
epoch 10, while masked-token top-1 accuracy on the validation set rose from 54.4\%
to 91.1\% over the same interval.  Improvement continued at a progressively slower
rate thereafter, reaching a validation loss of 0.145 and top-1 accuracy of 95.8\% at
epoch 50.  The full convergence trajectory at selected epochs is shown in
Table~\ref{tab:convergence}, and training and validation loss curves are shown in
Figure~\ref{fig:training}.

\begin{table}[h]
\centering
\caption{MLM pretraining convergence trajectory at selected epochs. Content top-1
accuracy excludes structural tokens (\texttt{(}, \texttt{)}, \texttt{.}, \texttt{>>})
and special tokens from the accuracy computation.}
\label{tab:convergence}
\begin{tabular}{rcccc}
\hline
\textbf{Epoch} & \textbf{Train loss} & \textbf{Val loss} & \textbf{Val top-1} & \textbf{Content top-1} \\
\hline
  1  & 3.104 & 1.882 & 54.4\% & 54.2\% \\
  5  & 0.654 & 0.483 & 86.8\% & 85.9\% \\
 10  & 0.430 & 0.314 & 91.1\% & 90.3\% \\
 25  & 0.271 & 0.195 & 94.4\% & 93.8\% \\
 50  & 0.201 & 0.144 & 95.8\% & 95.3\% \\
 75  & 0.176 & 0.125 & 96.3\% & 95.9\% \\
100  & 0.168 & 0.121 & 96.4\% & 96.0\% \\
\hline
\end{tabular}
\end{table}

**Final performance and plateau.** At epoch 100 the model achieves a validation loss
of 0.121 and a masked-token top-1 accuracy of 96.43\%, with a top-5 accuracy of
99.52\%.  The training loss (0.168) exceeds the validation loss at convergence, as is
typical in span-masked pretraining: dropout remains active during training, and the
stochastic masking schedule presents a fresh prediction challenge at each step,
whereas validation is performed in deterministic evaluation mode.  Over the final ten
epochs, validation loss decreased by only 0.0008 (0.7\% relative), indicating that
the model had reached a plateau and that continued training beyond 100 epochs would
yield negligible gains.

**Content-token accuracy.** The content-token top-1 accuracy — computed exclusively
on chemically informative tokens (atom symbols, bond types, ring-closure digits, and
SMARTS primitives) after excluding structural punctuation (\texttt{(}, \texttt{)},
\texttt{.}, \texttt{>>}) and special tokens — reaches 96.01\% at epoch 100
(top-5: 99.43\%).  The small gap between overall accuracy (96.43\%) and content
accuracy (96.01\%) indicates that structurally uninformative tokens do not contribute
disproportionately to the overall score; the model genuinely predicts masked
atoms, bond types, and SMARTS constraint tokens with high fidelity.  This implies
that the encoder has learned a representation of SMARTS syntax that captures
chemical grammar well beyond surface-level token frequency, forming a strong basis
for the downstream embedding tasks evaluated in Sections~3.3--3.5.

### 3.3 Pooling Ablation

Table~\ref{tab:pooling} reports EC depth-1 classification accuracy and macro-F1 for all four
pooling–classifier combinations across three model sizes on the 19,996-sample subset with
10-fold stratified cross-validation (Section~\ref{sec:pooling-ablation}).
Mean pooling outperforms CLS pooling for every combination of model size and classifier head,
yielding accuracy gains of +5.4 to +5.9 percentage points (pp) for the MLP head,
equivalent to relative improvements of +6.9% to +8.0%.
The logreg head shows the same direction of effect, confirming that the advantage is
attributable to the pooling strategy rather than to a particular classifier architecture.

**Table~\ref{tab:pooling}. EC depth=1 pooling ablation (mean ± std over 10 folds, $N=19{,}996$).**

| Pooling | Head | Small (256) Acc | Small F1-mac | Medium (512) Acc | Medium F1-mac | Large (512) Acc | Large F1-mac |
|:--|:--|--:|--:|--:|--:|--:|--:|
| CLS | LogReg | 0.5924 ± 0.0099 | 0.4741 ± 0.0484 | 0.6035 ± 0.0097 | 0.5091 ± 0.0586 | 0.7159 ± 0.0070 | 0.6255 ± 0.0480 |
| CLS | MLP | 0.7367 ± 0.0124 | 0.5837 ± 0.0538 | 0.7416 ± 0.0060 | 0.5820 ± 0.0437 | 0.8009 ± 0.0105 | 0.6917 ± 0.0569 |
| Mean | LogReg | 0.6709 ± 0.0111 | 0.5557 ± 0.0519 | 0.6746 ± 0.0152 | 0.5693 ± 0.0520 | 0.7818 ± 0.0097 | 0.6806 ± 0.0558 |
| **Mean** | **MLP** | **0.7959 ± 0.0102** | **0.6464 ± 0.0490** | **0.7959 ± 0.0123** | **0.6599 ± 0.0652** | **0.8561 ± 0.0094** | **0.7343 ± 0.0611** |

The superiority of mean pooling is mechanistically consistent with the pretraining objective.
Span-masked language modelling does not impose any sequence-level aggregation task on the BOS
position, so the \texttt{[CLS]}-equivalent token develops under local contextual pressure rather
than as a global reaction summary.
Mean pooling, by contrast, averages all non-padding hidden states and thereby integrates the
full-sequence context acquired during pretraining.
A similar phenomenon is well established for general-domain BERT variants~\citep{reimers2019}
and is amplified here because individual atom-map tokens and SMARTS primitives carry distinct,
complementary semantic roles that together encode reaction type.

Larger model capacity interacts positively with mean pooling: the large model achieves
0.8561 ± 0.0094 with Mean + MLP, a +5.5 pp gain over its own CLS + MLP baseline and
+6.0 pp over the medium model under the same condition.
Based on these findings, mean pooling is adopted as the extraction strategy for all
subsequent experiments (Sections~\ref{sec:ec-classifier}–\ref{sec:nn-retrieval}).

### 3.4 EC Number Classification

[This is the main quantitative result. Report a table:]

**Table 1. EC classification accuracy (mean ± std over 10 CV folds)**

| Method | Depth=1 | Depth=2 | Depth=3 |
|---|---|---|---|
| TF-IDF (SMARTS tokenizer) | [val] | **0.3979 ± 0.0059** | **0.2493 ± 0.0079** |
| TF-IDF (SentencePiece) | [val] | **0.4617 ± 0.0083** | **0.3169 ± 0.0064** |
| Random init + logreg | [val] | **0.3330 ± 0.0076** | **0.2476 ± 0.0071** |
| Random init + MLP | [val] | **0.5356 ± 0.0088** | **0.4832 ± 0.0072** |
| Pretrained + logreg | [val] | **0.6182 ± 0.0090** | **0.5632 ± 0.0067** |
| **Pretrained + MLP** | **[PENDING]** | **0.7542 ± 0.0063** | **0.6986 ± 0.0065** |

[Also include F1 macro. Note: depth=1 result from TODO (83.8%) needs confirming with
the MAX_LENGTH=512 rerun.]

[Discussion points:
1. Pretraining gain: +21.9 p.p. over random init (depth=2 MLP); confirms MLM encodes
   reaction-type chemistry, not just parameter capacity
2. Embedding vs TF-IDF: +29.3 p.p. over best TF-IDF (depth=2); continuous embeddings
   capture structural information that bag-of-tokens misses
3. Degradation across depth: accuracy drops from depth=1 to depth=3 (83.8% → 75.4% → 69.9%)
   — this is expected as label granularity increases and class imbalance grows
4. Rare class behavior: EC classes with < 50 examples show poor recall regardless of method;
   discuss as a data limitation, not a model failure
5. Logreg vs MLP: the gap between logreg and MLP on pretrained embeddings suggests the
   embedding space is not linearly separable for all EC classes — the representation has
   structure that a nonlinear head can exploit]

### 3.5 Embedding–Structural Similarity Correlation

[Report:
- Pearson r = **0.591** (p < 1e-300)
- Spearman ρ = **0.575** (p < 1e-300)
- N = **4,498,500** pairs
- Include hexbin figure

Discussion:
- Moderate positive correlation — the model captures a real chemical signal
- What explains the ceiling? Possible reasons: (a) Tanimoto on reaction fingerprints
  measures structural overlap, while embeddings may capture functional similarity beyond
  atoms/bonds; (b) MAX_LENGTH=256 truncation loses information for long templates;
  (c) model capacity (~5M params) may be a bottleneck
- Contrast with a random-init baseline correlation (should be ~0) — [check if you ran this]
- Note: after MAX_LENGTH=512 retraining, expect this to improve]

### 3.6 UMAP Visualization

[Report qualitatively:
- Do EC classes cluster in the embedding space? (describe at depth=1, 2, 3)
- Which classes are well-separated? Which overlap?
- What does increasing label depth reveal about the embedding geometry?
- Include UMAP figures for all three depths
- Relate to the EC classification results: classes that overlap in UMAP tend to have
  lower per-class F1]

### 3.7 Nearest-Neighbor Retrieval

[Report qualitatively with 2–3 illustrative examples from nn_1191305:
- Query reaction → top-10 retrieved reactions
- Do retrieved reactions share EC class? Reaction mechanism? Substrate type?
- Discuss one interesting case where neighbors are chemically meaningful but differ in EC
  (false negative in classification, but correct in chemistry)
- This section is qualitative and supports the quantitative results]

---

## 4. Conclusions

[~300–500 words. No new results here. Structure: (1) summary of what was done and found,
(2) significance, (3) limitations, (4) future work.]

### Summary

[We have presented a self-supervised approach to learning reaction SMARTS embeddings via
MLM pretraining on RetroRules v3.0. Recap key numbers:
- Best EC depth=1: [PENDING]%, depth=2: 75.4%, depth=3: 69.9% (Pretrained+MLP)
- Pretraining gain: ~+21–31 p.p. over random init; ~+29–38 p.p. over TF-IDF
- Embedding–Tanimoto correlation: r = 0.591 on 4.5M pairs
The results demonstrate that MLM pretraining on SMARTS encodes chemically meaningful
information that generalizes to labeled downstream tasks without any task-specific supervision.]

### Significance

[What does this enable?
- Foundation model for reaction templates: the pretrained encoder can be fine-tuned for
  retrosynthesis scoring, enzyme engineering, metabolic pathway design
- Unsupervised reaction retrieval: similarity search in embedding space without labels
- Scalable to larger databases (e.g., BRENDA, Rhea) with zero labeling effort]

### Limitations

[Be honest:
1. MAX_LENGTH=256 truncates ~25% of sequences (being addressed with MAX_LENGTH=512 rerun)
2. Model capacity: ~5M parameters is modest; scaling to d_model=512 may improve results
3. RetroRules is enzymatic — generalization to non-enzymatic reaction templates is untested
4. Similarity correlation is moderate (r=0.591); the model does not perfectly recover
   structural similarity, especially in long or complex templates
5. Class imbalance at depth=3 limits evaluation robustness for rare EC classes]

### Future Work

[Concrete next steps:
1. Retrain with MAX_LENGTH=512 (in progress) and report updated benchmark numbers
2. Scale to d_model=512 to test model capacity effect
3. Fine-tune on specific downstream tasks (retrosynthesis ranking, reaction classification)
4. Extend to non-enzymatic reaction databases (USPTO, ORD)
5. Explore contrastive learning objectives to improve structural similarity alignment]

---

## Associated Content

### Supporting Information

[JCIM requires SI for detailed tables, hyperparameter sensitivity, extended figures.
Suggest including:
- Full per-class F1 tables at all three EC depths
- Pooling ablation full results table
- Training loss curves for all runs
- Additional UMAP figures
- Nearest-neighbor examples table
- Dataset statistics figures]

---

## Data and Software Availability

[JCIM requires this section. Be explicit about what is publicly available and where.
Cover three things: code, data, and model weights.]

**Code:**
[The source code for the SmartRxnEmbeddings pipeline — including preprocessing, tokenization,
model architecture, MLM training, and all evaluation scripts — is available at:
`https://github.com/jcorreia11/SmartRxnEmbeddings` (add the actual URL once the repo is public).
Include the exact commit hash or release tag used to produce the reported results.]

**Data:**
[The RetroRules v3.0 database used for training is publicly available at
`https://retrorules.org/` (cite the original paper). The preprocessed SMARTS file
(`data/processed/validated_smarts.csv`) and vocabulary (`data/processed/vocab.json`)
can be reproduced by running `dvc repro` with the provided `dvc.yaml` pipeline,
or downloaded directly via `dvc pull` from the DVC remote (add remote URL or Zenodo DOI).]

**Model weights and embeddings:**
[The pretrained transformer weights and the extracted reaction embeddings (.npy) are
tracked with DVC and available at [add Zenodo/Figshare DOI or repository release URL].
Weights can alternatively be restored by running `dvc pull` after cloning the repository.
Report the exact model run ID used for publication: `smarts_transformer_[PENDING_RUN_ID]`.]

**Reproducibility:**
[All experiments are fully reproducible via the SLURM sbatch scripts in `scripts/` with
the hyperparameters documented in `experiments-dependency-graph.md`. A `pyproject.toml`
and `uv.lock` file pin all software dependencies. Python ≥ 3.9 and uv are required.]

---

## Author Information

[Standard JCIM format. Fill in affiliations, corresponding author, ORCID.]

---

## Acknowledgments

[Include: HPC cluster access (grant/project number for the cluster allocation),
funding sources, any data providers (RetroRules team).]

---

## References

[JCIM uses ACS citation style. Key references to include:
- RetroRules v3.0 paper (Duignan et al. or original RetroRules citation)
- BERT (Devlin et al., 2019)
- SpanBERT (Joshi et al., 2020)
- RDKit (cite as software)
- ChemBERTa or MolBERT (for related work comparison)
- RXNMapper / rxnfp (Schwaller et al.) for reaction SMARTS context
- UMAP (McInnes et al., 2018)
- RetroSynthesis framework paper(s) relevant to motivation
- Any EC/enzyme function papers used for motivation]

---

## Figures Checklist

- [ ] Fig 1: Pipeline schematic (SMARTS → tokenizer → transformer → embedding space)
- [ ] Fig 2: Dataset statistics (length distribution, EC class distribution)
- [ ] Fig 3: Training/validation loss curve
- [ ] Fig 4: Pooling ablation bar chart (CLS vs mean, EC depth=1)
- [ ] Fig 5: EC classification comparison bar chart (all methods, depth=1/2/3)
- [ ] Fig 6: UMAP colored by EC class (depth=1, depth=2, depth=3 — 3 panels)
- [ ] Fig 7: Hexbin similarity correlation plot (cosine vs Tanimoto)
- [ ] Fig 8: Nearest-neighbor retrieval example (qualitative)

---

## Results Pending (before submission)

| Experiment | Status | Expected impact |
|---|---|---|
| Retrain MAX_LENGTH=512 | **TODO** | Updates all benchmark numbers; required for publication |
| EC depth=1 with MAX_LENGTH=512 | **TODO** | Expected >83.8% (current 256 result) |
| EC depth=2 with MAX_LENGTH=512 | **TODO** | Expected >75.4% |
| EC depth=3 with MAX_LENGTH=512 | **TODO** | Expected >69.9% |
| Larger model (d_model=512) | Optional | Scaling result; strengthens paper if positive |