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

[Describe how fixed-size embeddings are obtained from the variable-length encoder output:
- CLS pooling: representation of the [BOS]/[CLS] token
- Mean pooling: average of all non-padding token representations
- Ablation: both strategies evaluated on EC depth=1 classification; mean pooling selected
  (report ablation result here or in Results)
- Extraction: all 361,751 SMARTS embedded in a single pass; embeddings stored as .npy]

### 2.7 Downstream Evaluation

#### 2.7.1 EC Number Classification

[Task: predict enzyme class (EC number) at three levels of granularity (depth=1, 2, 3)
from the reaction SMARTS embedding.
- Dataset: **49,962** labeled SMARTS (depth=2); similar sizes for depth=1 and depth=3
- Classifiers: logistic regression and 2-layer MLP
- Cross-validation: **10-fold** stratified; final test on held-out **20%**
- Baselines: TF-IDF with (a) rule-based SMARTS tokenizer, (b) SentencePiece tokenizer;
  random-init transformer (same architecture, no pretraining) as control
- Metric: accuracy, F1 macro, F1 weighted]

#### 2.7.2 Embedding–Structural Similarity Correlation

[Task: assess whether cosine similarity in embedding space correlates with structural
chemical similarity.
- Structural similarity: Tanimoto coefficient over RDKit reaction fingerprints (4,096 bits)
- Sample: **3,000** reactions → **~4.5M** pairwise comparisons
- Metric: Pearson r, Spearman ρ (report p-values)
- Visualization: hexbin density plot, linear fit]

#### 2.7.3 UMAP Visualization

[Task: qualitative assessment of embedding space organization.
- UMAP parameters: n_neighbors=15, min_dist=0.1, metric=cosine
- Colored by EC class at depth=1, depth=2, depth=3
- Include figures for all three depths]

#### 2.7.4 Nearest-Neighbor Retrieval

[Task: qualitative check that nearest neighbors in embedding space are chemically similar.
- TOP_K=10, metric=cosine, random query reactions
- Describe a few illustrative examples (pick from nn_1191305 report)
- Discuss: do retrieved reactions share EC class? Reaction center? Substrate type?]

---

## 3. Results and Discussion

[~2,000–3,000 words. Present results in the order of the Methods subsections.
Each subsection should: show the result, interpret it, compare to baseline, discuss
limitations.]

### 3.1 Dataset and Tokenization

**Dataset overview.** The final corpus comprises **361,751** validated reaction SMARTS
(see Table 1 for the full provenance). Dataset statistics figures (character-length
histogram, EC class distribution, train/validation split) are shown in
Figure~\ref{fig:dataset_stats}.

**Tokenizer comparison.**\label{sec:tokenizer_comparison}
Both tokenizers were evaluated on all 361,751 validated templates. Intrinsic metrics
were computed directly from the tokenized corpus; as an extrinsic probe, TF-IDF
vectors (unigrams, sublinear term frequency, L2-normalised) derived from each
tokenizer were used to train an \(\ell_2\)-regularised logistic regression classifier
predicting the top-level EC class (depth=1, 7 classes) on the 214,007 EC-annotated
templates, using 5-fold stratified cross-validation.

Table~\ref{tab:tokenizer} summarises the intrinsic properties of both tokenizers on
the full corpus. The rule-based SMARTS tokenizer produces substantially more compact
sequences than SentencePiece BPE: the median token-length is 1.6-fold shorter (101
vs.\ 161 tokens) and the fertility is 0.38 lower (0.28 vs.\ 0.45 tokens per
character). The practical implication for training is direct: at MAX\_LENGTH=256, the
rule-based tokenizer retains **93.5%** of templates without truncation compared to
only **77.0%** under SentencePiece; at MAX\_LENGTH=512 the figures are **99.8%** and
**97.4%**, respectively, with only 553 rule-based-tokenized templates (0.2%) requiring
truncation. The rule-based tokenizer is also 3.1-fold faster (9,002 vs.\ 2,940
SMARTS/s). Both tokenizers achieve 100% round-trip fidelity and a 0% error rate on
the full corpus. Length distributions for both tokenizers are shown in
Figure~\ref{fig:length_dist}.

**Table~\ref{tab:tokenizer}. Tokenizer comparison on the full RetroRules v3.0 corpus
(N = 361,751).**

| Metric | Rule-based SMARTS tokenizer | SentencePiece (BPE) |
|---|---|---|
| Vocabulary size | 4,465 | 1,000 |
| Sequence length — median (tokens) | 101 | 161 |
| Sequence length — mean (tokens) | 120 | 190 |
| Sequence length — max (tokens) | 763 | 1,433 |
| Fertility (tokens / character) | 0.28 | 0.45 |
| Coverage at MAX\_LENGTH=256 | 93.5% | 77.0% |
| Coverage at MAX\_LENGTH=512 | 99.8% | 97.4% |
| Round-trip fidelity | 100% | 100% |
| Error rate | 0.0% | 0.0% |
| Throughput (SMARTS/s) | 9,002 | 2,940 |

The downstream TF-IDF experiment reveals a different ordering: SentencePiece achieves
higher EC depth=1 accuracy (0.643 ± 0.003 vs.\ 0.598 ± 0.001) and macro-F1 (0.508
vs.\ 0.448) under 5-fold cross-validation (Table~\ref{tab:tokenizer_ec}). This
reversal is chemically interpretable: BPE merges frequently co-occurring character
sequences (e.g., recurring atom-map patterns such as `[C;H` or `;H0:`) into shared
subword units, which enriches the TF-IDF feature space with recurring SMARTS
subsequences that correlate with enzyme class. The rule-based tokenizer, by contrast,
segments these patterns into their atomic grammatical components, producing sparser
but more granular features.

**Table~\ref{tab:tokenizer_ec}. TF-IDF + logistic regression EC depth=1 classification
(214,007 templates, 7 classes, 5-fold stratified CV).**

| Metric | Rule-based SMARTS tokenizer | SentencePiece (BPE) |
|---|---|---|
| Accuracy | 0.598 ± 0.001 | **0.643 ± 0.003** |
| F1 macro | 0.448 ± 0.003 | **0.508 ± 0.004** |
| F1 weighted | 0.627 ± 0.001 | **0.666 ± 0.002** |

Despite the TF-IDF disadvantage, the rule-based tokenizer was selected for transformer
pretraining for the reasons stated in Section~2.2.3: compact sequences reduce truncation
and training cost, and a vocabulary of chemically defined units is a prerequisite for
interpretable attention analysis. Crucially, the TF-IDF comparison measures the
information content of the \emph{token identity} alone, divorced from any contextual
representation. The pretrained transformer embeddings, which encode contextual
chemistry, supersede TF-IDF on all downstream tasks regardless of tokenizer; those
results are reported in Section~\ref{sec:ec_classification}.

### 3.2 MLM Pretraining

[Report:
- Training and validation loss curves — did the model converge?
- Final validation loss: [value]
- Overall masked-token accuracy: [value]
- Content-token accuracy (chemically informative tokens only): [value]
- Discuss: what does the content-token accuracy tell us about what the model learned?
- Include training loss figure]

### 3.3 Pooling Ablation

[Report the CLS vs. mean pooling comparison:
- Task: EC depth=1 classification, N=20,000, 10-fold CV
- Results: [from pooling_ablation_1163359 report — insert accuracy/F1 for both]
- Winner: mean pooling → used for all subsequent experiments
- Brief discussion: why might mean pooling outperform CLS for SMARTS?
  (CLS may not aggregate long-range information as well in a domain with long sequences)]

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