# JCIM Paper Template — SmartRxnEmbeddings

> **How to use this template**
> Text in `[square brackets]` = instructions to the author (delete before submitting).
> Text in `**bold**` = already-known values you can paste in directly.
> `[PENDING]` = results not yet available (requires MAX_LENGTH=512 retraining).
> JCIM word limits: Abstract ≤ 250 words; full article typically 6,000–10,000 words.

---

## Title

**Learning Reaction Templates Representations from SMARTS via Span-Masked Language Modelling**

**Author:** João Correia†‡

†INESC TEC — Institute for Systems and Computer Engineering, Technology and Science, Porto, Portugal
‡Department of Informatics Engineering, Faculty of Engineering of the University of Porto, Porto, Portugal

E-mail: jfcorreia@fe.up.pt

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

Reaction templates were retrieved from RetroRules [CITE], a database of enzymatic reaction rules derived from the biochemical databases MetaNetX [CITE] and Rhea [CITE], as well as from the USPTO [CITE] dataset of organic chemistry reactions. RetroRules contains chemical reaction transformations encoded as reaction SMARTS at multiple levels of chemical specificity. As this work focuses exclusively on biochemical reactions, only templates derived from MetaNetX and Rhea were used, yielding a total of 428,721 entries.

**Deduplication and validation.** Since there is some overlap from the two sources, all entries were deduplicated by exact SMARTS string, yielding 361,751 unique templates. Each template was then parsed with RDKit [CITE] to confirm syntactic and chemical validity, confirming that all 361,751 templates were valid. The full data provenance is summarised in Table 1.

**Table 1. Dataset construction and usage summary for the RetroRules-derived reaction SMARTS corpus.** Percentages are reported relative to the combined raw source set unless otherwise noted.

| Step | Subset | Entries | Share | Use |
|---|---|---|---|---|
| Raw input — MetaNetX | MetaNetX | 302,879 | 70.6% | Source corpus |
| Raw input — Rhea | Rhea | 125,842 | 29.4% | Source corpus |
| Combined | Combined | 428,721 | 100.0% | Before filtering |
| Deduplicated | Unique SMARTS strings | 361,751 | 84.4% | Pretraining corpus |
| Validated | RDKit-valid reactions | 361,751 | 84.4% | Final corpus |
| Annotated | ≥1 EC number | 212,952 | 58.9%^a | Downstream evaluation |
| Unannotated | No EC number | 148,799 | 41.1%^a | Pretraining only |
| Random split — Training | Training | 325,576 | 90.0%^a | Model fitting |
| Random split — Validation | Validation | 36,175 | 10.0%^a | MLM monitoring |
| **Final** | **Validated unique SMARTS** | **361,751** | **84.4%** | **All experiments** |

^a Percentage relative to the validated corpus.

**EC number annotations.** Each template may be associated with one or more Enzyme Commission (EC) numbers. Among the validated templates, 212,952 (58.9%) have at least one EC annotation, while 148,799 (41.1%) are unannotated and thus used exclusively for self-supervised pretraining. For downstream classification, EC labels are considered at three hierarchical levels (depths 1 to 3). At depth 1, 7 enzyme classes are represented, depth 2 and 3 contain 72 and 249 distinct classes, respectively. Detailed EC class distributions and depth-specific label coverage are provided in Supporting Information Tables S1 and S2.

**Train/validation split.** The 361,751 templates were split randomly into a training set (90%, 325,576 sequences) and a hold-out validation set (10%, 36,175 sequences) for monitoring model training loss and masked-token accuracy. No label information was used at any stage of pretraining, with the split being purely used for convergence diagnostics.

### 2.2 Tokenization

Two tokenizers were applied and evaluated on the RetroRules corpus. A rule-based SMARTS tokenizer was used for pretraining, while a SentencePiece model served as a data-driven baseline for TF–IDF experiments. Sequences were not filtered by length. Instead, truncation was applied at tokenization time during dataset collation, with maximum sequence lengths of 256 and 512 tokens evaluated in subsequent experiments. A quantitative comparison is provided in Section 3.1.

#### 2.2.1 Rule-Based SMARTS Tokenizer

Reaction SMARTS were tokenized using a custom deterministic pipeline aligned with the Daylight SMARTS guidelines. Tokenization proceeds left-to-right in two passes. First, substrings enclosed in `[...]` are extracted as single tokens by tracking bracket depth, preserving complete atom expressions, including nested elements. Second, remaining characters are matched using a priority-ordered regular expression covering: the reaction arrow (`>>`), two-letter atoms (`Cl`, `Br`), one-letter atoms and primitives (`B`, `C`, `N`, `O`, `P`, `S`, `F`, `I` and aromatic analogues; `A`, `a`, `*`), bond symbols (`-`, `=`, `#`, `~`, `:`, `@`, `/`, `\`), directional bonds (`/?`, `\?`), parentheses, disconnection dots, ring closures (`%nn`, digits), logical operators (`!`, `&`, `,`, `;`) and the separator `>`. Longer patterns are matched first (e.g. `>>` before `>`) to avoid ambiguity. Unrecognized characters raise a parse error.

The vocabulary was constructed from all 361,751 validated templates by frequency-based indexing. Special tokens `[PAD]`, `[UNK]`, `[BOS]`, and `[EOS]` were assigned fixed IDs (0–3), while `[MASK]` is introduced dynamically during training. The final vocabulary contains 4,465 tokens and achieves 0% reconstruction error for all reaction SMARTS with no parsing errors. An example of the rule-based SMARTS tokenizer is shown in Figure 1.

#### 2.2.2 SentencePiece Tokenizer

SentencePiece [CITE] was used as an unsupervised tokenizer to provide a data-driven baseline against the rule-based SMARTS tokenizer. SentencePiece learns a fixed-size vocabulary directly from raw input strings and supports standard subword segmentation algorithms, including byte-pair encoding (BPE) [CITE], without requiring language-specific pre- or post-processing. Here, a BPE SentencePiece model was trained on the same 361,751 validated reaction SMARTS used for the rule-based tokenizer. To preserve SMARTS syntax, training used `character_coverage=1.0` and `normalization_rule_name="identity"`. The vocabulary size was fixed at 1,000 subword units, yielding 98.8% vocabulary utilization. The same special-token convention was used for compatibility with downstream experimentation (`[PAD]`=0, `[UNK]`=1, `[BOS]`=2, and `[EOS]`=3).

### 2.3 Model Architecture

We used an encoder-only transformer [CITE Vaswani et al., 2017] following the pretraining architecture of BERT [CITE Devlin et al., 2019], adapted to the SMARTS domain. The model consists of a sequence encoder used for both pretraining and inference and a masked language modeling (MLM) head used only during pretraining.

**Sequence encoder.** Token indices produced by the rule-based SMARTS tokenizer are mapped to continuous representations by summing token and learned positional embeddings, both of dimension 256. Learned positional embeddings were used instead of fixed sinusoidal encodings to better handle the right-skewed SMARTS length distribution and position-dependent chemical patterns.

The resulting representations are processed by 6 transformer encoder layers. Each layer applies multi-head self-attention with 8 heads (head dimension of 32), followed by a position-wise feed-forward network (FFN) with inner dimension 1,024, ReLU activation, residual connections and post-layer normalization. Dropout with probability 0.1 is applied to embeddings and within each layer. The maximum sequence length is 512 tokens.

**MLM head.** During pretraining, encoder outputs are passed through a masked language modeling head that predicts the original token identity at each masked position. The head first transforms each hidden state with a linear layer, applies a GELU nonlinearity and layer normalization, and then projects the result to the full tokenizer vocabulary. This prediction head is used only for self-supervised pretraining and is discarded during downstream evaluation. The overall model structure is summarized in Figure 2. A component-wise parameter breakdown is also provided in Supporting Information Table S3.

### 2.4 Masked Language Modeling with Span Masking

**Pretraining.** The model is pretrained using a masked language modeling (MLM) objective where a subset of input tokens is replaced with the special `[MASK]` token and the model is trained to recover the original tokens from context. The loss is the mean cross-entropy over masked positions only.

**Span masking.** Independent token masking disproportionately selects frequent structural tokens (e.g. `(`, `)`, `.`, `>>`), which are locally deterministic and trivially predictable from syntax alone. To mitigate this, we adopt contiguous span masking [CITE SpanBERT, Joshi et al., 2020], which masks short token sequences and encourages reconstruction of chemically meaningful contexts (e.g. atom expressions or reaction fragments).

**Span sampling.** For each sequence, a masking budget of $B = \lfloor 0.25 \cdot L_\text{real} \rfloor$ tokens is allocated, where $L_\text{real}$ excludes padding. Spans are sampled iteratively until the budget is exhausted. At each step, a start position is drawn uniformly from unmasked tokens, and a span length $\ell$ is sampled from a truncated geometric distribution with stop probability $p = \frac{1}{3}$ (mean ≈ 3, maximum 10):

$$P(\ell = k) = \left(1 - \tfrac{1}{3}\right)^{k-1}\!\tfrac{1}{3}, \quad k = 1,\ldots,9;
\qquad P(\ell = 10) = \left(\tfrac{2}{3}\right)^{9}.$$

This yields an expected span length of ≈ 2.95 tokens and ∼8–9 spans per median-length sequence. The masking rate (0.25) is higher than the standard 0.15 (used in BERT) to compensate for the autocorrelation introduced by consecutive token masking.

**Substitution rule.** The 80/10/10 replacement strategy [CITE BERT] is applied once per span rather than per token, so that all positions within a span receive the same treatment. 80% of spans are replaced with `[MASK]`, 10% with random tokens, and 10% left unchanged. Applying substitutions per span preserves contiguous surface structure and reduces reliance on local syntactic cues.

**Content-token accuracy.** Standard masked-token accuracy is inflated by trivially predictable structural tokens. We therefore report *content-token accuracy*, computed over the 4,457 chemically informative tokens (atom expressions, bonds, ring closures, directional qualifiers, and logical operators), excluding structural tokens (`(`, `)`, `.`, `>>`) and special tokens. This metric better reflects recovery of chemical information and is used as the primary intrinsic evaluation measure (Section 3.2).

### 2.5 Training

The model was trained on the 325,576-template training split for 100 epochs using the MLM objective described in Section 2.4. A held-out validation set of 36,175 templates (10%) was used exclusively for monitoring convergence and did not influence model parameters or hyperparameter selection.

**Optimisation.** Parameters were updated using AdamW [CITE Loshchilov & Hutter, 2019] with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, and weight decay = 0.01. The learning rate followed a linear warmup from 0 to $10^{-4}$ over the first 2,000 steps, followed by cosine annealing to 0 over the remaining training steps. Gradient norms were clipped to 1.0.

**Implementation.** Training was performed in PyTorch [CITE] on a single NVIDIA A100-SXM4-40GB GPU (40 GB) with CUDA 12.4. Mixed-precision training used the `bfloat16` datatype via `torch.autocast`, and the model was compiled with `torch.compile`. Data loading used 4 persistent workers with pinned memory.

**Hyperparameters.** All training hyperparameters are summarised in Table 2.

**Table 2. Pretraining hyperparameters.**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, wd=0.01) |
| Peak learning rate | $10^{-4}$ |
| LR schedule | Linear warmup (2,000 steps) → cosine decay |
| Epochs | 100 (5,088 steps/epoch; 508,800 total) |
| Batch size | 64 sequences |
| Max sequence length | 512 tokens |
| Gradient clipping | max norm 1.0 |
| Precision | bfloat16 |
| Training set size | 325,576 |
| Validation set size | 36,175 (10%) |
| Hardware | NVIDIA A100-SXM4-40GB |
| Training time | 6.19 h |

### 2.6 Embedding Extraction

All 361,751 validated reaction SMARTS were embedded using the final trained model with mean pooling. Inference was performed on a single GPU in evaluation mode (no gradient computation), processing sequences in batches of 256. Each SMARTS was padded or truncated to 512 tokens.

### 2.7 Downstream Evaluation

#### 2.7.1 EC Number Classification

Enzyme Commission (EC) numbers classify enzyme-catalysed reactions hierarchically. The first three levels denote the class, subclass, and sub-subclass, whereas the fourth specifies the enzyme entry, typically distinguished by substrate specificity or catalytic transformation. We evaluate embeddings as features for predicting EC numbers at the first three levels of granularity to assess how well pretraining captures biochemical specificity at increasing resolution.

**Dataset.** EC labels were retrieved from the EC-annotated subset of the dataset described in Section 2.1. For each template, the first EC number was taken and truncated to $k$ components to form depth-$k$ labels. Classes with fewer than 10 representatives were excluded to ensure all folds in stratified cross-validation contained at least one example per class. This yielded approximately 50,000 labelled templates at each depth: 49,996 (depth = 1), 49,962 (depth = 2), and 49,884 (depth = 3).

**Experimental protocol.** Each labelled set was partitioned into an 80% training set and a 20% held-out test set using stratified random sampling. Performance was estimated by 10-fold stratified cross-validation on the training partition, with mean accuracy and F1-macro reported across folds. Final per-class metrics were obtained from the held-out test set. Embedding features were standardised to zero mean and unit variance before training any linear classifier.

**Methods and baselines.** We compared the proposed pretrained embeddings against two classes of baselines. First, sparse lexical baselines were constructed by applying TF-IDF weighting to either the rule-based SMARTS tokens or SentencePiece tokens. In both cases, unigram features were used with sublinear term-frequency scaling, no lowercasing, and $\ell_2$ normalisation, followed by an $\ell_2$-regularised logistic regression classifier ($C = 1.0$) with balanced class weights.

Second, embedding-based baselines were used to isolate the effect of MLM pretraining. Reaction SMARTS were encoded either with the pretrained transformer or with a randomly initialised transformer of identical architecture, yielding 256-dimensional mean-pooled embeddings in both cases. Each embedding set was evaluated with two downstream classifiers: logistic regression and a two-hidden-layer MLP. The MLP used hidden layers of 256 and 128 units, ReLU activations, `max_iter = 300`, and early stopping with a 10% held-out validation fraction.

Together, these comparisons separate the contribution of token-level lexical features, transformer architecture, MLM pretraining, and nonlinear classification. Performance is reported as accuracy, F1-macro, and F1-weighted across the 10-fold CV (mean ± standard deviation), with results discussed in Section 3.4.

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

Table~\ref{tab:ec-classification} reports mean accuracy and macro-averaged F1 over 10-fold cross-validation for all six methods at EC depths 1 (7 classes, $n$=49,996), 2 (72 classes, $n$=49,962), and 3 (237 classes, $n$=49,884).
Classes with fewer than 10 representatives were excluded from each depth prior to stratified splitting.

**Table X. EC classification results (mean ± std, 10-fold CV). Best result per column in bold.**

| Method | D=1 Acc. | D=2 Acc. | D=3 Acc. | D=1 F1-mac | D=2 F1-mac | D=3 F1-mac |
|:--|--:|--:|--:|--:|--:|--:|
| TF-IDF (SMARTS tokenizer) | 0.577 ± 0.007 | 0.398 ± 0.006 | 0.249 ± 0.008 | 0.420 ± 0.011 | 0.237 ± 0.017 | 0.148 ± 0.011 |
| TF-IDF (SentencePiece) | 0.623 ± 0.010 | 0.462 ± 0.008 | 0.317 ± 0.006 | 0.470 ± 0.026 | 0.292 ± 0.023 | 0.188 ± 0.011 |
| Random init + LR | 0.488 ± 0.005 | 0.337 ± 0.005 | 0.252 ± 0.008 | 0.348 ± 0.012 | 0.182 ± 0.014 | 0.138 ± 0.011 |
| Random init + MLP | 0.697 ± 0.004 | 0.541 ± 0.006 | 0.486 ± 0.007 | 0.464 ± 0.011 | 0.257 ± 0.022 | 0.198 ± 0.019 |
| Pretrained + LR | 0.680 ± 0.005 | 0.618 ± 0.009 | 0.563 ± 0.007 | 0.529 ± 0.032 | 0.437 ± 0.024 | 0.394 ± 0.021 |
| **Pretrained + MLP** | **0.838 ± 0.004** | **0.754 ± 0.006** | **0.699 ± 0.007** | **0.684 ± 0.053** | **0.521 ± 0.029** | **0.428 ± 0.020** |

**Pretraining gain.**
Comparing Pretrained+MLP against the architecturally identical Random+MLP isolates the contribution of span-masked pretraining from model capacity alone.
The accuracy gain is +14.1 p.p. at depth=1, widening to +21.3 p.p. at depths 2 and 3, confirming that the pretrained representations encode chemically meaningful information rather than acting as random projectors.
The disproportionate gain at finer EC granularities suggests that pretraining is especially valuable when discriminating among closely related enzymatic sub-functions.

**Contextual embeddings vs. bag-of-tokens.**
Pretrained+MLP surpasses the strongest TF-IDF baseline (SentencePiece) by +21.5 p.p. at depth=1, a gap that grows to +29.3 p.p. and +38.2 p.p. at depths 2 and 3.
This increasing margin demonstrates that continuous, context-sensitive representations capture sub-structural distinctions in reaction SMARTS that term-frequency features cannot, and that this advantage compounds as label granularity increases.

**Linear separability of the embedding space.**
The MLP head consistently outperforms logistic regression on pretrained embeddings by 15.8 p.p. at depth=1 and 13.5–13.6 p.p. at depths 2–3.
This indicates that the learned representation organises EC classes in a non-linearly separable manifold, and that a linear probe underestimates embedding quality — a property shared with contextual language model representations in other domains.

**Degradation with label granularity and class imbalance.**
Accuracy declines monotonically from 83.8% (7 classes) to 75.4% (72 classes) to 69.9% (237 classes), consistent with the combinatorial explosion of fine-grained labels and growing example scarcity.
The divergence between accuracy and macro-F1 is more pronounced, falling from 68.4% to 52.1% to 42.8%, reflecting poor recall on rare sub-subclasses.
At depth=1, per-class analysis of Pretrained+MLP reveals high fidelity for well-populated classes — Transferases (F1=0.917) and Hydrolases (F1=0.776) — while Translocases (EC 7), represented by only three test examples after stratified splitting, achieves F1=0.0 across all methods.
This failure is a data artefact rather than a model limitation: the RetroRules corpus contains only 70 Translocase templates out of 212,952 annotated rules, making reliable classification of this class intractable without additional data.

### 3.5 Embedding–Structural Similarity Correlation

To assess whether the learned representations encode chemical structure beyond SMARTS token syntax, we computed pairwise cosine similarity in embedding space alongside pairwise Tanimoto similarity over RDKit structural reaction fingerprints (4096 bits) for all $\binom{3000}{2} = 4{,}498{,}500$ reaction pairs drawn from 3,000 randomly sampled templates (seed=42, $n_\text{total}=361{,}751$).
The Tanimoto distribution is identical across model sizes (mean=0.338, std=0.160) since it depends only on the reaction SMARTS.

**Table~\ref{tab:sim-corr}. Embedding–structural similarity correlation by model size.**
All p-values $< 10^{-300}$ ($n = 4{,}498{,}500$ pairs, same reaction sample for all rows).

| Model | max\_len | Pearson $r$ | Spearman $\rho$ | Cosine mean ± std |
|:--|--:|--:|--:|--:|
| Small (6L, $d$=256) | 256 | 0.606 | 0.586 | 0.741 ± 0.096 |
| Medium (6L, $d$=256) | 512 | 0.591 | 0.575 | 0.757 ± 0.093 |
| Large (8L, $d$=512) | 512 | 0.574 | 0.563 | 0.688 ± 0.103 |
| Random init (medium arch.) | 512 | [PENDING] | [PENDING] | — |

**Chemical signal in the embeddings.**
All pretrained models show a moderate-to-strong positive correlation ($r \in [0.574, 0.606]$, $p < 10^{-300}$), confirming that span-masked pretraining drives the encoder to place structurally similar reactions close together in embedding space — a property not enforced by the training objective, which operates on masked token prediction.

**Capacity and the correlation ceiling.**
$r$ decreases monotonically from small to large ($0.606 \to 0.591 \to 0.574$) even as model capacity grows and EC classification performance improves.
This apparent paradox is resolved by noting that structural fingerprints encode atom-neighbourhood environments present in reactant and product templates, providing a fixed, bounded view of chemical similarity.
Larger, deeper models develop representations that increasingly diverge from this structural proxy — capturing functional and mechanistic aspects of reactivity that structural overlap does not detect.
A similar pattern is observed in natural language: larger language models encode semantic similarity that diverges from lexical overlap measures.

**Random-init baseline.**
To confirm that the observed correlation originates from learned weights rather than the transformer inductive bias, an identical analysis was run on models with the same architecture but randomly initialised weights.
The results ($r = \mathbf{[PENDING]}$ for the medium architecture) will be inserted here once the baseline job completes.
A near-zero baseline would establish a direct causal link between pretraining and the structural signal reported above.

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

The SI contains the following materials:

- **Table S1.** EC class distribution at depth=1 (templates with EC annotation, N=212,952).
- **Table S2.** EC annotation coverage and class granularity at depths 1–3.
- **Table S3.** Model architecture and parameter budget (encoder + MLM head).
- **Section: EC Annotation Coverage.** [PENDING — extended depth=2 and depth=3 distributions]
- **Section: Extended Model Evaluation.** [PENDING — full per-class F1 tables, pooling ablation results, training curves]
- **Section: Additional Qualitative Examples.** [PENDING — nearest-neighbor retrieval examples]

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