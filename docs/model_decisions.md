# Model decisions

Paper (TARCC_ABC draft) is the starting outline, not a specification. Every choice
that departs from it, or that the paper leaves open, is a config switch in
configs/default.yaml, with the paper's choice as default where it has one.
Choices are made on the validation split; the test split is used once, at the end.

## Aims
- Aim 1: progression subtypes (mixture prior over the latent state).
- Aim 2: next-visit forecasting of function (PSMS, IADL), behavior (NPI-Q severity,
  count) and cognition (MMSE), with uncertainty and per-domain category attention.
- Aim 3: what-if simulation. Presented as model-based scenario exploration unless a
  causal method (e.g. g-computation) is added; requires forecasts to flow through z.

## Piece 1: config and data (configs/default.yaml, data.py)
- Hyperparameters from paper Table I. Open values (TODO, tune on validation):
  prior warm-up 10 epochs, KL annealing 20 epochs; J starts at 4, selected from 2-6.
- Training batches shuffled with a fixed seed; validation/test keep their order.

## Piece 2: visit encoder (encoder.py; paper Sec. III-B, III-C)
- Category tokens (Eq. 1), fusion attention with a global query (Eq. 2),
  Time2Vec on (t_abs, dt) (Eq. 3), summed static embeddings (index 0 = missing).
- Additions to the paper:
  - token_norm (default on): LayerNorm per category token. Dot-product attention
    favors large tokens; categories differ 42x in input size (Biomarker 336 vs
    Genomic 8 encoder inputs). Normalizing makes attention reflect content, which
    Aim 2's category attention depends on.
  - Static embeddings initialized with std 0.02 (PyTorch default 1), so the sum of
    5 tables does not dominate h_t early in training.
  - category_dropout (default 0, tune e.g. 0.1): hides whole categories during
    training, encoded as unobserved. Counters reliance on one category given
    uneven coverage (ABC history 88% observed; RBM 16% of patients).
- Kept from paper: one affine encoder per category (more capacity risks
  overfitting with ~7,700 training pairs); fusing tokens before the transformer
  (decoders still attend over the full token bank).
- Ablation switches: use_masks, use_time2vec, drop_categories.

## Open design questions (decide with validation results)
- Decoder input: [z; c] (paper; required for Aim 3) vs [h; z; c] (figure).
- Likelihood per outcome: beta-binomial (paper) vs ordinal for PSMS, IADL,
  NPI-Q severity. NPI-Q count is capped at 12, so negative binomial is unsuitable.
- RBM panel (116 markers, 16% of patients): keep or drop, by panel ablation.