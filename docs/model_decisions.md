# Model decisions

Paper (TARCC_ABC draft) is the starting outline, not a specification. Every choice
that departs from it, or that the paper leaves open, is a config switch in
configs/default.yaml, with the paper's choice as default where it has one.
Choices are made on the validation split; the test split is used once, at the end.

## Aims
- Aim 1: patient-level progression subtypes that shape the dynamics; new patients assigned from baseline, refined per visit.
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

## Piece 3: history encoder and posterior (history.py; paper Sec. III-D, III-E)
   - Pre-norm causal transformer (+ final LayerNorm); padding excluded via key mask;
     nested-tensor path disabled. Test: changing visit t+1 leaves h~_1..t unchanged.
   - Diagonal Gaussian posterior; log-variance clamped to [-10, 5] for stability.
   - Risk: posterior collapse. The decoder context includes ABC history (current
     outcomes) and can bypass z, which would void Aims 1 and 3. Plan: track active
     latent units each epoch; free-bits option in the loss (piece 7); test decoder
     context with vs without ABC history (piece 6).
   - Subtypes are per visit (history-conditioned states). Report at the last visit;
     measure stability across a patient's visits.

## Piece 4 (redesigned): patient-level progression subtypes (subtypes.py)
   - Replaces the paper's per-visit mixture prior (Sec. III-F): that design gives
     visit-level clusters of the latent state, likely recovering disease stage, and
     subtypes would not affect forecasts.
   - Each patient has one discrete subtype c, fixed over time, conditioning the dynamics
     f(z, c) (piece 5). Stage lives in z; subtype = how a patient progresses from there.
   - Membership network: P(c | baseline visit) assigns new patients at enrollment;
     updated by Bayes after each visit (online posterior uses only past transitions).
   - Objective: log p(Y_i) = logsumexp_c [log P(c | baseline) + sum_t l_t(c)], exact over J.
   - Prior on z: standard normal.
   - Safeguard: temperature annealing (5 -> 1) against component starvation; subtype
     embeddings initialized apart (std 1). Monitor subtype sizes each epoch.
   - Related work: latent class trajectory models (lcmm, Proust-Lima); SuStaIn's
     stage/subtype separation. Paper Sec. III-F and related work to be rewritten.

## Piece 5: latent dynamics (dynamics.py; paper Sec. III-G)
   - dz/dr = dt * f(z, e_c), RK4 8 steps; batched over subtypes x samples x patients x visits.
   - f: one tanh hidden layer (128), input [z; subtype embedding]; output layer initialized
     small (near "no change"). Modes: ode (default), sde (Euler-Maruyama), none.
   - RK4 error vs exact solution: 0.07% over a 3-year gap.
   - Risk: field may ignore the subtype (identical dynamics). Monitor endpoint distance
     between subtypes from the same start; remedy: per-subtype additive velocity term.
   - Aim 3: roll out in ~1-year steps; horizons beyond observed gaps (max 5.7 y) are exploratory.
   - Ablation "none" also silences subtypes; decoder then receives dt and the subtype directly.

## Piece 6: domain decoders (decoders.py; paper Sec. III-H)
   - Per-domain history-dependent query; category attention over the visit's tokens;
     MLP([z_delta; c^D]) -> beta-binomial per outcome. Formula verified against SciPy.
   - Departure from paper (switch): decoders do not attend to ABC_HISTORY by default.
     Current scores then reach the forecast only through z (guards against posterior
     collapse; keeps subtypes/dynamics meaningful). Aim 2 attention then answers which
     non-outcome categories shape each domain's forecast. Compare with [] on validation.
   - h~ not fed to the decoder (paper); figure variant available as a switch.
   - NPI-Q severity and count modeled independently given z (count 0 implies severity 0);
     matters only for joint simulation (Aim 3).

## Piece 7: model assembly and loss (model.py, loss.py; paper Sec. III-I)
   - Per patient: log p(Y_i) = T * logsumexp_c [(log P(c | baseline) + sum_t l_t(c)) / T];
     l_t(c) = masked beta-binomial log-likelihood of visit t+1, averaged over samples.
   - KL to N(0, I) in closed form (paper used a Monte Carlo estimate for its mixture prior).
   - Loss normalized by forecast positions; masked outcomes and padding contribute nothing
     (tests: visit-1 outcomes never targets; masked target values have no effect).
   - Free bits available (default 0); enable if active_units collapses.
   - Logged per step: active_units, subtype_sizes, subtype_divergence, nll_per_outcome,
     kl_per_position, velocity.

## Baselines (validation split; scripts/baselines.py)
| MAE | LOCF | train mean |
| PSMS | 0.64 | 2.05 | 
 IADL | 1.07 | 5.71 |  
 NPIQ_SEV | 1.69 | 2.98 |
| NPIQ_COUNT | 1.04 | 1.93 |  
MMSE | 1.89 | 5.12 |
- base_v1 (collapsed) and coupled_fb (free bits) lose to LOCF on PSMS, IADL, NPI-Q;
  only MMSE beats LOCF (1.74). Cause: z is the only path for current scores and is a
  noisy bottleneck. Candidate fix: anchor each decoder on its own domain's current score.