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

## ABC history-influence map (ownS4, 3 seeds; scripts/abc_map.py)
   % CRPS increase when one domain's history is shuffled across patients (mean ± sd):
   | history \ forecast | PSMS | IADL | NPIQ_SEV | NPIQ_COUNT | MMSE |
   | A function  | 3.0±2.2 | 8.1±3.5 | 0.9±0.2 | 1.1±0.2 | 1.0±0.4 |
   | B behavior  | ~0 | ~0 | 25.2±3.9 | 16.0±1.8 | 0.5±0.5 |
   | C cognition | 8.8±0.8 | 18.2±3.6 | 2.5±0.8 | 3.5±1.1 | 16.8±2.3 |
   - Cognition -> function strongest (exceeds function's own history beyond the anchor);
     cognition -> behavior weak; function -> others marginal; behavior -> none.
   - Learned within-gap coupling W ~ 0: influence acts via inference of each domain's state
     from the full history. Interpretation: unique predictive (Granger-style) influence,
     not causation; conservative where information is shared with other inputs.
   - To do: robustness (no_cdr, scores_only), retrain-based confirmation of C -> A,
     item-level behavior (apathy).
   - Robustness: without CDR boxes, C->A unchanged (PSMS 8.1±0.4, IADL 18.0±3.3); C->B
  disappears (0.7/0.8%: it came from CDR ratings, not reported). MMSE alone carries ~1/4
  of C->A (2.5±1.1 / 4.0±0.8): detailed neuropsych profile predicts function beyond MMSE.


## Choice of J (subtypes), 3 seeds each
   - CRPS equivalent for J = 2, 3, 4 (differences within seed noise).
   - J = 3 recovers the same three subtypes in every seed: stable (33-47%), steady decliner
     (32-47%, behavior flat), behavior-escalating decliner (14-24%, NPI-Q severity +1.0 to
     +1.6/yr, fastest functional and cognitive decline), also within MCI, dementia, normal.
   - J = 2 merges the behavior-escalating group into general decliners; J = 4 splits steady
     decliners inconsistently across seeds.
   - ARI 0.41-0.45 with consistent profiles: soft phenotypes on a continuum; final
     assignments via consensus across seeds; report probabilities.

## Ablation study (scripts/ablations.py; 16 variants x 3 seeds vs base J3; validation CRPS)
% change vs base (positive = ablation worse); * = beyond ~2 sd of seed noise

| Variant | PSMS | IADL | NPIQ_SEV | NPIQ_COUNT | MMSE |
|---|---|---|---|---|---|
| no_ode | -2.2 | -2.7 | 1.3 | 1.2 | -0.9 |
| sde | -1.5 | -1.3 | 0.8 | 1.1 | 0.4 |
| no_time2vec | 0.0 | -1.1 | 0.4 | 0.5 | 2.2 |
| no_masks | 0.5 | 4.1* | 0.8 | 1.4 | 3.1* |
| no_anchor | 2.3 | 1.3 | -2.0 | -1.8 | 1.7 |
| no_coupling | -0.1 | 0.2 | -0.7 | -0.6 | 0.6 |
| deterministic | 0.2 | 0.4 | -0.6 | 0.0 | 1.3 |
| one_subtype | -1.4 | -1.4 | -1.5 | -0.6 | -0.6 |
| paper_context | -3.5 | -4.2* | -0.3 | -0.1 | -1.8* |
| drop_demographic | 1.7 | -0.3 | -0.1 | 0.2 | 0.5 |
| drop_clinical | -1.2 | 0.7 | -1.0 | -1.5 | -1.7* |
| drop_biomarker | -4.4 | -2.1 | -2.0 | -1.8 | -0.2 |
| drop_genomic | -1.0 | -0.5 | 0.3 | 0.3 | -0.6 |
| drop_social | -1.5 | -0.8 | 1.0 | 0.7 | -0.2 |
| drop_caregiver | -1.2 | -1.6 | -0.3 | -0.2 | -0.9 |
| drop_abc_history | 2.4 | 8.2* | 1.0 | 1.0 | 9.8* |

Base J3 CRPS: PSMS 0.499, IADL 0.850, NPIQ_SEV 1.244, NPIQ_COUNT 0.771, MMSE 1.144.
- Accuracy driven by ABC history and observation masks.
- paper_context (decoders also attend to ABC-history tokens) is better on IADL/MMSE:
  adopt if latent blocks stay used (check with diagnose.py).
- ODE, subtypes, SDE, coupling, deterministic, Time2Vec and most categories: no significant
  one-step effect; they serve Aims 1 and 3. Biomarkers do not help (sparse coverage).
- To do: GRU and linear baselines; per-domain anchor (no anchor helps NPI-Q slightly, n.s.)