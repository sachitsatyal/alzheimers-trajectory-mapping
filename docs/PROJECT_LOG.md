# Alzheimer's Trajectory Mapping (TARCC ABC) — Project Log

Complete record of the project from environment setup to the current state.
Purpose: (1) brief a new Claude conversation in this Project, (2) personal study reference,
(3) source material for the paper's Data, Method and Results sections.

**Version 1.3 — last updated: 28 October 2026.** (Version history: Section 19.)
**Status:** preprocessing done; model built and selected (J3); Aim 1 and Aim 2 results in hand; ablation study
complete; two professor meetings held (9 and ~20 October); next work = making the ABC (function–behavior–cognition)
relationships clear (Section 13); Aim 3 not started.

---

## How this log is maintained

- This file is the single source of truth. It lives in the claude.ai Project ("TARCC Project") and should also be
  committed to the repository as `docs/PROJECT_LOG.md`.
- **After every major decision**, add an entry to the **Decision Log (Section 16)** using the template there, and update
  any section whose facts changed (results tables, configuration, open items). Change the "Last updated" date.
- Major decision = anything that changes the data pipeline, the model, the evaluation, the reported results, the aims
  or the paper plan. Small code fixes go in git commit messages, not here.
- Numbers are on the **validation split** unless stated. The **test split has not been used**.
- **Versioning:** every update after a decision raises the version (minor = new decision/result, e.g. 1.1 → 1.2;
  major = new phase or aim, e.g. 1.x → 2.0) and adds one line to the **Version History (Section 19)**.
  Claude keeps the Project copy (`claude/PROJECT_LOG.md`) current after every decision from 28 Oct 2026 onward;
  the repository copy (`docs/PROJECT_LOG.md`) is refreshed from it and committed.

---

## 0. How to use this file in a new conversation

Upload or paste this file and say what you want to do next. Key facts a new thread needs:
- Base model = **J3 configuration** (`configs/default.yaml`), runs `J3_s1`, `J3_s2`, `J3_s3`.
- The paper draft (TARCC_ABC_4.pdf) is an **outline, not a specification**; many parts were redesigned (Section 6).
- Rules: decisions on the **validation** split; **test split used once**, at the end; design choices confirmed with
  **3 seeds**; architecture choices are **config switches**; differences count as real only if > ~2 SD of seed noise.
- Work happens on a Windows remote workstation without admin rights (Section 2).
- Working style: incremental steps, one at a time, each with a short description; explanations in plain language.

---

## 1. Aims

| Aim | Goal | Status |
|---|---|---|
| **Aim 1** | Patient-level **progression subtypes** that a new patient can be assigned to (from baseline, refined per visit) | Done: 3 subtypes, consensus across seeds, replicated on held-out patients; precision (PPV) of early identification still to add |
| **Aim 2** | Next-visit **forecasting** of function (PSMS, IADL), behavior (NPI-Q severity, count) and cognition (MMSE), with uncertainty and per-domain category attention; plus the **ABC map** (how the domains influence each other) | Forecasting beats baselines on all 5 outcomes; ABC map shows a robust cognition → function link; other links weak → being addressed (Section 13) |
| **Aim 3** | **What-if simulation**: change a modifiable input and project trajectories | Not started; which inputs are modifiable is an open question for the professor |

The project started as "ABC trajectory mapping": it must address how the **A**ctivities of daily living (function),
**B**ehavior and **C**ognition domains influence each other, not only forecast them. The professor's team is focused
on this relationship part.

---

## 2. Environment and tooling

**Workstation**
- Baylor-managed Windows workstation `DRJG-Resh01.baylor.edu`, reached via Windows Remote Desktop
  (Display → Full Screen; "Smart sizing" for windowed mode).
- **No admin rights**: installers asking for admin fail. Everything is installed per user.
- GPU: NVIDIA GeForce RTX 5090 (32 GB), driver 610.62, CUDA 13.3. Needs PyTorch CUDA 12.8+ wheels.

**Installed (per user)**
- Git: **Portable Git** (`C:\Users\sachit_satyal\Tools\PortableGit`), `cmd` folder added to the user PATH via
  "Edit environment variables for your account".
- VS Code: **.zip** version in `Tools\VSCode` with an empty `data` folder (portable mode); extensions Python + Jupyter.
- uv 0.12.21 (PowerShell install script, per user).
- Python 3.12.14 via `uv python install 3.12` (a system Python 3.12.3 in `C:\Python312` exists; not used).
- PyTorch 2.11.0+cu128, NumPy 2.5.3, pandas 3.0.6, scikit-learn, SciPy, PyYAML, tqdm, matplotlib, openpyxl;
  dev: pytest, ipykernel.
- Windows Firewall prompt when starting a Jupyter kernel: click Cancel (local kernel works without it).

**Project**
- Folder `C:\Users\sachit_satyal\alzheimers-trajectory-mapping` (renamed from `tarcc-abc`).
- Private GitHub repo `sachitsatyal/alzheimers-trajectory-mapping`, branch `main`, git identity uses the GitHub
  noreply email.
- Python package `alzheimers_trajectory_mapping`, src layout.
- PyTorch index in `pyproject.toml`:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu128"
  url = "https://download.pytorch.org/whl/cu128"
  explicit = true
  [tool.uv.sources]
  torch = [{ index = "pytorch-cu128" }]
  ```
- `uv.lock` committed; `uv sync` recreates the environment exactly.

**Repository layout**
```
configs/        default.yaml (model + training), feature_map.csv (every variable's category/rules)
data/           raw/ (TARCC files), processed/ (tensors)          -- git-ignored
docs/           preprocessing_decisions.md, model_decisions.md, PROJECT_LOG.md
notebooks/      exploration only                                  -- git-ignored
outputs/        runs/<name>/ (config, log.jsonl, best.pt, evals), figures/, consensus_*.csv  -- git-ignored
scripts/        preprocess, baselines, train, evaluate, diagnose, seeds, abc_map,
                subtypes, consensus, ablations, plot_subtypes
src/alzheimers_trajectory_mapping/   package modules (Section 15)
tests/          pytest suite (76 tests)
```

**.gitignore** keeps patient data out: `data/`, `outputs/`, `*.csv` (except `!configs/*.csv`), `*.xlsx`, `*.sav`,
`*.sas7bdat`, `*.pt`, `*.ckpt`, `.env`, `notebooks/`, `*.ipynb`. `.gitattributes`: `* text=auto`.
Lesson: the `*.csv` rule silently excluded `configs/feature_map.csv` until the `!configs/*.csv` exception was added.

---

## 3. Data

**Files**: `TARCC_Main_Data.xlsx` (14,655 rows × 949 columns, one row per visit), `TARCC_Codebook.xlsx`
(2020-08-13 version, 946 variables), `TARCC_ABC_Feature_List.xlsx` (curated ABC measures with tiers).

**Structure**
- 3,670 patients (`PATID`), visit number `VISIT`; no duplicate (PATID, VISIT) rows.
- 1–13 visits per patient (median 3) → T_max = 13; ~800 patients have one visit (no forecast pairs).
- Visit date `A1_EVENTDATEX`: parses for all rows, strictly increasing within patient; 2007-01-23 to 2018-01-31.
- Gaps between visits: median 1.00 y, min 0.10, max 5.73.
- `AGE` = visit year − birth year → age predictor.
- `A1_EID`, `DATA_ORDER`: row IDs (dropped). `A1_INMDS`: enrolled in NACC Minimum Data Set (administrative, dropped).
- `STUDYID`: 3,011 values; ~580 groups contain 2–4 PATIDs. They are **different people** (birth year matches in 1.4%
  of groups, sex 51%, enrollment overlap 58%), probably related participants (not confirmed; ask TARCC).
  → splits are grouped by STUDYID.
- Cohort size 3,670 matches the published Texas Harris Alzheimer's Research Study count.

**Column matching** (`codebook.py`): data uses UPPERCASE; `PROTEOMICS_` → `PROTEO_`, punctuation → `_`; manual renames:
C1_C1DATX→C1_DATEX, I1_INDTECONTX→I1_INDATECONTX, APOE_GENOTYPE_DIGIT→APOE_GENOTYPE_DIGITS,
PROTEOMOICS_TP→PROTEOMICS_TP, RBM_ALPHA_F→RBM_ALPGA_F, MISC_TARC_PAT→PATID. 943 of 946 codebook variables matched.

**Special codes** (parsed per variable): missing, not applicable, below detection, above detection.
- Proteomics: −999999 below detection, −888888 above, −777777 and 0 missing.
- Quanterix: −333 below, −555 above, −999/−888/−444 missing.
- "88 = None reported" (secondary race) is a real answer, not missing.
- Non-assay visits are filled with codes, not blanks → codes converted before any statistic.

---

## 4. Outcomes (K = 5; `outcomes.py`)

| Outcome | Domain | Range | Worse is | How it is computed | Cleaning rule |
|---|---|---|---|---|---|
| PSMS (self-care) | A function | 6–30 | higher | `F1_PSMSTOTSCR`, sum of 6 items (toileting, feeding, dressing, grooming, walking, bathing) | total 0 = form not done → missing (523 visits); outside 6–30 → missing |
| IADL (Lawton) | A function | 8–31 | higher | `F2_IADLTOTSCR`, sum of 8 items (phone, shopping, cooking, housekeeping, laundry, transport, medication, money) | missing unless all 8 items valid and total = item sum (2,483 missing in total); not prorated |
| NPI-Q count | B behavior | 0–12 | higher | number of the 12 symptoms present | missing if informant code 0 (form not done) or any item "unknown" (524 visits) |
| NPI-Q severity | B behavior | 0–36 | higher | sum of severities (1–3) of present symptoms | as count, plus missing if a "yes" lacks a valid severity (527 missing) |
| MMSE | C cognition | 0–30 | lower | `C1_MMSE` | 0 kept only if CDR global ≥ 2 (117), else missing (15) |

- ~521 **partial visits**: forms F1, F2, B5 not done but stored as zeros (PSMS = 0, IADL = 0, NPI informant 0, all NPI
  items "No"); MMSE present at 520 of them. These forms are also masked on the predictor side.
- Floor effects: PSMS 74.3% at floor, IADL 61.0%, NPI-Q 48.8% at 0, MMSE 0.8% at 0.
- Outcomes are never imputed; unobserved → outcome mask µ = 0 (not trained or scored at that visit).
- In the model, outcomes are shifted to 0…n_k (e.g. IADL 0–23) for the beta-binomial; reported numbers are in original points.

---

## 5. Predictors: feature map, quality audit, transforms, tensors

**Categories** (`configs/feature_map.csv`, one row per variable with category, transform, reason, `extra_missing`,
`range_override`, `rule`):

| Category | Variables | Model inputs |
|---|---|---|
| Static (embeddings) | 5: sex, race, Hispanic, handedness, APOE ε4 count | 5 tables |
| Demographic | 4 (age, education, AMNART, WAT) | 4 |
| Clinical | 73 (medical history, vitals, diagnosis, medication flags, decline onset/duration, 2 combined flags) | 73 |
| Biomarker | 168 + 3 "years since assay" | 171 |
| Genomic | 4 (APOE ε2, mother/father dementia, siblings with dementia) | 4 |
| Social | 6 | 26 (one-hot) |
| Caregiver | 11 | 19 (one-hot) |
| ABC history | 31 (CDR boxes/global, GDS-30, neuropsych scaled scores, independence, decline reports, too-impaired, + 5 outcomes at visit t) | 31 |
| **Total** | | **328** |

Key mapping decisions: APOE ε4 only static (ε2 + family history in Genomic); medications reduced to yes/no flags;
medical history 0/1/2 → "any history"; totals not items (CDR sum dropped: sum of boxes); CCR revisions, E1 exit forms,
P1 sample logistics, IDs, dates, free text dropped; X1 risk-factor form dropped (discontinued; duplicated by A5 history
and B1 vitals).

**Quality audit** (`quality.py`; 363 mapped predictors; flags: EMPTY, LOW_COVERAGE, NEAR_CONSTANT, OUT_OF_RANGE,
COVERAGE_CHANGES_BY_YEAR, ZERO_PATTERN_BY_YEAR). Decisions:
- 21 dropped for no/low coverage (<10% of patients), incl. 7 Proteomics markers never measured.
- Near-constant: kept if ≥30 patients have the rare value; combined D1_ANY_VASC (VASC + POS_VASC) and
  ANY_PARKINSONISM (A5_PD + A5_PDOTHR + D1_PARK); always-zero diagnoses dropped.
- Codebook coding ≠ data: A42 vitamin E stored 0/1 (Now/Ever descriptions swapped); A43 anti-dementia medication
  confirmed 1 = yes via diagnosis; A41_PMEDS dropped (codes contradict medication use); I1_INHowContact uses "2 - phone".
- Binary conversions: `binary` {0,1}; `binary_12` {1→1, 2→0}; `ever_present` {0→0; 1,2,8→1}; anything else → missing.
- Informant form not done (relationship 0) at 636 visits → all informant inputs masked.
- Logical fills: packs/day = 0 for never-smokers (4,919 visits); informant contact = daily when living with the patient (8,855 visits).
- Year rules (tests introduced/stopped; zeros on the inactive side → missing): Clock from 2010; AMNART from 2011 (then
  carried forward: premorbid); Animal fluency before 2010; VR I/II and siblings in 2007; X1 decline reports from 2013.
- Also dropped: I1_INRACE, X1_DeclineAgePhy, X1_PhyEstDur, C1_TAPS (undocumented), X1_BMIgt30, CDR sum.

**Transforms** (`transforms.py`, deterministic, no data statistics): codes, ranges, rules, binary/one-hot, derived
flags, informant/never-smoker rules, year rules. **Biomarkers are placed only at the panel's assay visit (`_tp`) and
carried forward, never backward**; "years since assay" added per panel. Panel coverage: Proteomics 2,193 patients (60%),
Quanterix 1,882 (51%), RBM 594 (16%). APOE ε4 unknown for 285 patients (7.8%).

**Split and tensors** (`sequences.py`, `scripts/preprocess.py`):
- Split by STUDYID, stratified by baseline dementia, seed 42: **train 2,567 / val 552 / test 551 patients**;
  visits 10,223 / 2,196 / 2,236; forecast pairs 7,656 / 1,644 / 1,685.
- Balance: demented at baseline 36/35/37%; women 62.3/59.4/61.3% (A1_SEX coded 1 = male, 0 = female; sex not
  stratified because balance verified).
- Standardization and detection-limit fills (below → training min, above → training max) fit on training visits only;
  missing → 0 with mask 0.
- Tensors per split (`data/processed/*.npz`): X, M (N × 13 × 328), tau (t_abs, Δt), dt_next, Y (shifted to 0..n_k),
  MU, PI (padding), S (static indices, 0 = missing), PATID, STUDYID; `meta.json` (columns, category blocks, outcome
  ranges, standardizer).

**Preprocessing in five steps (as presented):** 1 codebook matching and special codes → 2 outcome scoring with
not-done forms → 3 per-variable quality audit → 4 cleaning rules (year rules, biomarkers forward only) → 5 sequences
(family-grouped split, train-only scaling, one sequence per patient). Biggest catch: ~520 visits stored skipped forms
as zeros, which would have read as "perfectly healthy".

---

## 6. Model: final architecture and how it evolved

### 6.1 Final base model (configuration "J3")
Updated architecture figure: `architecture_v2` (PNG/JPG/PDF/SVG, 6 panels a–f; replaces the paper draft's figure).

1. **Visit encoder** (`encoder.py`, panel b): one affine encoder per category over [values; masks] → token,
   **LayerNorm per token**, fusion attention with a global query, + Time2Vec(t_abs, Δt), + summed static embeddings.
2. **History encoder** (`history.py`, panel c): 3-layer pre-norm causal transformer (4 heads, d = 128) + final
   LayerNorm. Causal mask: visit i sees only visits j ≤ i. Output h̃_t per visit.
3. **Posterior** (panel d): diagonal Gaussian q(z_t | history) = N(μ_t, σ_t²), μ and log σ² linear in h̃_t
   (log-variance clamped [−10, 5], bias init −2). Samples z⁽ˢ⁾ = μ + σ ⊙ ε⁽ˢ⁾ (8 in training; 32 by default in
   evaluation, see Section 7).
4. **Latent blocks**: z = [z_S (4), z_A (8), z_B (8), z_C (8)] = 28 numbers. Blocks get their meaning only from the
   wiring: each decoder reads only its own block. S reduced 8 → 4 after testing (S was unused either way; S = 4 pushed
   more information into A/B/C with equal or better CRPS). 8 per domain is a design choice (not yet tuned).
5. **Patient-level subtypes** (`subtypes.py`, panel d): one discrete subtype c per patient (J = 3), fixed over time.
   Membership network π(c | visit 1) assigns new patients; online posterior P(c | visits so far) ∝ π(c) Π p(y | c)
   uses only past transitions. Training marginalizes exactly over c, temperature annealed 5 → 1. Each subtype has a
   learned 16-d embedding e_c. Subtypes **emerge** from training: the three dynamics settings specialize on the
   patients they explain best; names are assigned afterwards from observed progression.
6. **Coupled dynamics** (`dynamics.py`, panel e): dz/dr = Δt · f_θ(z, e_c), r ∈ [0, 1], RK4 with 8 steps over the
   actual gap (longer gap ⇒ more change). Per block: dz_S/dt = f_S(z_S, e_c); dz_D/dt = f_D(z_D, z_S, e_c) +
   Σ W_{D←D'}(c) tanh(z_D'); W starts at 0 with a group-lasso penalty. Velocity penalty discourages jumpy paths.
   SDE variant available (no gain).
7. **Decoders** (`decoders.py`, panel f): each domain decoder reads **only its own block**; attention query from the
   latent block (not from h̃); category attention over the visit's tokens **excluding ABC history**; **anchor** = own
   outcomes' last observed value, presence and age; output = beta-binomial per outcome (p, κ), parameterized as change
   from the anchor: logit p = logit((y_last + 0.5)/(n + 1)) + MLP.
8. **Loss** (`loss.py`): per patient, log p(Y) = T·logsumexp_c[(log π(c | visit 1) + Σ_t ℓ_t(c))/T]
   + β·KL (closed form vs N(0, I)) + λ_v·velocity penalty + λ_W·group lasso on W; normalized by forecast positions.
   Missing outcomes are skipped (never imputed).
9. **Training** (`train.py`): AdamW; every next visit of every training patient is a prediction task (all positions at
   once, enforced causal by the mask); β rises to **kl_max = 0.01** over 20 epochs; early stopping on validation after
   annealing, patience 10. ≈ 567k parameters; ~9 s/epoch alone; a full run ≈ 15 minutes.

### 6.2 Departures from the paper draft and why
| Paper draft | Now | Reason |
|---|---|---|
| Per-visit mixture-of-Gaussians prior on z (soft subtypes per visit) | Patient-level discrete subtype that conditions the dynamics | Draft subtypes recovered disease stage and did not affect forecasts; Aim 1 needs categories for new patients |
| Single latent state | Blocks S/A/B/C with cross-domain coupling W | Needed for an ABC map |
| Decoder query from history h̃ | Query from the latent block | h̃-based query was a side channel that bypassed z (posterior collapse) |
| Decoders read all of z | Each decoder reads only its own block | Otherwise behavior/cognition blocks went unused |
| Decoders see all categories | ABC history excluded from decoder attention | Forces past scores through z (candidate to revert: Section 11) |
| Forecast the level | Forecast change from the last observed value (anchor) | z alone could not carry exact scores; models lost to LOCF |
| KL weight β → 1 | β ≤ 0.01 (kl_max) | β = 1 and 0.1 collapse the latent state |
| Monte Carlo KL to mixture prior | Closed-form KL to N(0, I) | Lower-variance gradients |
| Per-visit objective | Patient-level likelihood summed exactly over subtypes + coupling lasso | Follows from patient-level subtypes |
| J selected 2–6 | J = 3 chosen by seed sweep | Section 9 |

The draft's architecture figure panels a–c still match; panels d, e, f and the objective were redrawn (`architecture_v2`).

### 6.3 Chronology of experiments (what failed and why)
1. **smoke** (3 epochs): pipeline works on GPU.
2. **base_v1** (paper-like): active latent units 32 → 0 by epoch 4 → posterior collapse.
3. Diagnosis: decoder attention query built from h̃ let the decoder read history directly. Redesign: coupled blocks,
   z-based queries, log-variance init −2.
4. **coupled_v1**: collapsed again. **coupled_fb** (free bits 0.1): units "active" but worse forecasts.
5. **LOCF baselines**: models appeared to lose on function and behavior.
6. **coupled_det** (no KL): still appeared to lose → **anchor** added.
7. **coupled_anchor**: collapsed; still "lost" on the training-log MAE.
8. **Key realization**: training-log MAE used the predictive **mean**; on floor-heavy scores the correct point forecast
   is the **median**. Built `evaluate.py` (median MAE, RMSE, NLL, CRPS, coverage, baselines on identical positions).
   With fair evaluation the models **beat both baselines on every outcome**.
9. **kl_max sweep**: 0.1 collapsed; **0.01 kept 25–28 of 32 units active and was best** → adopted.
10. **Latent usage** (shuffle blocks across patients): z used; A used by function; **B and C unused**.
11. **Own-block decoders**: all domain blocks used; **S = 4 best (ownS4)**; S itself then unused.
12. Learned coupling W tiny (0.006–0.06) → ABC map measured by history influence instead (Section 8).
13. J sweep → **J = 3** (Section 9). 14. Ablations (Section 11).

---

## 7. Forecasting results (Aim 2), validation split

**Base J3, 3 seeds (mean):**

| | PSMS (6–30) | IADL (8–31) | NPI-Q sev (0–36) | NPI-Q count (0–12) | MMSE (0–30) |
|---|---|---|---|---|---|
| CRPS, our model | 0.498 | 0.851 | 1.245 | 0.771 | 1.145 |
| CRPS, probabilistic LOCF | 0.639 | 1.286 | 1.408 | 0.850 | 1.414 |
| Error reduction (CRPS) | 22% | 34% | 12% | 9% | 19% |
| MAE (median), our model | 0.681 | 1.175 | 1.694 | 1.062 | 1.602 |
| MAE, plain LOCF | 0.729 | 1.624 | 1.757 | 1.087 | 1.893 |
| 90% interval coverage | 0.961 | 0.939 | 0.931 | 0.934 | 0.942 |

Seed-to-seed SD of CRPS ≈ 0.005–0.024. Largest gains: IADL, PSMS, MMSE; NPI-Q modest (behavior fluctuates).
90% intervals slightly conservative (93–96%). Context: typical MMSE decline 1–3 points/yr and test–retest variation
1–2 points, so ~1 point CRPS is near the test's own noise.

**How the numbers are computed (for the Methods section):**
- Every validation patient (552) and every next visit with an observed score (no sampling of patients or visits);
  each outcome has its own count because missing scores are skipped.
- Model forecast at each position: posterior samples of z (**32 by default** in `scripts/evaluate.py`;
  `samples_test: 100` in the config is not used by this script), each run through ODE + decoder, beta-binomials
  averaged over samples, then mixed over subtypes with the **online** posterior (only visits before the forecast).
  No future information enters a forecast.
- Baseline at the same positions: **probabilistic LOCF** = last observed score + empirical distribution of
  visit-to-visit changes from training pairs (clipped to range). **Plain LOCF** = last score (used for MAE).
- CRPS per forecast (for a discrete score: Σ_k (F(k) − 1[y ≤ k])²; equals |error| for a point forecast), averaged over
  all positions (so patients with more visits contribute more), then averaged over 3 trained models.
  Error reduction = (baseline − model)/baseline.
- Coverage: share of real scores inside the model's central 90% interval.

---

## 8. ABC map (Aim 2): Granger-style history influence

**Method** (`diagnostics.history_influence`, `scripts/abc_map.py`), technically **grouped permutation importance** on
the trained model (no retraining):
1. Score all validation forecasts (CRPS) with intact inputs.
2. For one domain, replace each patient's history of that domain's input columns (values and masks) with another
   patient's: **the whole visit sequence at once, with one partner per patient** (visit 1 from partner's visit 1,
   visit 2 from partner's visit 2, …). Pairing is a random permutation **within each batch of 32 patients**, one
   permutation (seed 0). If the partner has fewer visits, the extra visits become "missing".
   All other inputs (other domains, age, timing, static) stay the patient's own.
3. Re-score; % CRPS increase per outcome = row of the map. Repeat on 3 trained models.

Domain assignment of history inputs (`DOMAIN_INPUTS`): A = past PSMS/IADL, independence, CDR community affairs /
home & hobbies / personal care; B = past NPI-Q, GDS-30; C = past MMSE, CDR memory/orientation/judgment, neuropsych
tests, clock, CERAD, animal fluency, too-impaired, decline reports. Unassigned: CDR global, literacy problems.

**Result (ownS4 J = 4 models, 3 seeds, mean ± sd % CRPS increase):**

| history ↓ / forecast → | PSMS | IADL | NPI-Q sev | NPI-Q count | MMSE |
|---|---|---|---|---|---|
| A function | 3.0±2.2 | 8.1±3.5 | 0.9±0.2 | 1.1±0.2 | 1.0±0.4 |
| B behavior | ≈0 | ≈0 | 25.2±3.9 | 16.0±1.8 | 0.5±0.5 |
| C cognition | **8.8±0.8** | **18.2±3.6** | 2.5±0.8 | 3.5±1.1 | 16.8±2.3 |

**Robustness:**
- Without CDR boxes in the cognition group: C → A unchanged (8.1±0.4, 18.0±3.3); C → B disappears (0.7/0.8%) → it
  came from CDR ratings; not reported.
- MMSE only: C → A 2.5±1.1 / 4.0±0.8 → about a quarter; the detailed neuropsych tests carry most of the effect.

**Robust finding:** cognition history → future function is strong (larger than function's own history beyond the
anchor), mostly via detailed neuropsychological tests beyond MMSE; consistent with cognitive decline preceding
functional loss. Function → others ~1%; behavior self-contained. Within-gap coupling W ≈ 0.

**Phase 1 step 1 — per-subtype maps with random and diagnosis-matched shuffles (28 Oct 2026)**
Code: `abc_groups.py`, `scripts/abc_map_groups.py` (shuffle over the whole val split; "matched" = partner has the same
baseline diagnosis; 10 repeated shuffles per model; J3_s1–s3; consensus subtypes from `outputs/consensus_val.csv`;
results in `outputs/abc_maps/`). Mean ± sd across 3 models of % CRPS increase; * = mean > 2 sd.
Val patients: stable 233, steady decliner 207, behavior-escalating 112; normal 231, dementia 194, MCI/other 127.
Forecasts: PSMS 1,642, IADL 1,429, NPI-Q sev 1,641, count 1,642, MMSE 1,644.

Matched shuffle, all patients (primary result):

| history ↓ / forecast → | PSMS | IADL | NPI-Q sev | NPI-Q count | MMSE |
|---|---|---|---|---|---|
| A function | 2.4±1.0* | 5.8±0.3* | 1.5±0.8 | 1.8±1.0 | 1.3±0.4* |
| B behavior | 0.4±0.5 | −0.3±0.3 | 21.2±3.6* | 13.1±1.5* | 0.7±1.0 |
| C cognition | 5.6±2.7* | 8.5±0.9* | 3.3±1.4* | 4.1±1.6* | 16.2±2.0* |

Random shuffle, all patients: C → A 6.6±2.9* / 12.0±2.5*; C → B 4.4±1.2* / 5.7±1.6*; A → A 2.2* / 6.9*; B → B 24.9* / 14.7*.

Matched shuffle per subtype (key cells; full tables in `outputs/abc_maps/summary_val_main.csv`):

| Link | Stable | Steady decliner | Behavior-escalating |
|---|---|---|---|
| C → function (PSMS / IADL) | 5.9* / 6.7* | 6.3 / 9.8* | 4.6* / 8.7* |
| C → behavior (sev / count) | **12.5* / 14.7*** (sd 6–7) | 0.3 / 0.9* | 2.1* / 1.4* |
| A → behavior (sev / count) | **7.2* / 7.8*** | 0.6 / 0.6 | 0.0 / −0.3 |
| A → MMSE | 1.0* | 1.6* | 1.3 |
| B → function / cognition | ≈0 | ≈0 | ≈0 (PSMS 0.7*) |
| B → B (sev / count) | 18.9* / 11.2* | 29.0* / 22.7* | 15.3* / 4.9 |

Findings:
1. **Cognition → function is robust**: present in every subtype and after diagnosis matching (IADL 8.5%, larger than
   function's own history 5.8%). Matching reduced it from 12.0% to 8.5%: about 30% of the earlier effect was shared
   disease stage; the rest is within-stage information.
2. **New, subtype-specific links**: in **stable** patients, past function and cognition predict future behavior
   (7–15%). Hidden in the overall average. Caution: stable patients have very low NPI-Q (mean 0.8), so a small absolute
   change is a large %; seed sd is large; CDR boxes are included in C (CDR previously explained C → B). To check:
   `--variant no_cdr` and absolute CRPS changes.
3. Behavior does not predict function or cognition in any subtype. In behavior-escalating patients, behavior's own
   history helps less (count 4.9%, n.s.): their symptoms change in ways their past does not predict.
4. Function → cognition small but consistent (~1–1.6%).
5. Weak C → B overall (3–4%, significant in this run with CDR included).

**Phase 1 step 2 — same analysis without CDR boxes (`--variant no_cdr`, 28 Oct 2026)**
Matched shuffle, % CRPS increase (with CDR → without CDR):

| Link | With CDR | Without CDR | Verdict |
|---|---|---|---|
| C → function, all (PSMS / IADL) | 5.6* / 8.5* | **5.5* / 8.4*** | robust |
| C → IADL per subtype (stable / steady / behavior-esc.) | 6.7* / 9.8* / 8.7* | **6.9* / 9.7* / 8.1*** | robust in all |
| C → behavior, stable (sev / count) | 12.5* / 14.7* (sd 6–7) | **9.5* / 10.6*** (sd ~3) | survives |
| A → behavior, stable (sev / count) | 7.2* / 7.8* | **3.7* / 4.2*** | survives, halved |
| C → behavior, steady / behavior-esc. | 0.3 / 2.1* | 0.2 / 0.5 | none |
| C → behavior, all | 3.3* / 4.1* | 2.0* / 2.3* | small; driven by stable patients |
| A → MMSE, all | 1.3* | 0.8 (n.s.) | came from CDR boxes; not claimed |
| B → function / cognition | ≈0 | ≈0 | none |

Current ABC picture: **cognition → function** strong and robust everywhere; **cognition and function → behavior only
in the stable (mostly cognitively normal) subtype**; behavior → others none; function → cognition none after removing CDR.
Pending: absolute-change check for the stable-group behavior links (low NPI-Q, mean 0.8, makes % changes large);
`--absolute` option added to `scripts/abc_map_groups.py` (also outputs intact CRPS and change in CRPS points).

**Interpretation and limits (for the paper):**
- This is **directed predictive influence (Granger-style), not causation**: hidden common causes (neuropathology),
  test sensitivity/timing, unmeasured factors (medications, caregivers) and model dependence remain possible.
- Validity: same model, only the patient–history link broken; realistic replacement values; past only; diagonals
  behave as expected; consistent across seeds and robust to removing CDR.
- Limits: correlated domains (information leaks through other inputs → underestimates); unrealistic combinations
  (→ may overestimate); **the anchor is not scrambled** (own last score still given → diagonal, especially
  function → function, understated); single permutation; random pairing across diagnoses ("sicker overall" effect).

**Why most links look weak** (analysis 23 Oct): (1) one-visit horizon (~1 year) leaves little change to explain
beyond the own last score; (2) domains overlap (sicker patients worse everywhere); (3) the shared encoder lets
cross-domain information enter each block directly, so W is redundant and the lasso drives it to 0; (4) NPI-Q is noisy;
(5) some asymmetry may be real (cognition → function direction). Plan to address: Section 13.

---

## 9. Progression subtypes (Aim 1)

**Choice of J** (3 seeds each): CRPS equivalent for J = 2, 3, 4. J = 3 recovers the same three subtypes in every
seed; J = 2 merges the behavior-escalating group into general decliners; J = 4 splits steady decliners inconsistently.
ARI between seeds: J2 0.38–0.63, J3 0.41–0.45, J4 0.33–0.55 → soft phenotypes on a continuum.

**Consensus** (`consensus.py`): labels matched across seeds (Hungarian algorithm on shared patients), probabilities
averaged. Seeds agree with consensus for 82–88% of patients (ARI 0.56–0.67). Names from observed progression.

| Subtype | Share train / val | NPI-Q sev /yr | IADL /yr | PSMS /yr | MMSE /yr | Baseline profile |
|---|---|---|---|---|---|---|
| Stable | 41% / 42% | −0.2 | +0.4 / +0.5 | +0.2 / +0.3 | −0.4 | youngest (~68), 64% normal, NPI-Q 0.8, MMSE 27.4, APOE ε4 0.32 |
| Steady decliner | 40% / 37% | ≈0 | +1.1 / +0.8 | +0.7 / +0.5 | −1.2 / −1.1 | ~72 y, 50% dementia, NPI-Q 3.2, MMSE 24.7, ε4 0.43 |
| Behavior-escalating | 19% / 20% | **+1.3 / +1.0** | **+2.0 / +1.8** | **+1.4 / +1.2** | **−2.0 / −1.6** | ~72 y, 56% dementia, **NPI-Q 6.5**, MMSE 23.9, ε4 0.44 |

(For PSMS, IADL, NPI-Q higher = worse.) Train counts: 1,054 / 1,025 / 488; val 233 / 207 / 112.
- Within diagnosis (train): MCI behavior-escalating NPI-Q +1.5/yr vs −0.4 in others, MMSE −0.9; dementia
  behavior-escalating MMSE −3.1/yr vs −1.8/−2.2, PSMS +2.4. In normal patients the behavioral rise (+0.8 train) did
  **not** replicate in val (+0.1) → claimed for MCI and dementia only.
- Behavior-escalating patients start with the highest symptom burden yet worsen (opposite of regression to the mean).

**Early identification of new patients** (val, ≥5 visits, n = 212; chance ≈ 33%):

| Visits seen | Overall | Stable recall | Steady recall | Behavior-escalating recall |
|---|---|---|---|---|
| 1 | 49% | 69% | 40% | 23% |
| 2 | 56% | 78% | 35% | 50% |
| 3 | 60% | 79% | 37% | 61% |
| 4 | 68% | 83% | 45% | 80% |

Caveats: per-year contrasts partly circular (subtypes defined from trajectories); evidence = val replication, seed
stability, within-diagnosis consistency, early predictability. Precision (PPV) still to add.

**Visualization:** `scripts/plot_subtypes.py` (written 9 Oct, tested on synthetic data; real figures not yet produced):
Figure 1 = mean observed MMSE, IADL, NPI-Q severity by years since enrollment per subtype with 95% CI (all patients
and MCI-at-entry); Figure 2 = same for J = 2, 3, 4, colored by overlap with J = 3 subtypes (dashed = second subtype
mapping to the same color). Saves aggregated CSVs only. Caption caveat: observed means among patients still followed
(dropout bias). Validated color-blind-safe palette: stable #1baf7a, steady #2a78d6, behavior-escalating #eb6834.

---

## 10. Evaluation methodology (for the paper)
- Point forecast for MAE = predictive **median**; RMSE uses the mean; probabilistic: NLL, **CRPS** (main metric),
  90% interval coverage and width.
- Predictive distribution = mixture of beta-binomials over posterior samples and subtypes, weighted by the **online**
  subtype posterior (only past visits).
- Baselines on identical positions: plain LOCF and probabilistic LOCF.
- Training: every next visit of every training patient is a task, from past visits only (causal mask). Split is by
  patient (family group), not by time.
- "Active units" (KL > 0.01 per dimension) does not prove use; use the **shuffle tests** (`diagnose.py`).
- Comparisons across configurations use 3 seeds; differences > ~2 SD of seed noise treated as real.

---

## 11. Ablation study (complete)
`scripts/ablations.py`, 16 variants × 3 seeds vs base J3. % change in CRPS vs full model (+ = worse without the part,
− = better), * = beyond 2 SD of run-to-run spread. Full-model CRPS 0.499 / 0.850 / 1.244 / 0.771 / 1.144.

| Variant | PSMS | IADL | NPI sev | NPI cnt | MMSE |
|---|---|---|---|---|---|
| **Inputs removed** | | | | | |
| Past ABC scores | +2.4 | +8.2* | +1.0 | +1.0 | +9.8* |
| Missing-data masks | +0.5 | +4.1* | +0.8 | +1.4 | +3.1* |
| Clinical | −1.2 | +0.7 | −1.0 | −1.5 | −1.7* |
| Biomarkers | −4.4 | −2.1 | −2.0 | −1.8 | −0.2 |
| Demographic | +1.7 | −0.3 | −0.1 | +0.2 | +0.5 |
| Genomic | −1.0 | −0.5 | +0.3 | +0.3 | −0.6 |
| Social | −1.5 | −0.8 | +1.0 | +0.7 | −0.2 |
| Caregiver | −1.2 | −1.6 | −0.3 | −0.2 | −0.9 |
| **Architecture changed** | | | | | |
| Decoders read past scores (paper_context) | −3.5 | −4.2* | −0.3 | −0.1 | −1.8* |
| No change-from-last (anchor) | +2.3 | +1.3 | −2.0 | −1.8 | +1.7 |
| No ODE | −2.2 | −2.7 | +1.3 | +1.2 | −0.9 |
| Stochastic ODE (SDE) | −1.5 | −1.3 | +0.8 | +1.1 | +0.4 |
| One subtype | −1.4 | −1.4 | −1.5 | −0.6 | −0.6 |
| Deterministic state | +0.2 | +0.4 | −0.6 | 0.0 | +1.3 |
| No coupling W | −0.1 | +0.2 | −0.7 | −0.6 | +0.6 |
| No Time2Vec | 0.0 | −1.1 | +0.4 | +0.5 | +2.2 |

Conclusions:
- One-step accuracy comes from **past ABC scores** and **missing-data masks**.
- ODE, subtypes, stochastic state, coupling, time encoding, and demographic/genomic/social/caregiver/biomarker
  categories: no measurable effect on next-visit accuracy → justified by Aims 1 and 3 (subtypes, simulation), not
  one-step accuracy. Kept for possible value at longer horizons and for Aim 3 (e.g. caregiver as a modifiable factor).
- **paper_context** (decoders also attend to ABC history) improves IADL/MMSE by 2–4%. Not adopted yet: first check with
  `diagnose.py` that latent blocks are still used (risk of a shortcut that bypasses z).
- Biomarkers slightly better removed (n.s.); RBM covers only 16% of patients.

---

## 12. Presentations and deliverables

| Date | Deliverable | Content |
|---|---|---|
| 8–9 Oct 2026 | Slides "ABC Trajectory Mapping — Progress Update" (14 slides) and Docs report "ABC Trajectory Mapping: Progress Report on Aims 1 and 2" (both claude.ai artifacts) | High-level results for the 9 Oct professor meeting |
| 19 Oct 2026 | Slides "ABC Trajectory Mapping — Methods and Results" (17 slides) | Overview, 5-step preprocessing, cohort and outcome ranges, architecture diagram, changes from the draft, forecasting, ABC map, ablation summary, full ablation table, subtypes, subtype per-year chart, within-diagnosis, early identification, caveats, next steps, questions |
| 20 Oct 2026 | `architecture_v2` figure (PNG, JPG 2800×1336, PDF, SVG) | Paper-style 6-panel architecture updated to the current model |
| 9 Oct 2026 | `scripts/plot_subtypes.py` | Subtype trajectory figures (to run on the workstation) |

Questions raised for the professor: which inputs are modifiable for Aim 3; one paper or two (methods paper, e.g.
CHIL 2027, plus clinical subtype paper, e.g. Alzheimer's & Dementia: DADM); request NACC access for external validation;
meaning of STUDYID.

**NACC** (external validation candidate): National Alzheimer's Coordinating Center (Univ. of Washington, NIA-funded),
Uniform Data Set from ~30+ ADRCs, >45,000 participants (verify current numbers). TARCC forms follow the UDS. Mapping
issues: MMSE replaced by MoCA in UDS v3 (2015) → conversion table; function measured by FAQ instead of PSMS/IADL;
NPI-Q and CDR identical. Free data request with a data use agreement.

---

## 13. Current plan: making ABC relationships clear (decided 28 Oct 2026)

Goal: the professor's team focuses on how function, behavior and cognition influence each other. Principle: design the
model so links **can** appear and are measured fairly; a link that stays near zero after these changes is a finding,
not a failure. Choices (e.g. lasso strength) are made on validation accuracy, not on link size.

| Phase | Work | Est. time |
|---|---|---|
| 1. Quick checks (current model, no retraining) | ABC map per subtype; diagnosis-matched scramble; 10 repeated shuffles for error bars; classic cross-lagged panel model as independent statistical check | 2 days |
| 2. Domain-separated encoders | Block A built only from function inputs, B from behavior, C from cognition, S from context (demographic, clinical, biomarker, genomic, social, caregiver), so cross-domain information must pass through W → **W becomes the ABC map**; weaker/tuned lasso; W per subtype; optional nonlinear coupling read via ∂f_A/∂z_C | 7 days |
| 3. Multi-year forecasts | Forecast 1, 2, 3 visits ahead by rolling the ODE without re-encoding; scramble test at 2–3 years (also needed for Aim 3) | 4 days |
| 4. Validation | Synthetic data with known links (does the model recover them?); 5 seeds with confidence intervals; agreement of 3 methods (W, scramble at multi-year, cross-lagged model); optional NPI-Q symptom clusters (mood / agitation / psychosis) | 5 days |

Final deliverable: ABC network figure (A, B, C nodes, arrows sized by strength, one panel per subtype, with CIs).
Other pending improvements: block-size sensitivity (4/4/4/4, 4/8/8/8, 4/16/16/16, 0/8/8/8); scramble only the most
recent vs older visits (how far back influence reaches).

---

## 14. Open items
1. **ABC relationship plan** (Section 13) — Phase 1 steps 1–2 done (D17, D18); next: absolute-change check, then cross-lagged panel model.
2. paper_context check (latent usage) → adopt or not.
3. Run `plot_subtypes.py` on the workstation and add real figures to the deck.
4. Stronger baselines: GRU and linear model.
5. Precision (PPV) for early subtype identification; baseline predictors of membership (odds ratios).
6. Aim 3 what-if simulation: decide modifiable inputs; roll forward in ~1-year steps; frame as model-based scenarios.
7. Final evaluation on the **test split, once**, after freezing all choices.
8. Paper rewrite (Method, evaluation, related work: latent class trajectory models / lcmm, SuStaIn; limitations).
9. Smaller: IADL "not applicable" masking by sex; STUDYID meaning; RBM panel value; slightly conservative calibration;
   silence pandas warning in ablation summary and the J = 1 std warning; consider `--samples 100` in evaluate.py for
   final numbers.

---

## 15. Code modules and commands

| Module | Role |
|---|---|
| `codebook.py` | Load codebook, match names, classify special codes |
| `outcomes.py` | Build the five outcomes, flag not-done forms |
| `quality.py` | Per-variable quality audit |
| `transforms.py` | Apply every feature-map decision → clean inputs per visit |
| `sequences.py` | Split, standardize, time inputs, pad, save tensors |
| `data.py` | PyTorch dataset/loader |
| `encoder.py` | Category tokens, fusion, Time2Vec, static embeddings |
| `history.py` | Causal transformer, Gaussian posterior |
| `subtypes.py` | Patient-level subtypes, membership, online posterior |
| `dynamics.py` | Coupled ABC latent dynamics (RK4/SDE/none), interaction map |
| `decoders.py` | Domain decoders, anchor, beta-binomial |
| `model.py` | Full model (with `perturb` hook) |
| `loss.py` | Patient-level objective and diagnostics |
| `train.py` | Training loop, schedules, checkpoints |
| `evaluate.py` | Distribution-aware evaluation and baselines |
| `diagnostics.py` | Latent-block usage, history influence (ABC map), `shuffle_inputs` |
| `subtype_analysis.py`, `consensus.py` | Aim 1 analyses |

```powershell
uv run pytest -q                                             # 76 tests
uv run python scripts\preprocess.py                          # raw -> data/processed
uv run python scripts\baselines.py val                       # LOCF / train-mean baselines
uv run python scripts\train.py --name RUN [--set key=value ...]
uv run python scripts\evaluate.py --name RUN [--samples 32]  # CRPS, MAE, coverage vs baselines
uv run python scripts\diagnose.py --name RUN                 # which latent blocks are used
uv run python scripts\seeds.py --name NAME [--set ...]       # 3 seeds, mean/sd
uv run python scripts\abc_map.py --names R1 R2 R3 [--variant main|no_cdr|scores_only]
uv run python scripts\subtypes.py --names R1 R2 R3
uv run python scripts\consensus.py --names J3_s1 J3_s2 J3_s3 --split train|val
uv run python scripts\ablations.py --part k --of 3 | --summarize
uv run python scripts\plot_subtypes.py [--split train|val]   # subtype figures -> outputs/figures/
```
PowerShell: quote overrides with spaces, e.g. `"model.latent_blocks={S: 4, A: 8, B: 8, C: 8}"`.

**Final configuration (`configs/default.yaml`)**
Model: d 128; encoder 3 layers, 4 heads, ff 256, dropout 0.2; Time2Vec k 16; latent_blocks {S: 4, A: 8, B: 8, C: 8};
velocity hidden 128; RK4 8 steps; decoder hidden 64, dropout 0.2; samples_train 8 (samples_test 100 in config, but
evaluate.py default is 32); n_subtypes 3; subtype embedding 16; membership hidden 64; kappa_min 0.01; token_norm true;
category_dropout 0.0; use_masks true; use_time2vec true; drop_categories []; deterministic false; dynamics ode;
decoder_exclude_categories [ABC_HISTORY]; decoder_uses_history false; coupling true; posterior_logvar_init −2.0;
decoder_anchor true; decoder_reads_shared false.
Training: seed 42; batch 32; AdamW lr 3e-4, weight decay 1e-4; grad clip 1.0; max 100 epochs, patience 10; velocity
penalty 1e-3; KL anneal 20 epochs to kl_max 0.01; subtype temperature 5 → 1 over 20 epochs; free_bits 0.0;
coupling_penalty 1e-3.

---

## 16. Decision Log

Chronological record of major decisions. Dates are approximate where marked (~).

**Template for new entries**
```
### D<n> — <short title> (<date>)
- Context: what problem or question led to this
- Options considered: ...
- Decision: what we chose
- Evidence: numbers / test that supported it (runs, seeds, split)
- Consequences: what changed in code, config, results, paper
```

### D1 — Per-user tooling on the workstation (~mid Sep 2026)
- Context: no admin rights; installers failed.
- Decision: Portable Git, VS Code zip in portable mode, uv + Python 3.12, PyTorch cu128 index; GitHub private repo.
- Consequences: environment reproducible with `uv sync`.

### D2 — Data hygiene rules (~late Sep 2026)
- Context: codebook codes/names did not match data; zeros used as filler for not-done forms.
- Decision: decode special codes per variable; not-done forms → missing (outcomes and predictors); year rules; biomarkers
  placed at assay visit and carried forward only; outcomes never imputed.
- Evidence: ~520 partial visits with all-zero forms; year-dependent zero patterns.
- Consequences: Sections 3–5; `docs/preprocessing_decisions.md`.

### D3 — Split by STUDYID, stratified by baseline dementia, 70/15/15 (~late Sep 2026)
- Context: STUDYID groups link different (probably related) people.
- Decision: grouped split, seed 42; scaling fit on training only; sex not stratified (balance verified).

### D4 — Paper draft treated as outline; patient-level subtypes (~early Oct 2026)
- Context: draft subtypes were per-visit clusters of z (likely disease stage); Aim 1 needs categories for new patients.
- Decision: one discrete subtype per patient conditioning the dynamics; membership from visit 1, Bayes updates;
  exact marginalization; N(0, I) prior.

### D5 — Coupled ABC latent blocks (~early Oct 2026)
- Context: project goal is the ABC relationship map.
- Decision: z = S/A/B/C blocks with subtype-specific coupling W and group lasso.

### D6 — Fixes for posterior collapse (~early Oct 2026)
- Context: latent state unused (active units → 0) in base_v1, coupled_v1, coupled_anchor.
- Decision: decoder queries from latent blocks (remove h̃ side channel); ABC history excluded from decoder attention;
  log-variance init −2; anchor (forecast change from last observed); kl_max = 0.01.
- Evidence: kl_max 0.01 kept 25–28 active units and best CRPS; kl 0.1 collapsed.

### D7 — Fair evaluation protocol (~early Oct 2026)
- Context: training-log MAE (predictive mean) made models look worse than LOCF.
- Decision: `evaluate.py` with median MAE, CRPS as main metric, coverage, LOCF and probabilistic LOCF on identical
  positions; online subtype posterior for forecasts.
- Evidence: under fair evaluation models beat both baselines on all outcomes.

### D8 — Own-block decoders with S = 4 (ownS4) (~early Oct 2026)
- Context: B and C blocks unused when decoders read the shared block.
- Decision: decoder_reads_shared false; latent_blocks S4/A8/B8/C8.
- Evidence: 3 seeds, CRPS equal or better than base on all outcomes; all domain blocks used.

### D9 — ABC map via history influence (scramble test) (~6 Oct 2026)
- Context: learned W ≈ 0; shared encoder lets cross-domain info bypass W.
- Decision: measure the map by shuffling each domain's past inputs across patients (Granger-style).
- Evidence: cognition → function 8.8% / 18.2% (3 seeds), robust without CDR; Section 8.

### D10 — J = 3 subtypes (~6–7 Oct 2026)
- Evidence: CRPS equal for J = 2/3/4; J = 3 recovers the same three profiles in every seed; J = 2 merges
  behavior-escalating; J = 4 unstable split. Consensus across seeds adopted (Section 9).

### D11 — Ablation conclusions (8 Oct 2026)
- Decision: keep full model; ODE/subtypes justified by Aims 1/3; paper_context is a candidate pending latent-usage check;
  keep all data categories for now (no significant harm; possible value for long horizons and Aim 3).
- Evidence: Section 11.

### D12 — Subtype visualization approach (9 Oct 2026)
- Decision: observed mean trajectories by years since enrollment with 95% CI, all patients + MCI-at-entry; J = 2/3/4
  comparison colored by overlap with J = 3 (`scripts/plot_subtypes.py`).

### D13 — Presentation framing (9 and 19 Oct 2026)
- Decision: high-level results for the professor; ABC map stated as predictive influence, not causation; baseline
  labeled probabilistic LOCF; caveats slide (one cohort, partly circular subtype contrasts, prediction ≠ cause,
  baselines limited, validation numbers only).

### D14 — Architecture figure updated (20 Oct 2026)
- Decision: redraw the draft's 6-panel figure to match the current model (`architecture_v2`); panels a–c unchanged.

### D15 — Correction: number of evaluation samples (28 Oct 2026)
- Context: explanations stated forecasts average 100 posterior samples.
- Finding: `scripts/evaluate.py` defaults to `--samples 32`; config `samples_test: 100` is not used by it.
- Decision: report "32 samples" unless runs were made with `--samples 100`; consider 100 for final test-set numbers.

### D16 — Plan to make ABC relationships clear (28 Oct 2026)
- Context: professor's team focuses on function–behavior–cognition relationships; current map shows only
  cognition → function robustly.
- Decision: four-phase plan (Section 13): quick checks → domain-separated encoders (W = ABC map) → multi-year
  forecasts → validation (synthetic recovery, seeds, three-method agreement).
- Decision: document everything in this log, with a Decision Log entry after each major decision.

### D17 — Diagnosis-matched, per-subtype ABC map as the primary scramble analysis (28 Oct 2026)
- Context: Phase 1 step 1; original map shuffled across all patients within batches (mixes disease stage) and averaged over subtypes.
- Decision: report the **matched** shuffle (same baseline diagnosis, whole val split, 10 repeats, 3 J3 models) as the
  primary map; random shuffle as a secondary; maps per consensus subtype.
- Evidence: Section 8 (Phase 1 step 1). Cognition → function survives matching in every subtype; stable-subtype
  function/cognition → behavior links found, pending CDR and absolute-change checks.
- Consequences: new `abc_groups.py`, `scripts/abc_map_groups.py`, tests (80 total). Next: `--variant no_cdr` run.

### D18 — No-CDR matched map is the primary ABC map (28 Oct 2026)
- Context: CDR boxes are clinician ratings that mix domains; earlier they created a spurious C → B link.
- Decision: report the **no_cdr, diagnosis-matched** map as primary (with-CDR and random as sensitivity analyses).
  Drop function → cognition (n.s. without CDR). Stable-subtype C/A → behavior links kept as provisional until the
  absolute-change check.
- Evidence: Section 8, Phase 1 step 2.
- Consequences: `abc_groups.group_change` now also returns `intact_crps` and `abs_change`; script options
  `--schemes`, `--absolute`; result files `outputs/abc_maps/<run>_<split>_<variant>[_scheme]_v2.csv`.

---

## 17. Glossary (plain language)
- **CRPS**: average "points off" of a probabilistic forecast, giving credit for putting probability near the truth;
  equals absolute error for a single-number forecast. Lower is better.
- **LOCF / probabilistic LOCF**: "same as last time" / "last score plus the typical change everyone has".
- **Posterior collapse**: the model learns to ignore its latent state and uses shortcuts; forecasts can still look fine.
- **Latent state / blocks**: 28 hidden numbers summarizing a patient at a visit (S shared, A function, B behavior,
  C cognition), each block given meaning by which decoder reads it.
- **Neural ODE**: a learned "speed and direction" rule that moves the latent state forward over the actual time gap.
- **Anchor**: the last observed score; decoders predict the change from it.
- **Scramble test (grouped permutation importance)**: swap one domain's history between patients and measure how much
  forecasts worsen.
- **Granger-style influence**: X's past helps predict Y's future beyond Y's own past; predictive, not causal.
- **ARI**: agreement between two groupings (1 = identical, 0 = chance).
- **Online posterior**: subtype probabilities using only visits seen so far.

---

## 18. Lessons learned
- Codebooks can be wrong: codes, descriptions and ranges must be checked against the data.
- "0" was used as filler for not done / not asked / not applicable; check zeros by year and by form.
- Biomarkers measured once must be placed at the assay visit and carried forward only.
- Posterior collapse hides behind good-looking forecasts; check that the latent state is actually used.
- Any path that lets the decoder read history directly (even attention queries) can bypass the latent state.
- Fair evaluation matters: the median for MAE, CRPS for distributions, baselines on identical positions.
- Confirm design choices with several seeds before adopting them.
- A shared encoder makes explicit coupling (W) redundant; to read relationships from the model, information paths must
  be separated by design.
- Check stated defaults against the code (evaluation sample count).

---

## 19. Version History

| Version | Date | Change |
|---|---|---|
| 1.0 | 28 Oct 2026 | Full log through 28 Oct: ablations, presentations, architecture v2, evaluation details, ABC plan, decision log D1–D16, glossary |
| 1.1 | 28 Oct 2026 | Added versioning rules and this history; from now on the log is updated after every decision |
| 1.2 | 28 Oct 2026 | Phase 1 step 1 results: per-subtype ABC maps, random vs diagnosis-matched shuffles (Section 8); decision D17 |
| 1.3 | 28 Oct 2026 | Phase 1 step 2: no-CDR maps; stable-subtype behavior links survive; function → cognition dropped; absolute-change option (D18) |
