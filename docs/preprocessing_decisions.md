# Preprocessing decisions

Codebook: TARCC_Codebook.xlsx (20200813 version). Data: TARCC_Main_Data.xlsx.

## Data structure
- One row per visit: 14,655 visits, 3,670 patients, 949 columns. No duplicate (PATID, VISIT) rows.
- Keys: `PATID` (patient), `VISIT` (visit number). Visits per patient: 1–13 (median 3), so T_max = 13.
- ~800 patients have a single visit; they contribute no forecast pairs.

## Time
- Visit date `A1_EVENTDATEX` parses for all rows and is strictly increasing within patient.
- Date range 2007-01-23 to 2018-01-31.
- Gaps between visits: median 1.00 yr, min 0.10, max 5.73. t_abs and Δt are computed from this date.

## Column matching
- Data uses UPPERCASE names; matching to the codebook is case-insensitive.
- Proteomics columns: `PROTEOMICS_` renamed to `PROTEO_`, punctuation replaced by `_`.
- One-off renames: C1_C1DATX→C1_DATEX, I1_INDTECONTX→I1_INDATECONTX,
  APOE_GENOTYPE_DIGIT→APOE_GENOTYPE_DIGITS, PROTEOMOICS_TP→PROTEOMICS_TP,
  RBM_ALPHA_F→RBM_ALPGA_F, MISC_TARC_PAT→PATID.
- In codebook but absent from data: A1_FORMVER, A1_REFER.

## Columns without codebook entries
- `AGE`: age at visit (= visit year − birth year, exact for all rows). Used as the age predictor; birth year not used separately.
- `A1_EID`, `DATA_ORDER`: unique per row (row IDs). Dropped.
- `A1_INMDS`: binary flag (246 = 1), undefined in codebook. Dropped until its meaning is confirmed.
- `STUDYID`: see below.

## STUDYID and data splits
- Each PATID has one STUDYID; ~580 STUDYIDs cover 2–4 PATIDs.
- Patients sharing a STUDYID are different people: birth year matches in 1.4% of groups,
  sex in 51%, and enrollment periods overlap in 58%. Likely related individuals (to confirm with TARCC).
- Decision: patients are not merged. Train/validation/test splits are grouped by STUDYID.

## Open questions
- Meaning of STUDYID.

## Special codes (parsed per variable from the codebook)
- Codebook matched to data: 943 of 946 variables (A1_FORMVER, A1_REFER absent; TARC_PID is an ID).
- Each variable's codes are classified into: missing, not applicable, below detection, above detection.
- "None reported" (e.g. 88 in secondary race) is a real answer, not missing.
- Biomarker detection limits are not missing values:
  Proteomics -999999 = below, -888888 = above, -777777 and 0 = missing;
  Quanterix -333 = below, -555 = above, -999/-888/-444 = missing.
  How to fill below/above-limit values: to be decided.

  ## Outcomes (K = 5)
- ~521 partial visits had forms F1, F2 and B5 not done, stored as 0:
  PSMS=0 and IADL=0 coincide (522/523), NPI informant code 0 (undefined; valid 1–3) at the same visits,
  and all 12 NPI items are "No" at every such visit. MMSE is present at 520 of them.
- PSMS (6–30): total 0 → missing. Total always equals item sum.
- IADL (8–31): total 0 → missing; any item = 0 ("not applicable") → missing (2,483 visits incl. the 522).
  Not prorated: outcomes are never imputed. To check later: N/A rate by sex.
- NPI-Q count (0–12) and severity (0–36): computed from the 12 items.
  Missing when NPI informant = 0. Severity also missing if any "Yes" lacks a severity (6 visits).
- MMSE (0–30): no out-of-range values. MMSE = 0 kept when CDR global ≥ 2 (117 visits);
  set to missing when CDR global < 2 (15 visits, implausible).
- Predictor side: forms F1, F2, B5 at partial visits are marked unobserved in the mask m_t.