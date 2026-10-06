"""Apply every preprocessing decision in configs/feature_map.csv to the raw data.

clean() is deterministic and uses no statistics from the data, so it is safe
to run on all patients before the train/validation/test split. Anything that
needs statistics (detection-limit fill values, standardization) is fit on the
training patients only, in a later step.

Output: one row per visit, model inputs with NaN = unobserved, plus the
column list of each category, the static covariates and the five outcomes.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .codebook import _CODE_PATTERN, normalize_name
from .outcomes import build_outcomes
from .quality import valid_values

CATEGORIES = ["DEMOGRAPHIC", "CLINICAL", "BIOMARKER", "GENOMIC",
              "SOCIAL", "CAREGIVER", "ABC_HISTORY"]
DERIVED_OUTCOMES = ["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]
# Column holding the visit number at which each biomarker panel was assayed.
PANEL_TIMEPOINT = {"RBM": "RBM_RULE_BASED_MEDICINE_TP",
                   "PROTEOMICS": "PROTEOMICS_TP",
                   "Q1": "Q1_QUANTERIX_TP"}

BINARY_MAPS = {
    "binary": {0: 0.0, 1: 1.0},
    "binary_12": {1: 1.0, 2: 0.0},
    "ever_present": {0: 0.0, 1: 1.0, 2: 1.0, 8: 1.0},
}


@dataclass
class CleanData:
    keys: pd.DataFrame                 # PATID, VISIT, STUDYID, DATE, YEAR
    features: pd.DataFrame             # model inputs, NaN = unobserved
    columns_by_category: dict          # category -> list of feature columns
    below_detection: pd.DataFrame      # True where a biomarker was below its assay limit
    above_detection: pd.DataFrame      # True where a biomarker was above its assay limit
    static: pd.DataFrame               # one row per PATID, integer codes, NaN = unknown
    outcomes: pd.DataFrame             # the five outcomes, original scale, NaN = unobserved
    notes: list = field(default_factory=list)


def _codes_of(info, kind):
    return set(info[f"{kind}_values"]) if info is not None else set()


def _extra_missing(text):
    return {float(x) for x in str(text).split(";") if str(x).strip() not in ("", "nan")}


def _range(info, override):
    if isinstance(override, str) and "-" in override:
        lo, hi = override.split("-")
        return float(lo), float(hi), None
    if info is None:
        return None, None, None
    special = set().union(*(_codes_of(info, k) for k in
                            ("missing", "not_applicable", "below_detection", "above_detection")))
    return valid_values(info["responses"], special)


def _onehot_levels(info):
    special = _codes_of(info, "missing") | _codes_of(info, "not_applicable")
    codes = {float(c) for c, _ in _CODE_PATTERN.findall(str(info["responses"]))}
    return sorted(codes - special)


def _year_rules(values, rules, year):
    for rule in rules:
        if rule.startswith("zero_missing_from:"):
            values = values.mask((values == 0) & (year >= int(rule.split(":")[1])))
        elif rule.startswith("zero_missing_before:"):
            values = values.mask((values == 0) & (year < int(rule.split(":")[1])))
    return values


def clean(df: pd.DataFrame, cb: pd.DataFrame, feature_map: pd.DataFrame,
          patient_column="PATID", visit_column="VISIT", group_column="STUDYID",
          date_column="A1_EVENTDATEX") -> CleanData:
    df = df.reset_index(drop=True)
    col = {normalize_name(c): c for c in df.columns}
    raw = lambda name: pd.to_numeric(df[col[normalize_name(name)]], errors="coerce")
    info_by = cb.dropna(subset=["data_column"]).copy()
    info_by.index = info_by["data_column"].map(normalize_name)

    date = pd.to_datetime(df[date_column], format="mixed")
    keys = pd.DataFrame({"PATID": df[patient_column], "VISIT": df[visit_column],
                         "STUDYID": df[group_column], "DATE": date, "YEAR": date.dt.year})
    order = keys.sort_values(["PATID", "VISIT"]).index
    keys, df, date = keys.loc[order].reset_index(drop=True), df.loc[order].reset_index(drop=True), date.loc[order].reset_index(drop=True)
    year = keys["YEAR"]
    patient = keys["PATID"]

    # Visit-level masks for forms that were not done but stored as zeros.
    i1_not_done = raw("I1_INRELTO") == 0

    features, by_cat, notes = {}, {c: [] for c in CATEGORIES}, []
    below, above = {}, {}
    fm = feature_map.fillna("")

    for _, r in fm[fm["category"].isin(CATEGORIES + ["STATIC"])].iterrows():
        name, cat, tf = r["variable"], r["category"], r["transform"]
        rules = [x for x in str(r.get("rule", "")).split(";") if x]
        if r["form"] == "DERIVED":
            continue  # handled after the loop
        key = normalize_name(r["data_column"])
        if key not in col:
            notes.append(f"{name}: not in data, skipped")
            continue
        info = info_by.loc[key] if key in info_by.index else None
        v = raw(key)

        # 1. codes from the codebook and the map
        below_mask = v.isin(_codes_of(info, "below_detection"))
        above_mask = v.isin(_codes_of(info, "above_detection"))
        drop_codes = (_codes_of(info, "missing") | _codes_of(info, "not_applicable")
                      | _extra_missing(r.get("extra_missing", "")))
        if "never_smoker_zero" in rules:   # 0 and 8 (N/A) mean "never smoked"
            never = raw("A5_TOBAC100") == 0
            v = v.mask(never, 0.0)
            drop_codes -= {0.0}
            v = v.mask(~never & v.isin(drop_codes | {8.0}))
        else:
            v = v.mask(v.isin(drop_codes))
        v = v.mask(below_mask | above_mask)

        # 2. range (not for binary/one-hot, whose valid values are fixed below)
        if tf in ("numeric", "carry_forward", "embedding"):
            lo, hi, allowed = _range(info, r.get("range_override", ""))
            if allowed is not None:
                v = v.where(v.isin(allowed))
            elif lo is not None:
                v = v.where(v.between(lo, hi))
            elif tf == "carry_forward":
                v = v.where(v >= 0)

        # 3. year rules and logical rules
        v = _year_rules(v, rules, year)
        if "cohabit_daily" in rules:
            v = v.mask(raw("I1_INLIVWTH") == 1, 1.0)
        if "mask_if_i1_not_done" in rules:
            v = v.mask(i1_not_done)

        # 4. transform into model inputs
        if tf in BINARY_MAPS:
            v = v.map(BINARY_MAPS[tf])
        elif tf == "age_at_visit":
            v = (year - v).where(v.between(1900, 2020))
        elif tf == "years_since_onset":
            v = (raw("AGE") - v).where(lambda s: s >= 0)

        if tf == "carry_forward":   # biomarkers: keep only the assay visit, then carry forward
            panel = "PROTEOMICS" if name.upper().startswith("PROTEOMICS") else r["form"].upper()
            assay_visit = raw(PANEL_TIMEPOINT[panel])
            at_assay = keys["VISIT"] == assay_visit
            v = v.where(at_assay)
            below[key] = below_mask.where(at_assay, False).groupby(patient).cummax().astype(bool)
            above[key] = above_mask.where(at_assay, False).groupby(patient).cummax().astype(bool)
            v = v.groupby(patient).ffill()
        elif "carry_forward" in rules:
            v = v.groupby(patient).ffill()

        if cat == "STATIC":
            features[key] = v
            continue
        if tf == "onehot":
            levels = _onehot_levels(info) if info is not None else sorted(v.dropna().unique())
            if key == "B5_NPIQINF":
                levels = [x for x in levels if x != 0]
            for level in levels:
                c = f"{key}={int(level)}"
                features[c] = (v == level).astype(float).where(v.isin(levels))
                by_cat[cat].append(c)
        else:
            features[key] = v
            by_cat[cat].append(key)

    feats = pd.DataFrame(features)

    # Derived predictors
    def any_of(names):
        parts = pd.concat([feats[normalize_name(n)] if normalize_name(n) in feats
                           else raw(n).map(BINARY_MAPS["binary"]) for n in names], axis=1)
        out = parts.max(axis=1)  # 1 if any is 1; 0 if all observed are 0
        return out.where(parts.notna().any(axis=1))

    for _, r in fm[(fm["form"] == "DERIVED") & fm["category"].isin(CATEGORIES)].iterrows():
        name = r["variable"]
        if name in DERIVED_OUTCOMES:
            continue
        if str(r["transform"]).startswith("any_of:"):
            feats[name] = any_of(r["transform"].split(":", 1)[1].split("|"))
            by_cat[r["category"]].append(name)

    outcomes = build_outcomes(df)
    for name in DERIVED_OUTCOMES:
        if name in set(fm.loc[fm["category"] == "ABC_HISTORY", "variable"]):
            feats[f"OUT_{name}"] = outcomes[name].to_numpy()
            by_cat["ABC_HISTORY"].append(f"OUT_{name}")

    # Staleness of carried-forward biomarkers: years since each panel was assayed.
    for panel, tp in PANEL_TIMEPOINT.items():
        if normalize_name(tp) not in col:
            continue
        assay_date = date.where(keys["VISIT"] == raw(tp)).groupby(patient).ffill()
        c = f"{panel}_YEARS_SINCE_ASSAY"
        feats[c] = (date - assay_date).dt.days / 365.25
        by_cat["BIOMARKER"].append(c)

    static_cols = [normalize_name(v) for v in fm.loc[fm["category"] == "STATIC", "data_column"]
                   if normalize_name(v) in feats]
    static = (feats[static_cols].groupby(patient).agg(lambda s: s.dropna().mode().iloc[0]
                                                      if s.notna().any() else np.nan))
    feats = feats.drop(columns=static_cols)

    det_index = feats.index
    return CleanData(
        keys=keys, features=feats[[c for cat in CATEGORIES for c in by_cat[cat]]],
        columns_by_category=by_cat,
        below_detection=pd.DataFrame(below, index=det_index),
        above_detection=pd.DataFrame(above, index=det_index),
        static=static, outcomes=outcomes.reset_index(drop=True), notes=notes)