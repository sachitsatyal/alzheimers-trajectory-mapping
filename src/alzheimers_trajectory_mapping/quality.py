"""Quality audit for every predictor kept in configs/feature_map.csv.

For each variable it reports coverage, range violations, near-constancy,
leftover codes, and whether zeros pile up in some calendar years only
(the "0 means not given" pattern). Nothing is changed in the data here:
the audit only describes, so decisions can be made and written down.
"""

import re

import numpy as np
import pandas as pd

from .codebook import _CODE_PATTERN, normalize_name

AUDITED_CATEGORIES = ["STATIC", "DEMOGRAPHIC", "CLINICAL", "BIOMARKER", "GENOMIC",
                      "SOCIAL", "CAREGIVER", "ABC_HISTORY"]
# Ranges the codebook text states ambiguously, set by hand.
MANUAL_RANGES = {"C1_CERAD_LL_TOT": (0, 30)}
_RANGE_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![\d.])")


def valid_values(responses: str, special_codes: set) -> tuple:
    """Read the valid values of a variable from its codebook Responses text.

    Returns (low, high, allowed_set). allowed_set is filled only for purely
    coded variables such as "0 = No, 1 = Yes"; otherwise (low, high) bounds
    all numbers the codebook mentions (ranges and listed values). All three
    are None when the codebook gives no usable range (e.g. "XX.XX").
    """
    text = str(responses)
    if re.fullmatch(r"[\d.,\s-]+", text):  # plain list such as "0, 0.5, 1-3"
        numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        return min(numbers), max(numbers), None
    codes = {float(c) for c, _ in _CODE_PATTERN.findall(text)} - special_codes
    ranges = [(float(a), float(b)) for a, b in _RANGE_PATTERN.findall(text)]
    if not ranges and codes:
        return min(codes), max(codes), codes
    if ranges:
        bounds = [x for r in ranges for x in r] + list(codes)
        return min(bounds), max(bounds), None
    return None, None, None


def _zero_share_by_year(values: pd.Series, years: pd.Series) -> pd.Series:
    observed = values.notna()
    return (values[observed] == 0).groupby(years[observed]).mean()


def audit_variable(raw: pd.Series, info: pd.Series, years: pd.Series, patients: pd.Series) -> dict:
    special = set(info["missing_values"]) | set(info["not_applicable_values"])
    detection = set(info["below_detection_values"]) | set(info["above_detection_values"])
    values = pd.to_numeric(raw, errors="coerce")
    non_numeric = int((raw.notna() & values.isna()).sum())

    n_missing_code = int(values.isin(info["missing_values"]).sum())
    n_not_applicable = int(values.isin(info["not_applicable_values"]).sum())
    n_detection = int(values.isin(detection).sum())
    clean = values.mask(values.isin(special | detection))

    low, high, allowed = valid_values(info["responses"], special | detection)
    if raw.name.upper() in MANUAL_RANGES:
        (low, high), allowed = MANUAL_RANGES[raw.name.upper()], None
    if allowed is not None:
        out_of_range = clean.notna() & ~clean.isin(allowed)
    elif low is not None:
        out_of_range = clean.notna() & ~clean.between(low, high)
    else:
        out_of_range = clean < 0  # no codebook range: only negatives are impossible
    valid = clean.mask(out_of_range)

    observed = valid.notna()
    by_year = observed.groupby(years).mean()
    zeros = _zero_share_by_year(valid, years)
    top_share = valid.value_counts(normalize=True).iloc[0] if observed.any() else np.nan

    return {
        "n_valid": int(observed.sum()),
        "pct_visits": round(100 * observed.mean(), 1),
        "n_patients": int(patients[observed].nunique()),
        "n_unique": int(valid.nunique()),
        "top_value_pct": round(100 * top_share, 1) if observed.any() else np.nan,
        "pct_zero": round(100 * (valid[observed] == 0).mean(), 1) if observed.any() else np.nan,
        "min": valid.min(), "max": valid.max(),
        "range_low": low, "range_high": high,
        "n_out_of_range": int(out_of_range.sum()),
        "n_missing_code": n_missing_code,
        "n_not_applicable": n_not_applicable,
        "n_detection_limit": n_detection,
        "n_non_numeric": non_numeric,
        "year_coverage_min": round(100 * by_year.min(), 1),
        "year_coverage_max": round(100 * by_year.max(), 1),
        "year_zero_min": round(100 * zeros.min(), 1) if len(zeros) else np.nan,
        "year_zero_max": round(100 * zeros.max(), 1) if len(zeros) else np.nan,
    }


def flag(row: pd.Series, n_patients_total: int) -> str:
    """Short labels for anything that needs a decision."""
    flags = []
    if row["n_valid"] == 0:
        return "EMPTY"
    if row["n_patients"] < 0.10 * n_patients_total:
        flags.append("LOW_COVERAGE")
    if row["top_value_pct"] >= 99:
        flags.append("NEAR_CONSTANT")
    if row["n_out_of_range"] > 0:
        flags.append("OUT_OF_RANGE")
    if row["n_non_numeric"] > 0:
        flags.append("NON_NUMERIC")
    if row["year_coverage_max"] - row["year_coverage_min"] >= 50:
        flags.append("COVERAGE_CHANGES_BY_YEAR")
    if row["year_zero_max"] - row["year_zero_min"] >= 50:
        flags.append("ZERO_PATTERN_BY_YEAR")
    return ", ".join(flags)


def audit(df: pd.DataFrame, cb: pd.DataFrame, feature_map: pd.DataFrame,
          date_column: str = "A1_EVENTDATEX", patient_column: str = "PATID") -> pd.DataFrame:
    """One report row per kept predictor that exists in the raw data."""
    years = pd.to_datetime(df[date_column], format="mixed").dt.year
    patients = df[patient_column]
    info_by_column = cb.dropna(subset=["data_column"]).set_index("data_column")
    empty_info = pd.Series({"missing_values": [], "not_applicable_values": [],
                            "below_detection_values": [], "above_detection_values": [],
                            "responses": ""})

    # Match on normalized names: some raw columns keep punctuation (e.g. "-")
    # that the map's normalized data_column does not.
    column_lookup = {normalize_name(c): c for c in df.columns}
    info_by_column = info_by_column.rename(index=lambda c: normalize_name(c))

    rows = []
    kept = feature_map[feature_map["category"].isin(AUDITED_CATEGORIES)]
    for _, fm in kept.iterrows():
        key = normalize_name(fm["data_column"])
        col = column_lookup.get(key)
        if col is None:
            continue  # derived outcomes are built and checked in outcomes.py
        info = info_by_column.loc[key] if key in info_by_column.index else empty_info
        row = {"variable": fm["variable"], "category": fm["category"], "transform": fm["transform"]}
        row.update(audit_variable(df[col].rename(key), info, years, patients))
        rows.append(row)

    report = pd.DataFrame(rows)
    report["flags"] = report.apply(flag, axis=1, n_patients_total=patients.nunique())
    return report