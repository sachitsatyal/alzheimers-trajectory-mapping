"""Load the TARCC codebook and match it to the data columns.

Produces one row per codebook variable with:
  - the matching column name in the data (or None if absent)
  - the codes that mean "missing" for that variable
  - the codes that mean "not applicable" (kept separate on purpose)
  - the codes that mean "below / above the assay's detection limit"
    (biomarkers only; these are real measurements outside the assay range,
    not missing values, so they are kept separate)
"""

import re
from pathlib import Path

import pandas as pd

# Data column names that differ from the codebook beyond case/punctuation.
# Keys are normalized codebook names, values are data column names.
MANUAL_RENAMES = {
    "C1_C1DATX": "C1_DATEX",
    "I1_INDTECONTX": "I1_INDATECONTX",
    "APOE_GENOTYPE_DIGIT": "APOE_GENOTYPE_DIGITS",
    "PROTEOMOICS_TP": "PROTEOMICS_TP",
    "RBM_ALPHA_F": "RBM_ALPGA_F",
    "MISC_TARC_PAT": "PATID",
}

# Response labels that mean the value was not obtained.
MISSING_WORDS = ("unknown", "missing", "unavailable", "no data", "not done", "refused",
                 "not performed", "no sample", "insufficient volume", "invalid")
# Labels that mean the item did not apply; handled separately from missing.
NOT_APPLICABLE_WORDS = ("not applicable", "n/a")
# Labels for biomarker values outside the assay's detection range.
BELOW_DETECTION_WORDS = ("lldl", "below lowest detectable", "below detection")
ABOVE_DETECTION_WORDS = ("ghdl", "above highest detectable", "above detection")

_CODE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*=\s*([^,;\n]+)")


def normalize_name(name: str) -> str:
    """Uppercase, punctuation to underscores, and the PROTEOMICS_ -> PROTEO_ rename."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip()).strip("_").upper()
    if s.startswith("PROTEOMICS_") and s != "PROTEOMICS_TP":
        s = "PROTEO_" + s[len("PROTEOMICS_"):]
    return MANUAL_RENAMES.get(s, s)


def _is_not_applicable(label: str) -> bool:
    return any(w in label for w in NOT_APPLICABLE_WORDS) or label == "na" or label.startswith("na ")


def parse_codes(responses: str, missing_col) -> dict[str, list[float]]:
    """Classify every special code of one variable.

    Returns a dict with keys "missing", "not_applicable", "below_detection",
    "above_detection", each a sorted list of codes.
    """
    found = {"missing": set(), "not_applicable": set(),
             "below_detection": set(), "above_detection": set()}
    text = str(responses)
    for code, label in _CODE_PATTERN.findall(text):
        label, code = label.strip().lower(), float(code)
        if any(w in label for w in BELOW_DETECTION_WORDS):
            found["below_detection"].add(code)
        elif any(w in label for w in ABOVE_DETECTION_WORDS):
            found["above_detection"].add(code)
        elif _is_not_applicable(label):
            found["not_applicable"].add(code)
        elif any(w in label for w in MISSING_WORDS):
            found["missing"].add(code)
    # Proteomics: 'If value is missing or "0" the ... analysis was not completed'.
    if 'missing or "0"' in text:
        found["missing"].add(0.0)
    # The dedicated "Missing Value Codes" column, e.g. "-999, -888" or "0 = Unavailable".
    # Codes already classified above (e.g. detection limits) keep that classification.
    if isinstance(missing_col, str):
        classified = set().union(*found.values())
        for code in re.findall(r"-?\d+(?:\.\d+)?", missing_col):
            if float(code) not in classified:
                found["missing"].add(float(code))
    return {k: sorted(v) for k, v in found.items()}


def load_codebook(codebook_path: str | Path, data_columns) -> pd.DataFrame:
    """Read the codebook and attach data-column matches and code tables."""
    cb = pd.read_excel(codebook_path)
    cb.columns = ["order", "variable", "form_order", "description", "responses", "missing_codes"]
    cb = cb.dropna(subset=["variable"]).copy()
    cb["variable"] = cb["variable"].astype(str).str.strip()

    data_lookup = {normalize_name(c): c for c in data_columns}
    cb["data_column"] = cb["variable"].map(lambda v: data_lookup.get(normalize_name(v)))
    cb["form"] = cb["variable"].str.split("_").str[0].str.upper()

    codes = [parse_codes(r, m) for r, m in zip(cb["responses"], cb["missing_codes"])]
    for kind in ("missing", "not_applicable", "below_detection", "above_detection"):
        cb[f"{kind}_values"] = [c[kind] for c in codes]
    return cb.reset_index(drop=True)


def unmatched_data_columns(cb: pd.DataFrame, data_columns) -> list[str]:
    """Data columns that no codebook entry maps to."""
    matched = set(cb["data_column"].dropna())
    return sorted(c for c in data_columns if c not in matched)