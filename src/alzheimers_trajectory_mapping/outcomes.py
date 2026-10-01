"""Build the five ABC outcomes from the raw TARCC columns.

Each outcome is a whole number inside a fixed range, or NaN when it was not
validly observed. Outcomes are never imputed; NaN becomes outcome mask = 0.
Decisions behind every rule are in docs/preprocessing_decisions.md.
"""

import numpy as np
import pandas as pd

# name: (domain, lowest score, highest score)
OUTCOMES = {
    "PSMS": ("A", 6, 30),
    "IADL": ("A", 8, 31),
    "NPIQ_SEV": ("B", 0, 36),
    "NPIQ_COUNT": ("B", 0, 12),
    "MMSE": ("C", 0, 30),
}

PSMS_ITEMS = [f"F1_PSMS{i}" for i in range(1, 7)]
# Valid (non-"not applicable") range of each IADL item, from the codebook.
IADL_ITEM_RANGES = {
    "F2_IADL1": (1, 5), "F2_IADL2": (1, 4), "F2_IADL3": (1, 4), "F2_IADL4": (1, 5),
    "F2_IADL5": (1, 3), "F2_IADL6": (1, 4), "F2_IADL7": (1, 3), "F2_IADL8": (1, 3),
}
NPI_SYMPTOMS = ["DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT",
                "APA", "DISN", "IRR", "MOT", "NITE", "APP"]
NPI_INFORMANT = "B5_NPIQINF"
VALID_INFORMANT = [1, 2, 3]  # spouse, child, other; 0 = form not done


def forms_not_done(df: pd.DataFrame) -> pd.DataFrame:
    """Flag visits where a form was skipped but stored as zeros.

    Used for the outcomes here and later for the predictor mask m_t.
    """
    return pd.DataFrame({
        "F1_NOT_DONE": df["F1_PSMSTOTSCR"] == 0,
        "F2_NOT_DONE": df["F2_IADLTOTSCR"] == 0,
        "B5_NOT_DONE": ~df[NPI_INFORMANT].isin(VALID_INFORMANT),
    }, index=df.index)


def _in_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.where(s.between(lo, hi))


def psms(df: pd.DataFrame) -> pd.Series:
    """PSMS total; values outside 6-30 (including the not-done 0) become NaN."""
    return _in_range(df["F1_PSMSTOTSCR"], 6, 30)


def iadl(df: pd.DataFrame) -> pd.Series:
    """IADL total; NaN unless all 8 items are applicable and valid."""
    items_ok = pd.Series(True, index=df.index)
    for col, (lo, hi) in IADL_ITEM_RANGES.items():
        items_ok &= df[col].between(lo, hi)
    total = _in_range(df["F2_IADLTOTSCR"], 8, 31)
    # The total must equal the item sum; anything else is an entry error.
    matches_items = total == df[list(IADL_ITEM_RANGES)].sum(axis=1)
    return total.where(items_ok & matches_items)


def npiq(df: pd.DataFrame) -> pd.DataFrame:
    """NPI-Q symptom count (0-12) and total severity (0-36)."""
    present = df[[f"B5_{s}" for s in NPI_SYMPTOMS]].to_numpy(dtype=float)
    severity = df[[f"B5_{s}SEV" for s in NPI_SYMPTOMS]].to_numpy(dtype=float)

    form_done = df[NPI_INFORMANT].isin(VALID_INFORMANT).to_numpy()
    all_answered = np.isin(present, [0, 1]).all(axis=1)  # excludes 9 = unknown
    count_ok = form_done & all_answered

    yes = present == 1
    sev_valid = np.isin(severity, [1, 2, 3])
    sev_ok = count_ok & ~(yes & ~sev_valid).any(axis=1)

    count = np.where(count_ok, yes.sum(axis=1), np.nan)
    total_sev = np.where(sev_ok, np.where(yes, severity, 0).sum(axis=1), np.nan)
    return pd.DataFrame({"NPIQ_SEV": total_sev, "NPIQ_COUNT": count}, index=df.index)


def mmse(df: pd.DataFrame) -> pd.Series:
    """MMSE 0-30; a 0 is kept only with CDR global >= 2 (severe dementia)."""
    score = _in_range(df["C1_MMSE"], 0, 30)
    implausible_zero = (score == 0) & ~(df["C1_CDRGLOB"] >= 2)
    return score.mask(implausible_zero)


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """All five outcomes, one row per visit, on their original scale."""
    out = pd.DataFrame({"PSMS": psms(df), "IADL": iadl(df)}, index=df.index)
    out = out.join(npiq(df))
    out["MMSE"] = mmse(df)
    return out[list(OUTCOMES)]