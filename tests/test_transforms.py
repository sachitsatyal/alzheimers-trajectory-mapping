import numpy as np
import pandas as pd

from alzheimers_trajectory_mapping.outcomes import NPI_SYMPTOMS
from alzheimers_trajectory_mapping.transforms import clean


def codebook_row(var, responses, missing=(), na=(), below=(), above=()):
    return {"variable": var, "data_column": var.upper(), "form": var.split("_")[0].upper(),
            "responses": responses, "missing_values": list(missing),
            "not_applicable_values": list(na), "below_detection_values": list(below),
            "above_detection_values": list(above)}


CB = pd.DataFrame([
    codebook_row("A1_SEX", "1 = Male, 2 = Female"),
    codebook_row("A43_ADMedNow", "1 = Yes, 2 = No"),
    codebook_row("A5_PACKSPER", "1 = (1 cigarette), 2 = x, 3 = y, 4 = z, 5 = w, 8 = N/A, 9 = Unknown", [9], [8]),
    codebook_row("I1_INVISITS", "1 = Daily, 2 = a, 3 = b, 4 = c, 5 = d, 6 = e"),
    codebook_row("I1_INRELTO", "1 = Spouse, 2 = Child, 3 = Other"),
    codebook_row("C1_CLOCK", "0-10"),
    codebook_row("Q1_NFL", "XX.X; -999 = No sample; -333 = below", [-999], below=[-333]),
])

FM = pd.DataFrame([
    # variable, data_column, form, description, category, transform, reason, extra_missing, range_override, rule
    ["A1_SEX", "A1_SEX", "A1", "", "STATIC", "embedding", "", "", "", ""],
    ["A43_ADMedNow", "A43_ADMEDNOW", "A43", "", "CLINICAL", "binary_12", "", "", "", ""],
    ["A5_PACKSPER", "A5_PACKSPER", "A5", "", "CLINICAL", "numeric", "", "", "0-5", "never_smoker_zero"],
    ["I1_INVISITS", "I1_INVISITS", "I1", "", "CAREGIVER", "numeric", "", "0", "", "cohabit_daily;mask_if_i1_not_done"],
    ["I1_INRELTO", "I1_INRELTO", "I1", "", "CAREGIVER", "onehot", "", "", "", "mask_if_i1_not_done"],
    ["C1_CLOCK", "C1_CLOCK", "C1", "", "ABC_HISTORY", "numeric", "", "", "", "zero_missing_from:2010"],
    ["Q1_NFL", "Q1_NFL", "Q1", "", "BIOMARKER", "carry_forward", "", "", "", ""],
    ["ANY_PARK", "", "DERIVED", "", "CLINICAL", "any_of:A5_PD|D1_PARK", "", "", "", ""],
    ["MMSE", "MMSE", "DERIVED", "", "ABC_HISTORY", "numeric", "", "", "", ""],
], columns=["variable", "data_column", "form", "description", "category", "transform",
            "reason", "extra_missing", "range_override", "rule"])


def visit(pat, visit, date, **kw):
    row = {"PATID": pat, "VISIT": visit, "STUDYID": pat, "A1_EVENTDATEX": date, "AGE": 75,
           "A1_SEX": 2, "A43_ADMEDNOW": 1, "A5_PACKSPER": 2, "A5_TOBAC100": 1,
           "I1_INVISITS": 3, "I1_INLIVWTH": 0, "I1_INRELTO": 2, "C1_CLOCK": 8,
           "Q1_NFL": 0, "Q1_QUANTERIX_TP": 2, "A5_PD": 0, "D1_PARK": 0,
           "F1_PSMSTOTSCR": 6, "F2_IADLTOTSCR": 8, "C1_MMSE": 28, "C1_CDRGLOB": 0, "B5_NPIQINF": 1}
    row.update({f"F1_PSMS{i}": 1 for i in range(1, 7)})
    row.update({f"F2_IADL{i}": 1 for i in range(1, 9)})
    for s in NPI_SYMPTOMS:
        row[f"B5_{s}"], row[f"B5_{s}SEV"] = 0, 0
    row.update(kw)
    return row


DATA = pd.DataFrame([
    visit(1, 1, "2009-01-10", A43_ADMEDNOW=0, C1_CLOCK=0, Q1_NFL=0),
    visit(1, 2, "2010-02-01", A5_TOBAC100=0, A5_PACKSPER=8, Q1_NFL=12.5, C1_CLOCK=0),
    visit(1, 3, "2011-03-01", I1_INLIVWTH=1, I1_INVISITS=0, A5_PD=1, Q1_NFL=0),
    visit(2, 1, "2012-01-05", I1_INRELTO=0, I1_INVISITS=0, Q1_NFL=-333, Q1_QUANTERIX_TP=1),
])


def run():
    return clean(DATA, CB, FM)


def test_binary_12_and_not_done_zero():
    f = run().features
    assert np.isnan(f.loc[0, "A43_ADMEDNOW"]) and f.loc[1, "A43_ADMEDNOW"] == 1


def test_never_smoker_is_zero_packs():
    f = run().features
    assert f.loc[1, "A5_PACKSPER"] == 0 and f.loc[0, "A5_PACKSPER"] == 2


def test_cohabit_daily_and_informant_not_done():
    f = run().features
    assert f.loc[2, "I1_INVISITS"] == 1           # lives with patient -> daily
    assert np.isnan(f.loc[3, "I1_INVISITS"])      # informant form not done
    assert np.isnan(f.loc[3, "I1_INRELTO=2"])     # one-hot masked too
    assert f.loc[0, "I1_INRELTO=2"] == 1 and f.loc[0, "I1_INRELTO=1"] == 0


def test_year_rule():
    f = run().features
    assert f.loc[0, "C1_CLOCK"] == 0              # 2009: real zero
    assert np.isnan(f.loc[1, "C1_CLOCK"])         # 2010: filler zero


def test_biomarker_carried_forward_never_backward():
    d = run()
    f = d.features
    assert np.isnan(f.loc[0, "Q1_NFL"])           # before assay visit
    assert f.loc[1, "Q1_NFL"] == 12.5 and f.loc[2, "Q1_NFL"] == 12.5
    assert np.isnan(f.loc[0, "Q1_YEARS_SINCE_ASSAY"]) and f.loc[1, "Q1_YEARS_SINCE_ASSAY"] == 0
    assert d.below_detection.loc[3, "Q1_NFL"] and not d.below_detection.loc[1, "Q1_NFL"]


def test_any_of_static_and_outcome():
    d = run()
    assert d.features.loc[2, "ANY_PARK"] == 1 and d.features.loc[0, "ANY_PARK"] == 0
    assert d.static.loc[1, "A1_SEX"] == 2
    assert d.features.loc[0, "OUT_MMSE"] == 28
    assert "A1_SEX" not in d.features.columns