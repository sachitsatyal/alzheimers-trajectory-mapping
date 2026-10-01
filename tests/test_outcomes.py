import numpy as np
import pandas as pd

from alzheimers_trajectory_mapping.outcomes import (
    NPI_SYMPTOMS, build_outcomes, forms_not_done,
)


def make_visit(**overrides):
    """One healthy, fully completed visit; override fields per test."""
    row = {"F1_PSMSTOTSCR": 6, "C1_MMSE": 28, "C1_CDRGLOB": 0, "B5_NPIQINF": 1}
    row.update({f"F1_PSMS{i}": 1 for i in range(1, 7)})
    row.update({f"F2_IADL{i}": 1 for i in range(1, 9)})
    row["F2_IADLTOTSCR"] = 8
    for s in NPI_SYMPTOMS:
        row[f"B5_{s}"], row[f"B5_{s}SEV"] = 0, 0
    row.update(overrides)
    return row


def outcomes(*rows):
    return build_outcomes(pd.DataFrame(list(rows)))


def test_complete_visit():
    out = outcomes(make_visit()).iloc[0]
    assert out.to_dict() == {"PSMS": 6, "IADL": 8, "NPIQ_SEV": 0, "NPIQ_COUNT": 0, "MMSE": 28}


def test_not_done_forms_become_missing():
    v = make_visit(F1_PSMSTOTSCR=0, F2_IADLTOTSCR=0, B5_NPIQINF=0,
                   **{f"F2_IADL{i}": 0 for i in range(1, 9)})
    out = outcomes(v).iloc[0]
    assert out[["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT"]].isna().all()
    assert out["MMSE"] == 28
    flags = forms_not_done(pd.DataFrame([v])).iloc[0]
    assert flags.all()


def test_iadl_not_applicable_item_masks_total():
    out = outcomes(make_visit(F2_IADL3=0, F2_IADLTOTSCR=7)).iloc[0]
    assert np.isnan(out["IADL"])
    assert out["PSMS"] == 6  # other outcomes unaffected


def test_npiq_count_and_severity():
    v = make_visit(B5_DEL=1, B5_DELSEV=2, B5_APA=1, B5_APASEV=3)
    out = outcomes(v).iloc[0]
    assert out["NPIQ_COUNT"] == 2 and out["NPIQ_SEV"] == 5


def test_npiq_yes_without_severity():
    out = outcomes(make_visit(B5_DEL=1, B5_DELSEV=0)).iloc[0]
    assert out["NPIQ_COUNT"] == 1 and np.isnan(out["NPIQ_SEV"])


def test_mmse_zero_depends_on_cdr():
    severe = outcomes(make_visit(C1_MMSE=0, C1_CDRGLOB=3)).iloc[0]
    mild = outcomes(make_visit(C1_MMSE=0, C1_CDRGLOB=0.5)).iloc[0]
    assert severe["MMSE"] == 0 and np.isnan(mild["MMSE"])