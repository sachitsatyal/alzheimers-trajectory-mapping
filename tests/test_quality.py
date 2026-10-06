import pandas as pd

from alzheimers_trajectory_mapping.quality import audit_variable, valid_values


def test_valid_values():
    assert valid_values("0-30, 99 = Missing", {99.0}) == (0.0, 30.0, None)
    assert valid_values("0, 0.5, 1-3", set()) == (0.0, 3.0, None)
    assert valid_values("0 = No, 1 = Yes, 9 = Unknown", {9.0}) == (0.0, 1.0, {0.0, 1.0})
    assert valid_values("XX.XX", set()) == (None, None, None)


def _info(responses, missing=()):
    return pd.Series({"responses": responses, "missing_values": list(missing),
                      "not_applicable_values": [], "below_detection_values": [],
                      "above_detection_values": []})


def test_codes_and_out_of_range():
    raw = pd.Series([10, 20, 99, 45, None], name="TEST")
    years = pd.Series([2008] * 5)
    patients = pd.Series([1, 1, 2, 2, 3])
    r = audit_variable(raw, _info("0-30, 99 = Missing", [99.0]), years, patients)
    assert r["n_valid"] == 2 and r["n_missing_code"] == 1 and r["n_out_of_range"] == 1
    assert r["n_patients"] == 1


def test_zero_pattern_by_year():
    # zeros only in 2008: looks like "test not given that year"
    raw = pd.Series([0, 0, 0, 12, 15, 18], name="TEST")
    years = pd.Series([2008, 2008, 2008, 2012, 2012, 2012])
    patients = pd.Series(range(6))
    r = audit_variable(raw, _info("0-30"), years, patients)
    assert r["year_zero_max"] == 100.0 and r["year_zero_min"] == 0.0