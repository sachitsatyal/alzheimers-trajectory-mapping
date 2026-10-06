import numpy as np
import pandas as pd

from alzheimers_trajectory_mapping.sequences import Standardizer, build, time_inputs
from alzheimers_trajectory_mapping.transforms import CATEGORIES, CleanData


def fake_clean(n_patients=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        start = pd.Timestamp("2008-01-01") + pd.Timedelta(days=int(rng.integers(0, 2000)))
        for v in range(int(rng.integers(1, 6))):
            rows.append({"PATID": p, "VISIT": v + 1, "STUDYID": p // 2,
                         "DATE": start + pd.Timedelta(days=365 * v + int(rng.integers(-30, 30)))})
    keys = pd.DataFrame(rows)
    keys["YEAR"] = keys["DATE"].dt.year
    n = len(keys)
    features = pd.DataFrame({"AGE": rng.normal(75, 5, n), "D1_DEMENTED": rng.integers(0, 2, n).astype(float),
                             "Q1_NFL": rng.normal(20, 5, n), "OUT_MMSE": rng.integers(0, 31, n).astype(float)})
    features.loc[::3, "Q1_NFL"] = np.nan
    by_cat = {c: [] for c in CATEGORIES}
    by_cat["DEMOGRAPHIC"], by_cat["CLINICAL"] = ["AGE"], ["D1_DEMENTED"]
    by_cat["BIOMARKER"], by_cat["ABC_HISTORY"] = ["Q1_NFL"], ["OUT_MMSE"]
    below = pd.DataFrame({"Q1_NFL": False}, index=features.index)
    below.loc[0, "Q1_NFL"] = True
    outcomes = pd.DataFrame({"PSMS": 6.0, "IADL": 8.0, "NPIQ_SEV": 0.0, "NPIQ_COUNT": 0.0,
                             "MMSE": features["OUT_MMSE"]}, index=features.index)
    outcomes.loc[1, "IADL"] = np.nan
    static = pd.DataFrame({"A1_SEX": rng.integers(1, 3, n_patients).astype(float),
                           "APOE_E4_COUNT": rng.integers(0, 3, n_patients).astype(float)},
                          index=pd.Index(range(n_patients), name="PATID"))
    static.loc[0, "APOE_E4_COUNT"] = np.nan
    return CleanData(keys, features, by_cat, below, below.copy() & False, static, outcomes)


def test_groups_stay_together_and_shapes():
    d = fake_clean()
    arrays, meta = build(d, t_max=5)
    seen = {}
    for split, a in arrays.items():
        assert a["X"].shape == (len(a["PATID"]), 5, 4)
        for sid in a["STUDYID"]:
            assert seen.setdefault(sid, split) == split  # a STUDYID never spans two splits
    assert sum(meta["n_patients"].values()) == 40


def test_standardizer_uses_training_rows_only():
    x = pd.DataFrame({"a": [0.0, 2.0, 100.0]})
    flags = pd.DataFrame(index=x.index)
    s = Standardizer().fit(x.iloc[:2], flags.iloc[:2], flags.iloc[:2])
    z, m = s.transform(x, flags, flags)
    assert s.mean["a"] == 1.0 and z.loc[0, "a"] == -z.loc[1, "a"]


def test_missing_becomes_zero_with_mask_zero_and_detection_fill():
    d = fake_clean()
    arrays, meta = build(d, t_max=5)
    a = arrays["train"]
    real = a["PI"] == 1
    assert np.all(a["X"][a["M"] == 0] == 0)
    assert np.all(a["M"][~real] == 0) and np.all(a["MU"][~real] == 0)


def test_time_inputs():
    keys = pd.DataFrame({"PATID": [1, 1, 1], "DATE": pd.to_datetime(["2010-01-01", "2011-01-01", "2013-01-01"])})
    t = time_inputs(keys)
    assert t["dt"].iloc[0] == 0 and t["dt_next"].iloc[-1] == 0
    assert abs(t["t_abs"].iloc[2] - 3.0) < 0.01 and abs(t["dt_next"].iloc[1] - 2.0) < 0.01


def test_outcomes_shifted_and_masked():
    d = fake_clean()
    arrays, meta = build(d, t_max=5)
    a = arrays["train"]
    psms = list(meta["outcomes"]).index("PSMS")
    assert np.all(a["Y"][..., psms][a["MU"][..., psms] == 1] == 0)  # PSMS 6 shifted to 0
    assert a["S"].min() >= 0