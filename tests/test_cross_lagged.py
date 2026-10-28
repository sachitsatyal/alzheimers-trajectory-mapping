import numpy as np
import pandas as pd

from alzheimers_trajectory_mapping.cross_lagged import cluster_ols, cross_lagged_table, visit_pairs


def test_cluster_ols_recovers_coefficients_and_widens_se_for_clustered_errors():
    rng = np.random.default_rng(0)
    G, m = 300, 4
    groups = np.repeat(np.arange(G), m)
    x = rng.normal(size=G * m)
    y = 1.0 + 0.5 * x + np.repeat(rng.normal(size=G), m) + 0.3 * rng.normal(size=G * m)
    X = np.column_stack([np.ones_like(x), x])
    beta, se, p, n_groups = cluster_ols(y, X, groups)
    assert n_groups == G
    assert abs(beta[1] - 0.5) < 3 * se[1]
    assert p[1] < 1e-6
    naive = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * np.var(y - X @ beta))
    assert se[0] > naive[0]  # shared patient effects -> larger intercept uncertainty


def _fake_pairs(n_patients=400, visits=4, effect=0.8, seed=0):
    """Pairs where cognition (MMSE, oriented worse = higher) drives IADL change."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for _ in range(visits):
            rows.append({"PATID": pid, "gap": rng.uniform(0.6, 1.8), "age": rng.normal(72, 8),
                         "dx": rng.choice(["normal", "MCI/other", "dementia"]),
                         "IADL_now": rng.normal(12, 3), "NPIQ_SEV_now": rng.normal(3, 2),
                         "MMSE_now": -rng.normal(25, 3), "PSMS_now": rng.normal(8, 2),
                         "NPIQ_COUNT_now": rng.normal(2, 1)})
    d = pd.DataFrame(rows)
    z_cog = (d["MMSE_now"] - d["MMSE_now"].mean()) / d["MMSE_now"].std()
    d["IADL_change"] = effect * z_cog + rng.normal(0, 1, len(d))
    for o in ["PSMS", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]:
        d[f"{o}_change"] = rng.normal(0, 1, len(d))
    return d


def test_cross_lagged_finds_planted_link_and_nothing_else():
    tab = cross_lagged_table(_fake_pairs()).set_index(["outcome", "predictor"])
    assert abs(tab.loc[("IADL", "cognition"), "coef"] - 0.8) < 0.1
    assert tab.loc[("IADL", "cognition"), "p"] < 1e-6
    assert abs(tab.loc[("IADL", "behavior"), "coef"]) < 0.1
    assert abs(tab.loc[("NPIQ_SEV", "cognition"), "coef"]) < 0.1


def test_visit_pairs_orientation_and_gaps():
    # 1 patient, 3 visits; outcomes: one "higher is better" (MMSE) and one normal (IADL)
    meta = {"outcomes": {"IADL": {"low": 8}, "MMSE": {"low": 0}},
            "feature_columns": ["AGE"], "standardizer": {"mean": {"AGE": 70.0}, "std": {"AGE": 10.0}}}
    arrays = {"Y": np.array([[[2, 28], [4, 26], [5, 25]]]), "MU": np.ones((1, 3, 2)), "PI": np.ones((1, 3)),
              "tau": np.array([[[0.0, 0], [1.0, 1], [4.0, 3]]]), "X": np.zeros((1, 3, 1)),
              "M": np.ones((1, 3, 1)), "PATID": np.array([7])}
    pairs = visit_pairs(arrays, meta)
    assert len(pairs) == 1                                     # second pair has a 3-year gap -> dropped
    row = pairs.iloc[0]
    assert row["IADL_now"] == 10 and row["IADL_change"] == 2   # 8 + 2 -> 8 + 4
    assert row["MMSE_now"] == -28 and row["MMSE_change"] == 2  # MMSE fell 28 -> 26 = 2 points worse
    assert row["age"] == 70.0


def test_within_person_effect_separated_from_between_person_effect():
    """Change driven by (a) cognition worse than the patient's own average (within) and
    (b) generally worse function (between only). Within-person fit must find (a) and not (b)."""
    rng = np.random.default_rng(1)
    rows = []
    for pid in range(500):
        f_mean, c_mean = rng.normal(), rng.normal()
        for _ in range(4):
            f, c = f_mean + 0.7 * rng.normal(), c_mean + 0.7 * rng.normal()
            rows.append({"PATID": pid, "gap": 1.0, "age": 70.0, "dx": "normal", "f": f, "c": c,
                         "f_mean": f_mean, "b": rng.normal()})
    d = pd.DataFrame(rows)
    d["IADL_now"], d["MMSE_now"], d["NPIQ_SEV_now"] = d["f"], d["c"], d["b"]
    c_dev = d["c"] - d.groupby("PATID")["c"].transform("mean")
    f_avg = d.groupby("PATID")["f"].transform("mean")
    d["PSMS_now"] = 0.0
    d["PSMS_change"] = 0.6 * c_dev + 1.0 * f_avg + 0.3 * rng.normal(size=len(d))
    for o in ["IADL", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]:
        d[f"{o}_change"] = rng.normal(size=len(d))
    d["NPIQ_COUNT_now"] = 0.0
    scale = {c: (0.0, 1.0) for c in ["IADL_now", "NPIQ_SEV_now", "MMSE_now"]}
    tab = cross_lagged_table(d, scale, within=True).set_index(["outcome", "predictor", "part"])
    assert abs(tab.loc[("PSMS", "cognition", "within"), "coef"] - 0.6) < 0.08
    assert abs(tab.loc[("PSMS", "function", "within"), "coef"]) < 0.08
    assert abs(tab.loc[("PSMS", "function", "between"), "coef"] - 1.0) < 0.1


def test_neuropsych_composite_orientation():
    from alzheimers_trajectory_mapping.cross_lagged import add_composites
    rng = np.random.default_rng(2)
    n = 200
    worse = rng.normal(size=n)                       # worse cognition (negated MMSE, standardized)
    X = np.zeros((n, 1, 3))
    X[:, 0, 1] = worse + 0.3 * rng.normal(size=n)    # e.g. trail-making time: higher = worse
    X[:, 0, 2] = -worse + 0.3 * rng.normal(size=n)   # e.g. scaled memory score: higher = better
    meta = {"category_blocks": {"ABC_HISTORY": [1, 3]}, "feature_columns": ["AGE", "C1_TRAILB", "C1_SS_LM"]}
    pairs = pd.DataFrame({"row": np.arange(n), "t": 0, "MMSE_now": -(25 - 3 * worse),
                          "IADL_now": rng.normal(size=n), "PSMS_now": rng.normal(size=n)})
    out, tests = add_composites(pairs, {"X": X, "M": np.ones_like(X)}, meta)
    assert list(tests["sign"]) == [1.0, -1.0]
    assert np.corrcoef(out["cognition_comp_now"], worse)[0, 1] > 0.9


def test_history_adjustment_removes_a_proxy_link():
    """Function only looks predictive because it tracks cognition's recent slope; adding
    each domain's previous slope must remove the function -> MMSE link."""
    rng = np.random.default_rng(3)
    n = 3000
    slope = rng.normal(size=n)                               # cognition's recent decline (true driver)
    d = pd.DataFrame({"PATID": np.arange(n) // 3, "gap": 1.0, "age": 70.0, "dx": "normal",
                      "MMSE_now": rng.normal(size=n), "NPIQ_SEV_now": rng.normal(size=n),
                      "IADL_now": 0.8 * slope + 0.6 * rng.normal(size=n)})
    d["MMSE_change"] = 1.0 * slope + 0.5 * rng.normal(size=n)
    d["MMSE_prev_slope"] = slope
    for c in ["IADL", "PSMS", "NPIQ_SEV", "NEURO"]:
        d[f"{c}_prev_slope"] = rng.normal(size=n)
    d["years_since_entry"] = rng.uniform(1, 5, n)
    for o in ["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT"]:
        d[f"{o}_change"] = rng.normal(size=n)
    d["PSMS_now"], d["NPIQ_COUNT_now"] = 0.0, 0.0
    from alzheimers_trajectory_mapping.cross_lagged import HISTORY_COVARIATES
    plain = cross_lagged_table(d).set_index(["outcome", "predictor"])
    adjusted = cross_lagged_table(d, extra=HISTORY_COVARIATES)
    adjusted = adjusted[adjusted.part != "covariate"].set_index(["outcome", "predictor"])
    assert plain.loc[("MMSE", "function"), "p"] < 1e-6
    assert abs(adjusted.loc[("MMSE", "function"), "coef"]) < 0.05
