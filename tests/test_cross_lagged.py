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
