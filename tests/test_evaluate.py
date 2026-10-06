import numpy as np

from alzheimers_trajectory_mapping.evaluate import (aggregate, change_distributions, last_observed_np,
                                                    locf_pmf, score_pmf)


def test_score_pmf_point_mass_is_perfect():
    pmf = np.zeros((3, 11))
    y = np.array([0, 4, 10])
    pmf[np.arange(3), y] = 1.0
    s = aggregate(score_pmf(pmf, y))
    assert s["MAE"] == 0 and s["RMSE"] == 0 and s["CRPS"] == 0 and s["cover90"] == 1.0
    assert abs(s["NLL"]) < 1e-9


def test_median_beats_mean_on_floor_heavy_scores():
    pmf = np.zeros((1, 25))
    pmf[0, 0], pmf[0, 3] = 0.8, 0.2       # mostly at the floor
    s = score_pmf(pmf, np.array([0]))
    assert s["abs_err"][0] == 0           # median = 0: correct
    assert s["sq_err"][0] > 0             # mean = 0.6: pulled off the floor


def test_locf_pmf_clips_to_range_and_uses_marginal_without_anchor():
    delta = np.zeros(7)
    delta[3 + 2] = 1.0                    # change of +2 with probability 1 (n = 3)
    marginal = np.full(4, 0.25)
    pmf = locf_pmf(np.array([1, 3, 0]), np.array([True, True, False]), delta, marginal, 3)
    assert pmf[0, 3] == 1.0               # 1 + 2 = 3
    assert pmf[1, 3] == 1.0               # 3 + 2 clipped to 3
    assert np.allclose(pmf[2], 0.25)
    assert np.allclose(pmf.sum(1), 1.0)


def test_last_observed_and_change_distribution():
    Y = np.array([[[2], [0], [5]]])
    MU = np.array([[[1.0], [0.0], [1.0]]])
    value, has = last_observed_np(Y, MU)
    assert value[0, :, 0].tolist() == [2, 2, 5] and has[0, :, 0].tolist() == [True, True, True]
    train = {"Y": Y, "MU": MU, "PI": np.ones((1, 3))}
    deltas, marginals = change_distributions(train, [5])
    assert deltas[0].argmax() == 5 + 3   # change 5 - 2 = +3 is the only observed change
    assert np.isclose(deltas[0].sum(), 1.0) and np.isclose(marginals[0].sum(), 1.0)