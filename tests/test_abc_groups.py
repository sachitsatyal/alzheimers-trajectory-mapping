import numpy as np
import pandas as pd
import torch

from alzheimers_trajectory_mapping.abc_groups import (group_change, grouped_permutation, permute_columns,
                                                      summarize)


def test_grouped_permutation_pairs_only_within_group():
    groups = np.array(["mci", "dem", "mci", "normal", "dem", "mci", "normal", "dem"])
    perm = grouped_permutation(groups, np.random.default_rng(0))
    assert sorted(perm) == list(range(len(groups)))          # a permutation
    assert (groups[perm] == groups).all()                      # partner has the same diagnosis


def test_permute_columns_swaps_whole_history_of_selected_columns():
    torch.manual_seed(0)
    t = {"X": torch.randn(4, 3, 6), "M": torch.ones(4, 3, 6), "PI": torch.ones(4, 3)}
    perm = [1, 0, 3, 2]
    out = permute_columns(t, [2, 4], perm)
    assert torch.equal(out["X"][0][:, [2, 4]], t["X"][1][:, [2, 4]])   # every visit from the partner
    keep = [0, 1, 3, 5]
    assert torch.equal(out["X"][..., keep], t["X"][..., keep])          # other columns untouched
    assert torch.equal(out["PI"], t["PI"])


def test_group_change_percentages():
    intact = pd.DataFrame({"patient": [0, 0, 1, 2], "outcome": ["MMSE"] * 4, "crps": [1.0, 1.0, 1.0, 2.0]})
    scrambled = intact.assign(crps=[1.1, 1.1, 1.2, 2.4])
    groups = pd.Series({0: "a", 1: "b", 2: "b"})
    res = group_change(intact, scrambled, groups).set_index("group")["pct"]
    assert np.isclose(res["a"], 10.0)
    assert np.isclose(res["b"], (3.6 / 3.0 - 1) * 100)
    assert np.isclose(res["all"], (5.8 / 5.0 - 1) * 100)
    absolute = group_change(intact, scrambled, groups).set_index("group")["abs_change"]
    assert np.isclose(absolute["a"], 0.1) and np.isclose(absolute["b"], 0.3)


def test_summarize_mean_over_runs_of_repeat_means():
    rows = []
    for run, base in [("r1", 10.0), ("r2", 20.0)]:
        for rep, noise in [(0, -1.0), (1, 1.0)]:
            rows.append({"run": run, "scheme": "random", "group": "all", "history": "C",
                         "outcome": "IADL", "repeat": rep, "pct": base + noise, "forecasts": 5})
    mean, sd = summarize(pd.DataFrame(rows), ["IADL"])[("random", "all")]
    assert np.isclose(mean.loc["C", "IADL"], 15.0)
    assert np.isclose(sd.loc["C", "IADL"], np.std([10.0, 20.0], ddof=1))
