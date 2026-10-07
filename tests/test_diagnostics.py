import torch

from alzheimers_trajectory_mapping.diagnostics import domain_history_columns, shuffle_inputs

META = {"category_blocks": {"ABC_HISTORY": [2, 9]},
        "feature_columns": ["AGE", "D1_DEMENTED", "C1_CDRMEM", "C1_CDRHOB", "C1_GDS30",
                            "OUT_MMSE", "OUT_PSMS", "C1_CDRGLOB", "OUT_NPIQ_SEV"]}


def test_domain_assignment():
    assigned, unassigned = domain_history_columns(META)
    assert assigned == {"A": [3, 6], "B": [4, 8], "C": [2, 5]}
    assert unassigned == ["C1_CDRGLOB"]


def test_shuffle_changes_only_selected_columns():
    torch.manual_seed(0)
    batch = {"X": torch.randn(6, 4, 9), "M": torch.ones(6, 4, 9), "PI": torch.ones(6, 4)}
    out = shuffle_inputs([2, 5])(batch)
    keep = [i for i in range(9) if i not in (2, 5)]
    assert torch.equal(out["X"][..., keep], batch["X"][..., keep])
    assert not torch.equal(out["X"][..., [2, 5]], batch["X"][..., [2, 5]])
    assert torch.equal(out["PI"], batch["PI"])