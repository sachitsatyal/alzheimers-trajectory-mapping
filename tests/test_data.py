import json

import numpy as np
import torch

from alzheimers_trajectory_mapping.data import TARCCSequences, load_meta, make_loader


def write_fake(tmp_path, n=10, t=4, p=6, k=5, s=3):
    rng = np.random.default_rng(0)
    arrays = {
        "X": rng.normal(size=(n, t, p)).astype(np.float32),
        "M": np.ones((n, t, p), np.float32),
        "tau": np.zeros((n, t, 2), np.float32),
        "dt_next": np.ones((n, t), np.float32),
        "Y": rng.integers(0, 10, (n, t, k)),
        "MU": np.ones((n, t, k), np.float32),
        "PI": np.ones((n, t), np.float32),
        "S": rng.integers(0, 3, (n, s)),
        "PATID": np.arange(n),
        "STUDYID": np.arange(n),
    }
    np.savez_compressed(tmp_path / "train.npz", **arrays)
    (tmp_path / "meta.json").write_text(json.dumps({"t_max": t}))


def test_dataset_item(tmp_path):
    write_fake(tmp_path)
    ds = TARCCSequences(tmp_path, "train")
    item = ds[0]
    assert len(ds) == 10
    assert item["X"].shape == (4, 6) and item["X"].dtype == torch.float32
    assert item["Y"].dtype == torch.long and item["S"].shape == (3,)
    assert load_meta(tmp_path)["t_max"] == 4


def test_loader_batches(tmp_path):
    write_fake(tmp_path)
    batch = next(iter(make_loader(tmp_path, "train", batch_size=4)))
    assert batch["X"].shape == (4, 4, 6) and batch["MU"].shape == (4, 4, 5)