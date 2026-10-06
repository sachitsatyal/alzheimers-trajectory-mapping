import json

import numpy as np
import yaml

from alzheimers_trajectory_mapping.train import load_config, schedules, train

META = {
    "category_blocks": {"DEMOGRAPHIC": [0, 2], "CLINICAL": [2, 5], "ABC_HISTORY": [5, 8]},
    "static_columns": ["A1_SEX"], "static_levels": {"A1_SEX": [0.0, 1.0]},
    "outcomes": {"PSMS": {"domain": "A", "low": 6, "high": 30, "n": 24},
                 "IADL": {"domain": "A", "low": 8, "high": 31, "n": 23},
                 "NPIQ_SEV": {"domain": "B", "low": 0, "high": 36, "n": 36},
                 "NPIQ_COUNT": {"domain": "B", "low": 0, "high": 12, "n": 12},
                 "MMSE": {"domain": "C", "low": 0, "high": 30, "n": 30}},
    "t_max": 4,
}
CFG = {
    "data": {},
    "model": {"d": 16, "encoder_layers": 1, "encoder_heads": 2, "encoder_ff": 32, "dropout": 0.0,
              "time2vec_k": 4, "latent_blocks": {"S": 2, "A": 2, "B": 2, "C": 2}, "velocity_hidden": 16, "ode_steps": 2,
              "decoder_hidden": 16, "decoder_dropout": 0.0, "kappa_min": 0.01, "samples_train": 2,
              "n_subtypes": 2, "subtype_embedding_dim": 4, "membership_hidden": 8},
    "training": {"seed": 0, "batch_size": 4, "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0,
                 "max_epochs": 3, "patience": 5, "velocity_penalty": 1e-3, "kl_anneal_epochs": 1,
                 "subtype_temperature_start": 5.0, "subtype_temperature_epochs": 2},
}


def write_split(path, split, n=8, t=4):
    rng = np.random.default_rng(0)
    pi = np.ones((n, t), np.float32)
    pi[0, 2:] = 0
    arrays = {"X": rng.normal(size=(n, t, 8)).astype(np.float32), "M": np.ones((n, t, 8), np.float32),
              "tau": rng.random((n, t, 2)).astype(np.float32), "dt_next": (rng.random((n, t)) * pi).astype(np.float32),
              "Y": rng.integers(0, 12, (n, t, 5)), "MU": np.repeat(pi[..., None], 5, -1),
              "PI": pi, "S": rng.integers(0, 3, (n, 1)), "PATID": np.arange(n), "STUDYID": np.arange(n)}
    np.savez_compressed(path / f"{split}.npz", **arrays)


def test_schedules():
    tcfg = {"kl_anneal_epochs": 10, "subtype_temperature_start": 5.0, "subtype_temperature_epochs": 4}
    assert schedules(0, tcfg) == (0.0, 5.0)
    beta, temp = schedules(2, tcfg)
    assert abs(beta - 0.2) < 1e-9 and abs(temp - 3.0) < 1e-9
    assert schedules(20, tcfg) == (1.0, 1.0)


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(CFG))
    cfg = load_config(p, {"model.n_subtypes": 5, "training.lr": 0.1})
    assert cfg["model"]["n_subtypes"] == 5 and cfg["training"]["lr"] == 0.1


def test_end_to_end_training(tmp_path):
    for split in ("train", "val"):
        write_split(tmp_path, split)
    (tmp_path / "meta.json").write_text(json.dumps(META))
    cfg = {**CFG, "data": {"processed_dir": str(tmp_path)}}
    summary = train(cfg, tmp_path / "run", device="cpu", verbose=False)
    assert summary["epochs_run"] == 3 and np.isfinite(summary["best_val_loss"])
    assert (tmp_path / "run" / "best.pt").exists()
    lines = (tmp_path / "run" / "log.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[-1])
    assert set(record["val"]["mae"]) == set(META["outcomes"])


def test_log_contains_interaction_map(tmp_path):
    for split in ("train", "val"):
        write_split(tmp_path, split)
    (tmp_path / "meta.json").write_text(json.dumps(META))
    cfg = {**CFG, "data": {"processed_dir": str(tmp_path)},
           "training": {**CFG["training"], "max_epochs": 1}}
    train(cfg, tmp_path / "run", device="cpu", verbose=False)
    record = json.loads((tmp_path / "run" / "log.jsonl").read_text().splitlines()[-1])
    assert "A<-B" in record["interaction_map"]["0"]