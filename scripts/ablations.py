"""Ablation study (paper Sec. III-K), 3 seeds each, compared with the base model (J3).

Run in parallel terminals by splitting the list:
  uv run python scripts/ablations.py --part 1 --of 3
  uv run python scripts/ablations.py --part 2 --of 3
  uv run python scripts/ablations.py --part 3 --of 3
Finished runs are skipped, so a stopped job can simply be restarted.
When all are done:
  uv run python scripts/ablations.py --summarize
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alzheimers_trajectory_mapping.evaluate import evaluate_run
from alzheimers_trajectory_mapping.train import load_config, train

BASE = "J3"  # base configuration = configs/default.yaml (J = 3), runs J3_s1..s3
CATEGORIES = ["DEMOGRAPHIC", "CLINICAL", "BIOMARKER", "GENOMIC", "SOCIAL", "CAREGIVER", "ABC_HISTORY"]
ABLATIONS = {
    "abl_no_ode": {"model.dynamics": "none"},             # no continuous-time dynamics
    "abl_sde": {"model.dynamics": "sde"},                 # stochastic dynamics (variant)
    "abl_no_time2vec": {"model.use_time2vec": False},
    "abl_no_masks": {"model.use_masks": False},
    "abl_no_anchor": {"model.decoder_anchor": False},
    "abl_no_coupling": {"model.coupling": False},
    "abl_deterministic": {"model.deterministic": True},
    "abl_one_subtype": {"model.n_subtypes": 1},           # no progression subtypes
    "abl_paper_context": {"model.decoder_exclude_categories": []},  # decoders also see ABC history
    **{f"abl_drop_{c.lower()}": {"model.drop_categories": [c]} for c in CATEGORIES},
}

parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--of", type=int, default=1)
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--summarize", action="store_true")
args = parser.parse_args()


def run_eval(run: str) -> pd.DataFrame:
    path = Path(f"outputs/runs/{run}/eval_val.csv")
    if not path.exists():
        evaluate_run(f"outputs/runs/{run}").to_csv(path)
    return pd.read_csv(path, index_col=0)


if not args.summarize:
    names = list(ABLATIONS)[args.part - 1::args.of]
    print("this part runs:", names, flush=True)
    for name in names:
        for seed in args.seeds:
            run = f"{name}_s{seed}"
            if Path(f"outputs/runs/{run}/eval_val.csv").exists():
                continue
            cfg = load_config("configs/default.yaml", {**ABLATIONS[name], "training.seed": seed})
            train(cfg, f"outputs/runs/{run}", verbose=False)
            crps = run_eval(run)["model_CRPS"].round(3).to_dict()
            print(f"{run}: CRPS {crps}", flush=True)
else:
    rows, sds = {}, {}
    for name in [BASE] + list(ABLATIONS):
        runs = [f"{name}_s{s}" for s in args.seeds if Path(f"outputs/runs/{name}_s{s}/best.pt").exists()]
        if len(runs) < len(args.seeds):
            print(f"{name}: only {len(runs)} finished runs, skipped")
            continue
        crps = pd.concat([run_eval(r)["model_CRPS"] for r in runs], axis=1)
        rows[name], sds[name] = crps.mean(1), crps.std(1)
    mean, sd = pd.DataFrame(rows).T, pd.DataFrame(sds).T
    change = (mean / mean.loc[BASE] - 1) * 100
    noise = np.sqrt(sd ** 2 + sd.loc[BASE] ** 2)  # sd of a difference of two means (approx.)
    flag = (mean - mean.loc[BASE]).abs() > 2 * noise
    pd.set_option("display.width", 200)
    print("mean CRPS (3 seeds)\n", mean.round(3).to_string())
    print("\n% change vs base (positive = ablation is WORSE); * = beyond ~2 sd of seed noise")
    print(change.round(1).astype(str).where(~flag, change.round(1).astype(str) + "*").to_string())
    mean.join(change.add_suffix("_%chg")).to_csv("outputs/ablations_summary.csv")