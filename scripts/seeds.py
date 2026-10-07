"""Train one configuration with several seeds and summarize (mean and sd).

  uv run python scripts/seeds.py --name kl001 --seeds 1 2 3
  uv run python scripts/seeds.py --name noanchor --set model.decoder_anchor=false
Runs go to outputs/runs/<name>_s<seed>; the summary to outputs/runs/<name>_seeds_summary.csv.
The data split is fixed by preprocessing; the seed changes initialization and batch order.
"""
import argparse

import pandas as pd
import yaml

from alzheimers_trajectory_mapping.evaluate import evaluate_run
from alzheimers_trajectory_mapping.train import load_config, train

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--set", nargs="*", default=[])
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()

overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
tables = []
for seed in args.seeds:
    run = f"outputs/runs/{args.name}_s{seed}"
    cfg = load_config(args.config, {**overrides, "training.seed": seed})
    train(cfg, run, verbose=False)
    table = evaluate_run(run)
    tables.append(table)
    print(f"seed {seed}: CRPS", table["model_CRPS"].round(3).to_dict(), flush=True)

metrics = ["model_CRPS", "model_MAE", "model_NLL", "model_cover90"]
summary = pd.concat(tables).groupby(level=0)[metrics].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 200)
print(summary.to_string())
summary.to_csv(f"outputs/runs/{args.name}_seeds_summary.csv")