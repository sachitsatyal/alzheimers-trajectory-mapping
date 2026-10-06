"""Train the model.

Examples:
  uv run python scripts/train.py --name first_run
  uv run python scripts/train.py --name j3 --set model.n_subtypes=3 training.max_epochs=50
"""
import argparse

import yaml

from alzheimers_trajectory_mapping.train import load_config, train

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--name", required=True, help="run name; results go to outputs/runs/<name>")
parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.n_subtypes=3")
args = parser.parse_args()

overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
cfg = load_config(args.config, overrides)
print(train(cfg, f"outputs/runs/{args.name}"))