"""Granger-style ABC map: does one domain's past improve forecasts of the others?

  uv run python scripts/abc_map.py --names ownS4_s1 ownS4_s2 ownS4_s3
Prints the input-to-domain assignment, each run's % CRPS increase table, and
the mean and sd across runs. Rows: whose history is shuffled. Columns: outcome.
"""
import argparse

import pandas as pd

from alzheimers_trajectory_mapping.diagnostics import domain_history_columns, history_influence
from alzheimers_trajectory_mapping.evaluate import load_run

parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", required=True)
parser.add_argument("--split", default="val")
args = parser.parse_args()
pd.set_option("display.width", 200)

_, _, meta, _ = load_run(f"outputs/runs/{args.names[0]}")
assigned, unassigned = domain_history_columns(meta)
for D, idx in assigned.items():
    print(D, [meta["feature_columns"][i] for i in idx])
print("unassigned:", unassigned, "\n")

changes = []
for name in args.names:
    _, change = history_influence(f"outputs/runs/{name}", args.split)
    print(name, "\n", change.round(1).to_string(), "\n")
    changes.append(change.drop(index="intact"))
stack = pd.concat(changes, keys=range(len(changes)))
print("MEAN % CRPS increase\n", stack.groupby(level=1).mean().round(1).to_string())
print("\nSD\n", stack.groupby(level=1).std().round(1).to_string())