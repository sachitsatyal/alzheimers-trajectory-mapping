"""Score a trained run's best checkpoint against the baselines.

  uv run python scripts/evaluate.py --name coupled_anchor
Use the validation split while developing; the test split is used once, at the end.
"""
import argparse

import pandas as pd

from alzheimers_trajectory_mapping.evaluate import evaluate_run

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--samples", type=int, default=32)
args = parser.parse_args()

table = evaluate_run(f"outputs/runs/{args.name}", args.split, args.samples)
pd.set_option("display.width", 200)
print(table.T.round(3).to_string())
table.to_csv(f"outputs/runs/{args.name}/eval_{args.split}.csv")