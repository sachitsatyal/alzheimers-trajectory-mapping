"""Aim 1: consensus subtypes across seeds, early identification, baseline profile.

  uv run python scripts/consensus.py --names J3_s1 J3_s2 J3_s3 --split train
  uv run python scripts/consensus.py --names J3_s1 J3_s2 J3_s3 --split val
Use train for describing subtypes (more patients) and val for early identification
(patients the models never trained on). Per-patient results: outputs/consensus_<split>.csv
"""
import argparse

import pandas as pd

from alzheimers_trajectory_mapping.consensus import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", required=True)
parser.add_argument("--split", default="val")
args = parser.parse_args()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

r = analyze([f"outputs/runs/{n}" for n in args.names], args.split)
r["patients"].to_csv(f"outputs/consensus_{args.split}.csv", index=False)

print(f"=== each run vs consensus ({args.split}) ===\n", r["agreement"].round(3).to_string())
print("\n=== consensus subtypes: size, confidence, progression per year ===")
print(r["progression"].join(r["confidence"].rename("confidence")).round(2).to_string())
print("\n=== progression per year WITHIN baseline diagnosis ===\n", r["within"].round(2).to_string())
print("\n=== baseline profile (what is known at enrollment) ===\n", r["baseline"].round(2).to_string())
print("\n=== early identification: subtype after k visits vs final (patients with >= 5 visits) ===")
print(r["online"].round(1).to_string())