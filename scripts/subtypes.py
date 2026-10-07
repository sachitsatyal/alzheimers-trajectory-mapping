"""Aim 1: describe the progression subtypes and their stability across seeds.

  uv run python scripts/subtypes.py --names ownS4_s1 ownS4_s2 ownS4_s3
Describes subtypes of the first run in detail (training patients) and reports
agreement of hard assignments between runs (adjusted Rand index, 1 = identical,
0 = chance; label order does not matter). Per-patient tables are saved to each
run folder as subtypes_<split>.csv (git-ignored).
"""
import argparse
import itertools

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from alzheimers_trajectory_mapping.subtype_analysis import analyze_run

parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", required=True)
parser.add_argument("--split", default="train")
args = parser.parse_args()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

assignments = {}
for i, name in enumerate(args.names):
    df, summary, within = analyze_run(f"outputs/runs/{name}", args.split)
    df.to_csv(f"outputs/runs/{name}/subtypes_{args.split}.csv", index=False)
    assignments[name] = df.set_index("PATID")["subtype"]
    if i == 0:
        print(f"=== {name}: subtype summary ({args.split}) ===")
        print(summary.round(2).T.to_string())
        print("\n=== annual change WITHIN baseline diagnosis ===")
        print(within.round(2).to_string())
    print(f"\n{name}: sizes", (df["subtype"].value_counts(normalize=True).sort_index().round(3)).to_dict(),
          "| mean confidence", round(df["confidence"].mean(), 3))

print("\n=== assignment agreement between runs (adjusted Rand index) ===")
for a, b in itertools.combinations(args.names, 2):
    common = assignments[a].index.intersection(assignments[b].index)
    print(f"{a} vs {b}: ARI = {adjusted_rand_score(assignments[a][common], assignments[b][common]):.3f}")