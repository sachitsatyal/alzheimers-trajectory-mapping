"""Check whether the latent state (and each block) is used by the forecasts.

  uv run python scripts/diagnose.py --name kl001
"""
import argparse

import pandas as pd

from alzheimers_trajectory_mapping.diagnostics import z_usage

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--split", default="val")
args = parser.parse_args()

crps, change = z_usage(f"outputs/runs/{args.name}", args.split)
pd.set_option("display.width", 200)
print("CRPS\n", crps.round(3).to_string())
print("\n% CRPS increase vs intact (higher = block is used)\n", change.round(1).to_string())