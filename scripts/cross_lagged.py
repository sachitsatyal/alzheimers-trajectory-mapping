"""Cross-lagged regression check of the ABC map (classical statistics, no deep learning).

  uv run python scripts/consensus.py --names J3_s1 J3_s2 J3_s3 --split train   # once, if not done
  uv run python scripts/cross_lagged.py                  # train split: all patients + each subtype
  uv run python scripts/cross_lagged.py --split val      # replication on held-out patients

Each cell: coefficient [95% CI] = points of WORSENING in the column's score by the
next visit per 1 SD WORSE level of the row's domain now, holding the other two domains,
the outcome's own level, age, baseline diagnosis and gap fixed.
Stars: * p < 0.05, ** p < 0.01, *** p < 0.001 (Bonferroni for 15 tests per table: p < 0.0033).
Saved: outputs/cross_lagged_<split>.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alzheimers_trajectory_mapping.cross_lagged import DOMAIN_LEVEL, OUTCOMES, cross_lagged_table, visit_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train")
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
pd.set_option("display.width", 220)

processed = Path(yaml.safe_load(open(args.config))["data"]["processed_dir"])
meta = json.loads((processed / "meta.json").read_text())
arrays = dict(np.load(processed / f"{args.split}.npz", allow_pickle=True))
cons_path = Path(f"outputs/consensus_{args.split}.csv")
cons = pd.read_csv(cons_path) if cons_path.exists() else None
dx_of = dict(zip(cons["PATID"], cons["baseline_dx"])) if cons is not None else None
subtype_of = dict(zip(cons["PATID"], cons["subtype_name"])) if cons is not None else {}
if cons is None:
    print(f"note: {cons_path} not found -> no diagnosis covariate and no per-subtype tables\n")

pairs = visit_pairs(arrays, meta, dx_of)
pairs["subtype"] = pairs["PATID"].map(subtype_of)
scale = {f"{v}_now": (pairs[f"{v}_now"].mean(), pairs[f"{v}_now"].std()) for v in DOMAIN_LEVEL.values()}
print(f"{args.split}: {len(pairs)} visit pairs (gap 0.5-2 y) from {pairs['PATID'].nunique()} patients")
print("predictor SDs (1 unit):", {k: round(float(scale[f"{v}_now"][1]), 2)
                                  for k, v in DOMAIN_LEVEL.items()}, "\n")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


groups = [("all", pairs)] + [(g, pairs[pairs["subtype"] == g])
                             for g in ["stable", "steady_decliner", "behavior_escalating"]
                             if (pairs["subtype"] == g).any()]
results = []
for label, sub in groups:
    tab = cross_lagged_table(sub, scale).assign(group=label)
    results.append(tab)
    cell = tab.apply(lambda r: f"{r.coef:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]{stars(r.p)}", axis=1)
    show = tab.assign(cell=cell).pivot(index="predictor", columns="outcome", values="cell")
    show = show.reindex(index=list(DOMAIN_LEVEL), columns=[o for o in OUTCOMES if o in show.columns])
    n = tab.groupby("outcome")["pairs"].first().reindex(show.columns).to_dict()
    print(f"=== {args.split} | {label} | pairs per outcome: {n}")
    print(show.to_string(), "\n")

out = pd.concat(results, ignore_index=True)
out.to_csv(f"outputs/cross_lagged_{args.split}.csv", index=False)
print(f"saved outputs/cross_lagged_{args.split}.csv")
