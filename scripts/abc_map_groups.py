"""ABC map, Phase 1: per subtype, random vs diagnosis-matched shuffles, repeated shuffles.

  uv run python scripts/consensus.py --names J3_s1 J3_s2 J3_s3 --split val     # once, if not done
  uv run python scripts/abc_map_groups.py --names J3_s1 J3_s2 J3_s3

Uses outputs/consensus_<split>.csv for each patient's consensus subtype and baseline
diagnosis. For every run: intact forecasts, then for each scheme (random = any patient,
matched = same baseline diagnosis) and each repeat, one domain's history is swapped
(whole visit sequence, one partner per patient) and CRPS is re-scored.
Results per run are saved as they finish (rerunning skips finished runs):
  outputs/abc_maps/<run>_<split>_<variant>.csv
Printed: mean % CRPS increase across runs (each run averaged over its repeats) and
the sd across runs; * = mean larger than 2 sd.
"""
import argparse
from pathlib import Path

import pandas as pd

from alzheimers_trajectory_mapping.abc_groups import abc_map_groups, summarize

parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--samples", type=int, default=32)
parser.add_argument("--variant", default="main", choices=["main", "no_cdr", "scores_only"])
parser.add_argument("--schemes", nargs="+", default=["random", "matched"], choices=["random", "matched"])
parser.add_argument("--absolute", action="store_true", help="also print absolute changes (CRPS points)")
args = parser.parse_args()
pd.set_option("display.width", 200)

cons = pd.read_csv(f"outputs/consensus_{args.split}.csv")
subtype_of = dict(zip(cons["PATID"], cons["subtype_name"]))
dx_of = dict(zip(cons["PATID"], cons["baseline_dx"]))
print("patients per subtype:", cons["subtype_name"].value_counts().to_dict())
print("patients per baseline diagnosis:", cons["baseline_dx"].value_counts().to_dict(), "\n")

out_dir = Path("outputs/abc_maps")
out_dir.mkdir(parents=True, exist_ok=True)
frames = []
for name in args.names:
    tag = "" if args.schemes == ["random", "matched"] else "_" + "_".join(args.schemes)
    path = out_dir / f"{name}_{args.split}_{args.variant}{tag}_v2.csv"
    if path.exists():
        print(f"{name}: already done, loading {path}")
        frames.append(pd.read_csv(path))
        continue
    print(f"{name}: running ({len(args.schemes) * args.repeats * 3} scrambled passes)...", flush=True)
    long = abc_map_groups(f"outputs/runs/{name}", args.split, subtype_of, dx_of, repeats=args.repeats,
                          variant=args.variant, n_samples=args.samples, schemes=tuple(args.schemes))
    long["run"] = name
    long.to_csv(path, index=False)
    frames.append(long)

long = pd.concat(frames, ignore_index=True)
outcomes = ["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]
tables = summarize(long, outcomes)
order = ["all", "stable", "steady_decliner", "behavior_escalating"]
for scheme in ("random", "matched"):
    for group in order:
        if (scheme, group) not in tables:
            continue
        mean, sd = tables[(scheme, group)]
        cell = mean.round(1).astype(str) + " ± " + sd.round(1).astype(str)
        star = (mean > 2 * sd).replace({True: "*", False: ""})
        n = long[(long.scheme == scheme) & (long.group == group)].groupby("outcome")["forecasts"].first()
        print(f"=== {scheme} shuffle | {group} | forecasts per outcome: {n.reindex(outcomes).to_dict()}")
        print((cell + star).to_string(), "\n")
if args.absolute and "abs_change" in long.columns:
    abs_tables = summarize(long, outcomes, "abs_change")
    base_tables = summarize(long, outcomes, "intact_crps")
    for scheme in args.schemes:
        for group in order:
            if (scheme, group) not in abs_tables:
                continue
            mean, sd = abs_tables[(scheme, group)]
            base = base_tables[(scheme, group)][0].iloc[0]
            cell = mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)
            star = (mean > 2 * sd).replace({True: "*", False: ""})
            print(f"=== ABSOLUTE change in CRPS points | {scheme} | {group}")
            print("intact CRPS:", base.round(3).to_dict())
            print((cell + star).to_string(), "\n")
summary_rows = []
for (scheme, group), (mean, sd) in tables.items():
    for h in mean.index:
        for o in mean.columns:
            summary_rows.append({"scheme": scheme, "group": group, "history": h, "outcome": o,
                                 "mean_pct": mean.loc[h, o], "sd_pct": sd.loc[h, o]})
pd.DataFrame(summary_rows).to_csv(out_dir / f"summary_{args.split}_{args.variant}{tag}.csv", index=False)
print(f"saved {out_dir}/summary_{args.split}_{args.variant}{tag}.csv")
