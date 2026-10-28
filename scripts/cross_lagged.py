"""Cross-lagged regression check of the ABC map (classical statistics, no deep learning).

  uv run python scripts/consensus.py --names J3_s1 J3_s2 J3_s3 --split train   # once, if not done
  uv run python scripts/cross_lagged.py                  # train split
  uv run python scripts/cross_lagged.py --split val      # replication on held-out patients

Four specifications are fitted (see cross_lagged.py):
  single            IADL / NPI-Q severity / MMSE as the domain levels
  single_within     same, split into within-person and between-person parts
  composite         function = IADL+PSMS, cognition = MMSE+neuropsych composite
  composite_within  composite measures, within/between split   <- primary for the ABC question
Coefficient = points of WORSENING in the outcome by the next visit per 1 SD WORSE domain
level now (within-person: per 1 SD worse than the patient's own average), holding the
other domains, the outcome's own level, age, baseline diagnosis and gap fixed.
Stars: * p < 0.05, ** p < 0.01, *** p < 0.001 (Bonferroni for 10 cross-domain links: p < 0.005).
Saved: outputs/cross_lagged_<split>.csv (all specifications, groups and parts)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alzheimers_trajectory_mapping.cross_lagged import (HISTORY_COVARIATES, OUTCOMES, SPECS, add_composites,
                                                        cross_domain, cross_lagged_table, predictor_scale,
                                                        visit_pairs)

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="train")
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)

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
pairs, tests = add_composites(pairs, arrays, meta)
pairs["subtype"] = pairs["PATID"].map(subtype_of)
print(f"{args.split}: {len(pairs)} visit pairs (gap 0.5-2 y) from {pairs['PATID'].nunique()} patients")
print("neuropsych tests in the cognition composite (sign -1 = flipped so higher = worse):")
print(tests.round(2).to_string(index=False))
print(f"pairs with cognition composite: {pairs['cognition_comp_now'].notna().sum()}, "
      f"with function composite: {pairs['function_comp_now'].notna().sum()}\n")


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def fmt(r):
    return f"{r.coef:+.2f}{stars(r.p)}"


groups = [("all", pairs)] + [(g, pairs[pairs["subtype"] == g])
                             for g in ["stable", "steady_decliner", "behavior_escalating"]
                             if (pairs["subtype"] == g).any()]
results = []
for spec_name, within in [("single", False), ("single_within", True), ("composite", False), ("composite_within", True)]:
    predictors = SPECS[spec_name.replace("_within", "")]
    scale = predictor_scale(pairs, predictors)
    for label, sub in groups:
        tab = cross_lagged_table(sub, scale, predictors, within)
        results.append(tab.assign(spec=spec_name, group=label))
out = pd.concat(results, ignore_index=True)
out.to_csv(f"outputs/cross_lagged_{args.split}.csv", index=False)
order = [f"{p} -> {o}" for o in OUTCOMES for p in ("function", "behavior", "cognition")]


def links(df):
    df = cross_domain(df).copy()
    df["link"] = df["predictor"] + " -> " + df["outcome"]
    return df


# 1) Agreement table: every cross-domain link, all patients, across specifications
x = links(out[(out.group == "all") & (out.part != "between")])
x["cell"] = x.apply(fmt, axis=1)
agree = x.pivot(index="link", columns="spec", values="cell")
agree = agree.reindex([l for l in order if l in agree.index])[["single", "single_within", "composite", "composite_within"]]
print(f"=== {args.split} | ALL PATIENTS | cross-domain links across specifications (level or within-person)")
print(agree.to_string(), "\n")

# 2) Between-person part for comparison (composite_within)
b = links(out[(out.group == "all") & (out.spec == "composite_within") & (out.part == "between")])
b["cell"] = b.apply(fmt, axis=1)
print("=== between-person part (composite_within): 'people who are generally worse'")
print(b.set_index("link")["cell"].reindex([l for l in order if l in b["link"].values]).to_string(), "\n")

# 3) Primary specification per subtype (within-person, cross-domain)
w = links(out[(out.spec == "composite_within") & (out.part == "within")])
w["cell"] = w.apply(lambda r: f"{r.coef:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]{stars(r.p)}", axis=1)
per_group = w.pivot(index="link", columns="group", values="cell")
cols = [g for g in ["all", "stable", "steady_decliner", "behavior_escalating"] if g in per_group.columns]
print("=== PRIMARY (composite measures, within-person) | cross-domain links per subtype")
print(per_group.reindex([l for l in order if l in per_group.index])[cols].to_string(), "\n")
n = out[(out.spec == "composite_within") & (out.group == "all")].groupby("outcome")["pairs"].first()
print("pairs per outcome (composite_within, all):", n.reindex(OUTCOMES).to_dict())
print(f"saved outputs/cross_lagged_{args.split}.csv\n")

# 4) History adjustment: add what the deep model already knows (each domain's previous slope,
#    years since entry). Same subset of pairs (3 consecutive visits) for every column.
H = pairs.dropna(subset=HISTORY_COVARIATES)
comp = SPECS["composite"]
scale_c = predictor_scale(pairs, comp)
print(f"=== HISTORY-ADJUSTED | {len(H)} pairs with a previous interval (0.5-2 y) "
      f"from {H['PATID'].nunique()} patients")
hist_runs = [("composite (all pairs)", pairs, False, None), ("composite (history subset)", H, False, None),
             ("composite + history", H, False, HISTORY_COVARIATES),
             ("within (history subset)", H, True, None), ("within + history", H, True, HISTORY_COVARIATES)]
hist_results = []
for label, data, within, extra in hist_runs:
    for g_label, g_data in [("all", data)] + [(g, data[data["subtype"] == g])
                                              for g in ["stable", "steady_decliner", "behavior_escalating"]
                                              if (data["subtype"] == g).any()]:
        tab = cross_lagged_table(g_data, scale_c, comp, within, extra).assign(spec=label, group=g_label)
        hist_results.append(tab)
hist = pd.concat(hist_results, ignore_index=True)
hist.to_csv(f"outputs/cross_lagged_{args.split}_history.csv", index=False)
h = links(hist[(hist.group == "all") & (hist.part != "between")])
h["cell"] = h.apply(fmt, axis=1)
tab_h = h.pivot(index="link", columns="spec", values="cell")
tab_h = tab_h.reindex([l for l in order if l in tab_h.index])[[r[0] for r in hist_runs]]
print(tab_h.to_string(), "\n")

# 5) Which history terms carry the information for MMSE change (composite + history)?
cov = hist[(hist.group == "all") & (hist.spec == "composite + history") & (hist.outcome == "MMSE")].copy()
cov["cell"] = cov.apply(lambda r: f"{r.coef:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]{stars(r.p)}", axis=1)
print("=== MMSE change, composite + history: all terms (covariates per 1 SD)")
print(cov.set_index(["part", "predictor"])["cell"].to_string(), "\n")

# 6) Per subtype, composite + history
s6 = links(hist[(hist.spec == "composite + history")])
s6["cell"] = s6.apply(fmt, axis=1)
per = s6.pivot(index="link", columns="group", values="cell")
cols6 = [g for g in ["all", "stable", "steady_decliner", "behavior_escalating"] if g in per.columns]
print("=== composite + history | cross-domain links per subtype")
print(per.reindex([l for l in order if l in per.index])[cols6].to_string(), "\n")
print(f"saved outputs/cross_lagged_{args.split}_history.csv")
