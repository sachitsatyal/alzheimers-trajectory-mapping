"""Aim 1, final step: consensus subtypes across seeds, how early a patient can be
assigned, and what at baseline predicts membership.

1. Each seed's subtype labels are arbitrary, so labels are matched to the first
   run by maximizing shared patients (Hungarian algorithm); matched subtype
   probabilities are averaged across runs -> consensus probabilities.
2. For J = 3, subtypes are named from their observed progression:
   stable (least MMSE decline), behavior-escalating (largest NPI-Q rise), steady decliner.
3. Online identification: the subtype assigned after the first k visits (online
   posterior; uses only those visits) vs the final consensus subtype.
4. Baseline profile per consensus subtype (what a clinician sees at enrollment).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .data import make_loader
from .evaluate import load_run
from .loss import next_visit_targets
from .subtype_analysis import patient_profiles
from .subtypes import SubtypeModel


@torch.no_grad()
def run_posteriors(run_dir, split="val", n_samples=32, device=None, seed=0) -> dict:
    """Full posterior (N, J) and online posterior by visit (N, T, J) for one run."""
    model, cfg, meta, device = load_run(run_dir, device)
    torch.manual_seed(seed)
    loader = make_loader(cfg["data"]["processed_dir"], split, cfg["training"]["batch_size"], shuffle=False)
    full, online = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, n_samples)
        y_next, mu_next, _ = next_visit_targets(batch)
        ll = model.decoders.log_likelihood(y_next, out["p"], out["kappa"]) * mu_next
        transition = ll.mean(1).sum(-1).permute(1, 2, 0)
        _, resp = SubtypeModel.patient_posterior(out["log_prior"], transition)
        full.append(resp.cpu().numpy())
        online.append(SubtypeModel.online_posterior(out["log_prior"], transition).cpu().numpy())
    return {"patid": np.asarray(loader.dataset.patid), "full": np.concatenate(full),
            "online": np.concatenate(online), "meta": meta, "cfg": cfg}


def match_labels(reference_hard, hard, J) -> np.ndarray:
    """Permutation p with p[reference_label] = this run's label, maximizing overlap."""
    overlap = np.zeros((J, J))
    np.add.at(overlap, (reference_hard, hard), 1)
    ref_idx, run_idx = linear_sum_assignment(-overlap)
    p = np.empty(J, dtype=int)
    p[ref_idx] = run_idx
    return p


def consensus(runs: list) -> dict:
    """Average label-matched probabilities across runs."""
    J = runs[0]["full"].shape[1]
    ref = runs[0]["full"].argmax(1)
    full, online, agreement = [], [], []
    for r in runs:
        p = match_labels(ref, r["full"].argmax(1), J)
        full.append(r["full"][:, p])
        online.append(r["online"][..., p])
    full, online = np.mean(full, 0), np.mean(online, 0)
    hard = full.argmax(1)
    for r in runs:
        p = match_labels(hard, r["full"].argmax(1), J)
        run_hard = r["full"][:, p].argmax(1)
        agreement.append({"ARI_vs_consensus": adjusted_rand_score(hard, run_hard),
                          "same_label_%": 100 * (run_hard == hard).mean()})
    return {"full": full, "online": online, "hard": hard, "confidence": full.max(1),
            "agreement": pd.DataFrame(agreement)}


def name_subtypes(df: pd.DataFrame, J: int) -> dict:
    """Names from observed progression (J = 3); generic names otherwise."""
    if J != 3:
        return {j: f"subtype_{j}" for j in range(J)}
    g = df.groupby("consensus")
    stable = g["MMSE_per_year"].mean().idxmax()
    rest = [j for j in range(J) if j != stable]
    behavior = max(rest, key=lambda j: g["NPIQ_SEV_per_year"].mean()[j])
    steady = [j for j in rest if j != behavior][0]
    return {stable: "stable", steady: "steady_decliner", behavior: "behavior_escalating"}


def online_identification(online, hard, n_visits, names, max_k=4) -> pd.DataFrame:
    """Agreement of the subtype assigned after k visits with the final consensus subtype,
    among patients followed for at least max_k + 1 visits (so later visits add information)."""
    keep = n_visits >= max_k + 1
    rows = []
    for k in range(1, max_k + 1):
        early = online[keep, k - 1].argmax(1)
        row = {"visits_seen": k, "patients": int(keep.sum()), "overall_%": 100 * (early == hard[keep]).mean()}
        for j, name in names.items():
            members = hard[keep] == j
            row[f"{name}_recall_%"] = 100 * (early[members] == j).mean() if members.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("visits_seen")


def analyze(run_dirs: list, split="val", device=None) -> dict:
    runs = [run_posteriors(r, split, device=device) for r in run_dirs]
    for r in runs[1:]:
        assert np.array_equal(r["patid"], runs[0]["patid"]), "runs must cover the same patients"
    c = consensus(runs)
    meta, cfg = runs[0]["meta"], runs[0]["cfg"]
    path = Path(cfg["data"]["processed_dir"]) / f"{split}.npz"
    prof = patient_profiles(meta, path)
    a = np.load(path, allow_pickle=True)
    si = meta["static_columns"].index("A1_SEX")
    sex_levels = [np.nan] + list(meta["static_levels"]["A1_SEX"])
    prof["male_%"] = [np.nan if s == 0 else 100.0 * (sex_levels[s] == 1) for s in a["S"][:, si]]  # 1 = male

    J = c["full"].shape[1]
    df = prof.assign(consensus=c["hard"], confidence=c["confidence"])
    names = name_subtypes(df, J)
    df["subtype_name"] = df["consensus"].map(names)
    for j in range(J):
        df[f"prob_{names[j]}"] = c["full"][:, j]

    outcomes = list(meta["outcomes"])
    g = df.groupby("subtype_name")
    progression = g[[f"{o}_per_year" for o in outcomes]].mean().join(g.size().rename("patients"))
    within = df.groupby(["baseline_dx", "subtype_name"])[[f"{o}_per_year" for o in outcomes]].mean()
    dx = pd.crosstab(df["subtype_name"], df["baseline_dx"], normalize="index") * 100
    base_cols = ["age", "male_%", "apoe_e4_carrier"] + [f"{o}_baseline" for o in outcomes]
    baseline = g[base_cols].mean().rename(columns={"apoe_e4_carrier": "apoe_e4_share"}).join(dx)
    early = online_identification(c["online"], c["hard"], a["PI"].sum(1).astype(int), names)
    return {"patients": df, "agreement": c["agreement"], "progression": progression,
            "within": within, "baseline": baseline, "online": early,
            "confidence": g["confidence"].mean()}