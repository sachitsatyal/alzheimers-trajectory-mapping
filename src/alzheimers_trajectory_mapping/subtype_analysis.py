"""Aim 1 analysis: who is in each progression subtype, and how do they progress?

For every patient:
  baseline membership   P(c | baseline visit)        (what a NEW patient would get)
  full posterior        P(c | all their transitions)  (exact Bayes over J subtypes)
  hard subtype          argmax of the full posterior, and its confidence
Then each subtype is described by baseline profile (diagnosis, age, APOE e4,
scores) and by observed progression (annualized change per outcome), overall
and WITHIN baseline diagnosis. The within-diagnosis view is the key test that
subtypes describe how patients progress, not just which stage they are in.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import make_loader
from .evaluate import load_run
from .loss import next_visit_targets
from .subtypes import SubtypeModel


@torch.no_grad()
def patient_subtypes(model, cfg, split="train", n_samples=32, device="cpu", seed=0) -> pd.DataFrame:
    torch.manual_seed(seed)
    loader = make_loader(cfg["data"]["processed_dir"], split, cfg["training"]["batch_size"], shuffle=False)
    resp_all, prior_all, ntrans_all = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, n_samples)
        y_next, mu_next, fp = next_visit_targets(batch)
        ll = model.decoders.log_likelihood(y_next, out["p"], out["kappa"]) * mu_next
        transition = ll.mean(1).sum(-1).permute(1, 2, 0)
        _, resp = SubtypeModel.patient_posterior(out["log_prior"], transition)
        resp_all.append(resp.cpu().numpy())
        prior_all.append(out["log_prior"].exp().cpu().numpy())
        ntrans_all.append(fp.sum(1).cpu().numpy())
    resp, prior = np.concatenate(resp_all), np.concatenate(prior_all)
    J = resp.shape[1]
    df = pd.DataFrame({"PATID": loader.dataset.patid, "n_transitions": np.concatenate(ntrans_all).astype(int),
                       "subtype": resp.argmax(1), "confidence": resp.max(1),
                       "baseline_subtype": prior.argmax(1)})
    for j in range(J):
        df[f"post_{j}"] = resp[:, j]
        df[f"base_{j}"] = prior[:, j]
    return df


def patient_profiles(meta, split_path) -> pd.DataFrame:
    """Baseline profile and annualized change per outcome, from the saved tensors."""
    a = np.load(split_path, allow_pickle=True)
    cols = meta["feature_columns"]
    st = meta["standardizer"]

    def baseline(name):
        if name not in cols:
            return np.full(len(a["PATID"]), np.nan)
        i = cols.index(name)
        x = a["X"][:, 0, i] * st["std"][name] + st["mean"][name]
        return np.where(a["M"][:, 0, i] > 0, x, np.nan)

    demented, normal = baseline("D1_DEMENTED"), baseline("D1_NORMCOG")
    dx = np.where(demented == 1, "dementia", np.where(normal == 1, "normal", "MCI/other"))
    prof = pd.DataFrame({"PATID": a["PATID"], "baseline_dx": dx, "age": baseline("AGE")})

    si = meta["static_columns"].index("APOE_E4_COUNT")
    levels = [np.nan] + list(meta["static_levels"]["APOE_E4_COUNT"])
    prof["apoe_e4_carrier"] = [np.nan if s == 0 else float(levels[s] > 0) for s in a["S"][:, si]]

    t_abs = a["tau"][..., 0]
    for k, (name, spec) in enumerate(meta["outcomes"].items()):
        y = a["Y"][..., k] + spec["low"]
        obs = a["MU"][..., k] > 0
        first = np.where(obs.any(1), obs.argmax(1), -1)
        last = np.where(obs.any(1), obs.shape[1] - 1 - obs[:, ::-1].argmax(1), -1)
        rows = np.arange(len(y))
        y0 = np.where(first >= 0, y[rows, first], np.nan)
        span = t_abs[rows, last] - t_abs[rows, first]
        slope = (y[rows, last] - y[rows, first]) / np.where(span > 0.5, span, np.nan)
        prof[f"{name}_baseline"] = y0
        prof[f"{name}_per_year"] = np.where(first >= 0, slope, np.nan)
    return prof


def describe(df: pd.DataFrame, outcomes: list) -> tuple:
    """Per-subtype summary, and annual change within baseline diagnosis."""
    d = df.assign(baseline_agrees=(df["baseline_subtype"] == df["subtype"]).astype(float),
                  **{f"{k}_%": 100.0 * (df["baseline_dx"] == v)
                     for k, v in [("dementia", "dementia"), ("MCI", "MCI/other"), ("normal", "normal")]},
                  apoe_e4_pct=100.0 * df["apoe_e4_carrier"])
    g = d.groupby("subtype")
    summary = pd.DataFrame({"patients": g.size(), "share": g.size() / len(d)})
    cols = ["confidence", "baseline_agrees", "dementia_%", "MCI_%", "normal_%", "age", "apoe_e4_pct"]
    summary = summary.join(g[cols].mean())
    summary = summary.join(g[[f"{o}_baseline" for o in outcomes]].mean())
    summary = summary.join(g[[f"{o}_per_year" for o in outcomes]].mean())
    within = d.groupby(["baseline_dx", "subtype"])[[f"{o}_per_year" for o in outcomes]].mean()
    within["patients"] = d.groupby(["baseline_dx", "subtype"]).size()
    return summary, within


def analyze_run(run_dir, split="train", device=None) -> tuple:
    model, cfg, meta, device = load_run(run_dir, device)
    subtypes = patient_subtypes(model, cfg, split, device=device)
    prof = patient_profiles(meta, Path(cfg["data"]["processed_dir"]) / f"{split}.npz")
    df = subtypes.merge(prof, on="PATID")
    return df, *describe(df, list(meta["outcomes"]))