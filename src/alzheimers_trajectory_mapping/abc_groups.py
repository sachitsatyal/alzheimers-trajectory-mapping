"""ABC map, Phase 1: per subtype, diagnosis-matched shuffles, repeated shuffles.

Same idea as diagnostics.history_influence (scramble one domain's past inputs and
measure how much every forecast gets worse), with three improvements:

1. Shuffles are done over the WHOLE split (not within batches of 32), and can be
   restricted to patients with the same baseline diagnosis ("matched"), so the
   borrowed history comes from someone at a similar disease stage.
2. The shuffle is repeated (default 10 times) with different random pairings,
   giving error bars for the shuffle itself.
3. CRPS is kept per forecast with its patient, so the map can be computed for any
   group of patients (e.g. each consensus subtype) from the same runs.

% increase for a group = mean CRPS (scrambled) / mean CRPS (intact) - 1, over that
group's forecasts; the model and every other input stay unchanged.
"""

import numpy as np
import pandas as pd
import torch

from .data import TARCCSequences
from .diagnostics import DOMAIN_INPUT_VARIANTS, domain_history_columns
from .evaluate import load_run, model_pmfs, score_pmf


def split_tensors(processed_dir, split) -> tuple:
    """All patients of a split as tensors (val/test are small), plus their PATIDs."""
    ds = TARCCSequences(processed_dir, split)
    return ds.tensors, np.asarray(ds.patid)


def grouped_permutation(groups, rng) -> np.ndarray:
    """Random permutation of 0..N-1 that only pairs patients within the same group.
    groups: (N,) labels; use one label for everyone for an unrestricted shuffle."""
    groups = np.asarray(groups)
    perm = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = rng.permutation(idx)
    return perm


def permute_columns(tensors: dict, columns, perm) -> dict:
    """Copy of the tensors where these input columns (values and masks) of patient i
    are replaced by patient perm[i]'s, at every visit. Everything else unchanged."""
    out = dict(tensors)
    perm = torch.as_tensor(perm, dtype=torch.long)
    for key in ("X", "M"):
        x = tensors[key].clone()
        x[..., columns] = tensors[key][perm][..., columns]
        out[key] = x
    return out


@torch.no_grad()
def forecast_crps(model, tensors, device, n_samples=32, batch_size=64, seed=0) -> pd.DataFrame:
    """CRPS of every observed next-visit forecast: columns patient (row index), outcome, crps."""
    torch.manual_seed(seed)  # identical posterior noise across compared settings
    names = model.decoders.outcomes
    n = len(tensors["X"])
    rows = []
    for start in range(0, n, batch_size):
        batch = {k: v[start:start + batch_size].to(device) for k, v in tensors.items()}
        pmfs, y_next, mu_next = model_pmfs(model, batch, n_samples)
        for k, name in enumerate(names):
            sel = mu_next[..., k] > 0
            if not sel.any():
                continue
            patient = sel.nonzero()[:, 0].cpu().numpy() + start
            crps = score_pmf(pmfs[k][sel].cpu().numpy(), y_next[..., k][sel].long().cpu().numpy())["crps"]
            rows.append(pd.DataFrame({"patient": patient, "outcome": name, "crps": crps}))
    return pd.concat(rows, ignore_index=True)


def group_change(intact: pd.DataFrame, scrambled: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """% CRPS increase per (group, outcome). groups: patient row index -> group label.
    Rows of intact and scrambled are the same forecasts in the same order."""
    df = intact.assign(scr=scrambled["crps"].to_numpy(), group=groups.reindex(intact["patient"]).to_numpy())
    out = []
    for label, sub in [("all", df)] + list(df.dropna(subset=["group"]).groupby("group")):
        m = sub.groupby("outcome")[["crps", "scr"]].mean()
        out.append(pd.DataFrame({"group": label, "outcome": m.index,
                                 "pct": (m["scr"] / m["crps"] - 1.0) * 100,
                                 "intact_crps": m["crps"].to_numpy(),
                                 "abs_change": (m["scr"] - m["crps"]).to_numpy(),  # CRPS points
                                 "forecasts": sub.groupby("outcome").size().reindex(m.index).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def abc_map_groups(run_dir, split, subtype_of_patid: dict, dx_of_patid: dict, repeats=10,
                   variant="main", n_samples=32, device=None, schemes=("random", "matched")) -> pd.DataFrame:
    """Long table: scheme, repeat, history (domain shuffled), group, outcome, pct, forecasts."""
    model, cfg, meta, device = load_run(run_dir, device)
    tensors, patid = split_tensors(cfg["data"]["processed_dir"], split)
    columns, _ = domain_history_columns(meta, DOMAIN_INPUT_VARIANTS[variant])
    subtype = pd.Series([subtype_of_patid.get(p) for p in patid])
    dx = np.array([dx_of_patid.get(p, "unknown") for p in patid])
    intact = forecast_crps(model, tensors, device, n_samples)
    results = []
    for scheme in schemes:
        match = dx if scheme == "matched" else np.zeros(len(patid))
        for r in range(repeats):
            rng = np.random.default_rng(1000 + r)
            perm = grouped_permutation(match, rng)  # same pairing for every domain in this repeat
            for D, idx in columns.items():
                scrambled = forecast_crps(model, permute_columns(tensors, idx, perm), device, n_samples)
                results.append(group_change(intact, scrambled, subtype).assign(scheme=scheme, repeat=r, history=D))
    return pd.concat(results, ignore_index=True)


def summarize(long: pd.DataFrame, outcomes: list, value: str = "pct") -> dict:
    """{(scheme, group): (mean table, sd table)}: mean over runs of each run's repeat-average,
    sd across runs (the seed-to-seed spread). Rows = history domain, columns = outcome.
    value: "pct" (% CRPS increase), "abs_change" (CRPS points) or "intact_crps"."""
    per_run = long.groupby(["run", "scheme", "group", "history", "outcome"])[value].mean().reset_index()
    tables = {}
    for (scheme, group), sub in per_run.groupby(["scheme", "group"]):
        mean = sub.pivot_table(index="history", columns="outcome", values=value, aggfunc="mean")
        sd = sub.pivot_table(index="history", columns="outcome", values=value, aggfunc="std")
        cols = [o for o in outcomes if o in mean.columns]
        tables[(scheme, group)] = (mean[cols], sd[cols])
    return tables
