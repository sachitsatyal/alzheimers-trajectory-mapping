"""Piece 9: evaluation of forecasts as distributions, against fair baselines.

Why: the training log's MAE uses the predictive MEAN. Scores like PSMS sit at
the floor most of the time (74%), so a mean is pulled off the floor even when
the model is right; LOCF predicts the exact value and wins MAE unfairly.
The correct point forecast for MAE is the predictive MEDIAN; RMSE uses the mean.

Metrics per outcome, on every observed next-visit score:
  MAE (median), RMSE (mean), NLL, CRPS, 90% interval coverage and width.
Baselines on exactly the same positions:
  LOCF        point forecast = last observed value (training median if none)
  LOCF-prob   last observed value + the empirical distribution of change
              (from training pairs), clipped to the score range; gives NLL/CRPS
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_meta, make_loader
from .decoders import beta_binomial_log_prob, last_observed
from .loss import next_visit_targets
from .model import ABCModel
from .subtypes import SubtypeModel


# ---------- scoring a discrete predictive distribution (numpy) ----------

def score_pmf(pmf: np.ndarray, y: np.ndarray) -> dict:
    """pmf (N, n+1) over 0..n; y (N,) observed integers. Per-observation metrics."""
    grid = np.arange(pmf.shape[1])
    cdf = np.cumsum(pmf, axis=1)
    quantile = lambda q: (cdf >= q - 1e-9).argmax(axis=1)
    median, lo, hi = quantile(0.5), quantile(0.05), quantile(0.95)
    mean = (pmf * grid).sum(1)
    prob_obs = pmf[np.arange(len(y)), y]
    step = (grid[None, :] >= y[:, None]).astype(float)
    return {"abs_err": np.abs(median - y), "sq_err": (mean - y) ** 2,
            "nll": -np.log(np.clip(prob_obs, 1e-12, None)),
            "crps": ((cdf - step) ** 2).sum(1),
            "covered": ((y >= lo) & (y <= hi)).astype(float), "width": (hi - lo).astype(float)}


def aggregate(scores: dict) -> dict:
    return {"MAE": scores["abs_err"].mean(), "RMSE": math.sqrt(scores["sq_err"].mean()),
            "NLL": scores["nll"].mean(), "CRPS": scores["crps"].mean(),
            "cover90": scores["covered"].mean(), "width90": scores["width"].mean()}


# ---------- baselines (numpy) ----------

def last_observed_np(Y, MU):
    """Last observed value up to and including each visit: value, has (B, T, K)."""
    B, T, K = Y.shape
    idx = np.where(MU > 0, np.arange(T)[None, :, None], -1)
    last = np.maximum.accumulate(idx, axis=1)
    has = last >= 0
    value = np.take_along_axis(Y, np.clip(last, 0, None), axis=1) * has
    return value, has


def change_distributions(train: dict, n: list) -> tuple:
    """Empirical change (next - last observed) per outcome over -n..n, and the
    marginal distribution of next-visit scores, from training forecast pairs."""
    value, has = last_observed_np(train["Y"], train["MU"])
    fp = train["PI"][:, :-1] * train["PI"][:, 1:]
    deltas, marginals = [], []
    for k, nk in enumerate(n):
        target = (train["MU"][:, 1:, k] * fp) > 0
        y1 = train["Y"][:, 1:, k][target]
        both = target & has[:, :-1, k]
        d = (train["Y"][:, 1:, k] - value[:, :-1, k])[both]
        counts = np.bincount(d + nk, minlength=2 * nk + 1).astype(float) + 0.5  # add-half smoothing
        deltas.append(counts / counts.sum())
        m = np.bincount(y1, minlength=nk + 1).astype(float) + 0.5
        marginals.append(m / m.sum())
    return deltas, marginals


def locf_pmf(anchor, has, delta_pmf, marginal_pmf, n):
    """Distribution of next score: anchor + change, clipped to 0..n; marginal if no anchor."""
    N = len(anchor)
    out = np.zeros((N, n + 1))
    shifts = np.arange(-n, n + 1)
    target = np.clip(anchor[:, None] + shifts[None, :], 0, n)
    rows = np.repeat(np.arange(N), len(shifts))
    np.add.at(out, (rows, target.ravel()), np.tile(delta_pmf, N))
    out[~has] = marginal_pmf
    return out


# ---------- the model's predictive distribution (torch) ----------

@torch.no_grad()
def model_pmfs(model, batch, n_samples, perturb=None):
    """Predictive distribution per outcome at every position: list of (B, T, n_k + 1),
    a mixture over posterior samples and subtypes (weighted by the ONLINE posterior)."""
    out = model(batch, n_samples, perturb)
    y_next, mu_next, _ = next_visit_targets(batch)
    ll = model.decoders.log_likelihood(y_next, out["p"], out["kappa"]) * mu_next
    transition = ll.mean(1).sum(-1).permute(1, 2, 0)                                # (B, T, J)
    log_w = SubtypeModel.online_posterior(out["log_prior"], transition).clamp_min(1e-12).log()
    log_w = log_w.permute(2, 0, 1).unsqueeze(-1)                                    # (J, B, T, 1)
    pmfs = []
    for k in range(len(model.decoders.outcomes)):
        n = model.decoders.n[k]
        grid = torch.arange(int(n.item()) + 1, device=n.device, dtype=out["p"].dtype)
        lp = beta_binomial_log_prob(grid, n, out["p"][..., k:k + 1], out["kappa"][..., k:k + 1])
        lp = torch.logsumexp(lp, dim=1) - math.log(lp.shape[1])                     # over samples
        lp = torch.logsumexp(lp + log_w, dim=0)                                     # over subtypes
        pmfs.append(lp.exp())
    return pmfs, y_next, mu_next


def load_run(run_dir, device=None):
    """Best checkpoint of a run -> (model in eval mode, config, meta, device)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    cfg, meta = ckpt["config"], ckpt["meta"]
    cfg["model"].setdefault("decoder_anchor", False)  # runs trained before the anchor existed
    model = ABCModel(meta, cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, meta, device


@torch.no_grad()
def model_scores(model, loader, device, n_samples=32, perturb=None, seed=0) -> dict:
    """{outcome: aggregated metrics} for the model on every observed next-visit score."""
    torch.manual_seed(seed)  # same posterior samples across compared settings
    names = model.decoders.outcomes
    pmf_all = {k: [] for k in names}
    y_all = {k: [] for k in names}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pmfs, y_next, mu_next = model_pmfs(model, batch, n_samples, perturb)
        for k, name in enumerate(names):
            sel = mu_next[..., k] > 0
            pmf_all[name].append(pmfs[k][sel].cpu().numpy())
            y_all[name].append(y_next[..., k][sel].long().cpu().numpy())
    return {name: aggregate(score_pmf(np.concatenate(pmf_all[name]), np.concatenate(y_all[name])))
            for name in names}


def evaluate_run(run_dir, split="val", n_samples=32, device=None) -> pd.DataFrame:
    """Score a trained run's best checkpoint and the baselines on one split."""
    model, cfg, meta, device = load_run(run_dir, device)
    data_dir = cfg["data"]["processed_dir"]
    names = list(meta["outcomes"])
    n = [int(meta["outcomes"][k]["n"]) for k in names]
    train = dict(np.load(Path(data_dir) / "train.npz"))
    deltas, marginals = change_distributions(train, n)

    collected = {k: {"pmf": [], "y": [], "anchor": [], "has": []} for k in names}
    for batch in make_loader(data_dir, split, cfg["training"]["batch_size"]):
        batch = {k: v.to(device) for k, v in batch.items()}
        pmfs, y_next, mu_next = model_pmfs(model, batch, n_samples)
        anchor, has, _ = last_observed(batch["Y"], batch["MU"], batch["tau"][..., 0])
        for k, name in enumerate(names):
            sel = mu_next[..., k] > 0
            collected[name]["pmf"].append(pmfs[k][sel].cpu().numpy())
            collected[name]["y"].append(y_next[..., k][sel].long().cpu().numpy())
            collected[name]["anchor"].append(anchor[..., k][sel].long().cpu().numpy())
            collected[name]["has"].append((has[..., k][sel] > 0).cpu().numpy())

    rows = []
    for k, name in enumerate(names):
        c = {key: np.concatenate(v) for key, v in collected[name].items()}
        model_scores = aggregate(score_pmf(c["pmf"], c["y"]))
        base = locf_pmf(c["anchor"], c["has"], deltas[k], marginals[k], n[k])
        base_scores = aggregate(score_pmf(base, c["y"]))
        fallback = int(np.searchsorted(np.cumsum(marginals[k]), 0.5))
        point = np.where(c["has"], c["anchor"], fallback)
        rows.append({"outcome": name, "n_obs": len(c["y"]),
                     **{f"model_{m}": v for m, v in model_scores.items()},
                     "LOCF_MAE": np.abs(point - c["y"]).mean(),
                     "LOCF_RMSE": math.sqrt(((point - c["y"]) ** 2).mean()),
                     **{f"LOCFprob_{m}": base_scores[m] for m in ("NLL", "CRPS", "cover90")}})
    return pd.DataFrame(rows).set_index("outcome")