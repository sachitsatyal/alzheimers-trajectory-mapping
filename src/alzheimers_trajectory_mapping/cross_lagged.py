"""Cross-lagged regression: an independent, classical check of the ABC map.

For consecutive visit pairs (t -> t+1) and each outcome Y:

    change in Y (t -> t+1) = b_A * function_t + b_B * behavior_t + b_C * cognition_t
                             + own level of Y at t (if not already a predictor)
                             + age + baseline diagnosis + gap (years) + intercept

Domain levels at t: function = IADL, behavior = NPI-Q severity, cognition = MMSE.
Everything is oriented so that POSITIVE = WORSE: MMSE is negated for predictors and
for its change. Predictors are z-scored (1 unit = 1 SD over the analysis split), so
b_C for outcome IADL reads: "1 SD worse cognition now -> b_C more IADL points of
worsening by the next visit, holding function and behavior now fixed".

Estimation: ordinary least squares with standard errors clustered by patient
(each patient contributes several pairs), CR1 small-sample correction, t-test with
G - 1 degrees of freedom (G = number of patients).
Pairs: both outcome values observed, all three domain levels observed at t,
gap between 0.5 and 2.0 years.
"""

import numpy as np
import pandas as pd
from scipy import stats

OUTCOMES = ["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]
DOMAIN_LEVEL = {"function": "IADL", "behavior": "NPIQ_SEV", "cognition": "MMSE"}
HIGHER_IS_BETTER = {"MMSE"}


def visit_pairs(arrays: dict, meta: dict, dx_of_patid: dict = None, min_gap=0.5, max_gap=2.0) -> pd.DataFrame:
    """One row per consecutive visit pair: PATID, gap, age, dx, and for each outcome its
    level at t and next value at t+1 (original scale, NaN if unobserved), oriented so
    that higher = worse."""
    names = list(meta["outcomes"])
    Y, MU, PI = arrays["Y"], arrays["MU"], arrays["PI"]
    t_abs = arrays["tau"][..., 0]
    cols = meta["feature_columns"]
    st = meta["standardizer"]
    i_age = cols.index("AGE")
    age = arrays["X"][..., i_age] * st["std"]["AGE"] + st["mean"]["AGE"]
    age = np.where(arrays["M"][..., i_age] > 0, age, np.nan)

    N, T = PI.shape
    ii, tt = np.nonzero((PI[:, :-1] > 0) & (PI[:, 1:] > 0))
    df = pd.DataFrame({"PATID": np.asarray(arrays["PATID"])[ii], "gap": t_abs[ii, tt + 1] - t_abs[ii, tt],
                       "age": age[ii, tt]})
    for k, name in enumerate(names):
        low = meta["outcomes"][name]["low"]
        sign = -1.0 if name in HIGHER_IS_BETTER else 1.0
        now = np.where(MU[ii, tt, k] > 0, Y[ii, tt, k] + low, np.nan)
        nxt = np.where(MU[ii, tt + 1, k] > 0, Y[ii, tt + 1, k] + low, np.nan)
        df[f"{name}_now"] = sign * now
        df[f"{name}_change"] = sign * (nxt - now)
    df["dx"] = [dx_of_patid.get(p, "unknown") if dx_of_patid else "unknown" for p in df["PATID"]]
    return df[(df["gap"] >= min_gap) & (df["gap"] <= max_gap)].reset_index(drop=True)


def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> tuple:
    """OLS coefficients, cluster-robust (CR1) standard errors, p-values (t, G-1 df)."""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    _, g_idx = np.unique(groups, return_inverse=True)
    G = g_idx.max() + 1
    scores = np.zeros((G, p))
    np.add.at(scores, g_idx, X * resid[:, None])
    meat = scores.T @ scores
    c = (G / (G - 1)) * ((n - 1) / (n - p))
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tval = beta / np.where(se > 0, se, np.nan)
    pval = 2 * stats.t.sf(np.abs(tval), df=G - 1)
    return beta, se, pval, G


def fit_outcome(pairs: pd.DataFrame, outcome: str, scale: dict) -> tuple:
    """Fit one outcome. scale: {column: (mean, sd)} for z-scoring the domain levels.
    Returns (coefficient table for the three domain levels, n pairs, n patients)."""
    level_cols = [f"{v}_now" for v in DOMAIN_LEVEL.values()]
    own = f"{outcome}_now"
    need = level_cols + [f"{outcome}_change", own, "age", "gap"]
    d = pairs.dropna(subset=list(dict.fromkeys(need))).copy()
    if len(d) < 30 or d["PATID"].nunique() < 10:
        return None, len(d), d["PATID"].nunique()
    parts = {"intercept": np.ones(len(d))}
    for domain, var in DOMAIN_LEVEL.items():
        m, s = scale[f"{var}_now"]
        parts[domain] = (d[f"{var}_now"] - m) / s
    if own not in level_cols:  # control for the outcome's own level
        parts["own_level"] = d[own] - d[own].mean()
    parts["age_per10y"] = (d["age"] - 70.0) / 10.0
    parts["gap_years"] = d["gap"] - 1.0
    for level in ("MCI/other", "dementia"):
        if (d["dx"] == level).any() and d["dx"].nunique() > 1:
            parts[f"dx_{level}"] = (d["dx"] == level).astype(float)
    X = np.column_stack(list(parts.values()))
    beta, se, pval, G = cluster_ols(d[f"{outcome}_change"].to_numpy(float), X, d["PATID"].to_numpy())
    names = list(parts)
    tcrit = stats.t.ppf(0.975, df=G - 1)
    rows = []
    for domain in DOMAIN_LEVEL:
        j = names.index(domain)
        rows.append({"outcome": outcome, "predictor": domain, "coef": beta[j], "se": se[j],
                     "ci_low": beta[j] - tcrit * se[j], "ci_high": beta[j] + tcrit * se[j], "p": pval[j]})
    return pd.DataFrame(rows), len(d), G


def cross_lagged_table(pairs: pd.DataFrame, scale: dict = None) -> pd.DataFrame:
    """All outcomes; scale defaults to these pairs' own mean/sd (pass the full split's
    scale when fitting subgroups, so coefficients stay comparable)."""
    if scale is None:
        scale = {f"{v}_now": (pairs[f"{v}_now"].mean(), pairs[f"{v}_now"].std()) for v in DOMAIN_LEVEL.values()}
    out = []
    for outcome in OUTCOMES:
        tab, n, G = fit_outcome(pairs, outcome, scale)
        if tab is not None:
            out.append(tab.assign(pairs=n, patients=G))
    return pd.concat(out, ignore_index=True)
