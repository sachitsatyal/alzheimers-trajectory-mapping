"""Cross-lagged regression: an independent, classical check of the ABC map.

For consecutive visit pairs (t -> t+1) and each outcome Y:

    change in Y (t -> t+1) = b_A * function_t + b_B * behavior_t + b_C * cognition_t
                             + own level of Y at t (if Y is not itself a predictor)
                             + age + baseline diagnosis + gap (years) + intercept

Domain measures at t (two specifications):
  single     function = IADL, behavior = NPI-Q severity, cognition = MMSE
  composite  function = mean z of IADL and PSMS; behavior = NPI-Q severity;
             cognition = mean z of MMSE and a neuropsych composite (mean of the available
             oriented neuropsych test z-scores, at least 2 tests)
Everything is oriented so that POSITIVE = WORSE (MMSE negated; each neuropsych test is
flipped if it correlates negatively with worse MMSE). Predictors are z-scored over the
analysis split, so b_C for outcome IADL reads: "1 SD worse cognition now -> b_C more IADL
points of worsening by the next visit, holding function and behavior fixed".

Within/between option: each predictor is split into the patient's own average over their
pairs (between-person: "people who are generally worse") and the deviation from that
average at this visit (within-person: "when THIS person is worse than usual"). The
within-person coefficients answer the ABC question for an individual. Caveat: the patient
average uses all of that patient's visits.

History adjustment (extra covariates): each domain's previous slope (points per year from
visit t-1 to t: IADL, PSMS, NPI-Q severity, MMSE, neuropsych composite) and years since
entry. Requires three consecutive visits with gaps of 0.5-2 years.

Estimation: OLS with standard errors clustered by patient (CR1), t-test with G - 1 df.
Pairs: both outcome values observed, all predictors observed at t, gap 0.5-2.0 years.
"""

import numpy as np
import pandas as pd
from scipy import stats

OUTCOMES = ["PSMS", "IADL", "NPIQ_SEV", "NPIQ_COUNT", "MMSE"]
DOMAIN_LEVEL = {"function": "IADL", "behavior": "NPIQ_SEV", "cognition": "MMSE"}
HIGHER_IS_BETTER = {"MMSE"}
NEURO_PREFIXES = ["C1_SS_", "C1_ANIMAL", "C1_CLOCK", "C1_CERAD", "C1_TRAIL", "C1_BOSTON", "C1_COWAT"]

SPECS = {
    "single": {"function": "IADL_now", "behavior": "NPIQ_SEV_now", "cognition": "MMSE_now"},
    "composite": {"function": "function_comp_now", "behavior": "NPIQ_SEV_now", "cognition": "cognition_comp_now"},
}


def visit_pairs(arrays: dict, meta: dict, dx_of_patid: dict = None, min_gap=0.5, max_gap=2.0) -> pd.DataFrame:
    """One row per consecutive visit pair: PATID, row/t (tensor position of visit t), gap,
    age, dx, and per outcome its level at t and change to t+1 (original scale, NaN if
    unobserved), oriented so that higher = worse."""
    names = list(meta["outcomes"])
    Y, MU, PI = arrays["Y"], arrays["MU"], arrays["PI"]
    t_abs = arrays["tau"][..., 0]
    cols = meta["feature_columns"]
    st = meta["standardizer"]
    i_age = cols.index("AGE")
    age = arrays["X"][..., i_age] * st["std"]["AGE"] + st["mean"]["AGE"]
    age = np.where(arrays["M"][..., i_age] > 0, age, np.nan)

    ii, tt = np.nonzero((PI[:, :-1] > 0) & (PI[:, 1:] > 0))
    tp = np.maximum(tt - 1, 0)
    has_prev = tt >= 1
    prev_gap = np.where(has_prev, t_abs[ii, tt] - t_abs[ii, tp], np.nan)
    df = pd.DataFrame({"PATID": np.asarray(arrays["PATID"])[ii], "row": ii, "t": tt,
                       "gap": t_abs[ii, tt + 1] - t_abs[ii, tt], "age": age[ii, tt],
                       "years_since_entry": t_abs[ii, tt] - t_abs[ii, 0], "prev_gap": prev_gap})
    prev_ok = has_prev & (prev_gap >= min_gap) & (prev_gap <= max_gap)
    for k, name in enumerate(names):
        low = meta["outcomes"][name]["low"]
        sign = -1.0 if name in HIGHER_IS_BETTER else 1.0
        now = np.where(MU[ii, tt, k] > 0, Y[ii, tt, k] + low, np.nan)
        nxt = np.where(MU[ii, tt + 1, k] > 0, Y[ii, tt + 1, k] + low, np.nan)
        prv = np.where(prev_ok & (MU[ii, tp, k] > 0), Y[ii, tp, k] + low, np.nan)
        df[f"{name}_now"] = sign * now
        df[f"{name}_change"] = sign * (nxt - now)
        df[f"{name}_prev_slope"] = sign * (now - prv) / prev_gap  # points per year, previous interval
    df["dx"] = [dx_of_patid.get(p, "unknown") if dx_of_patid else "unknown" for p in df["PATID"]]
    return df[(df["gap"] >= min_gap) & (df["gap"] <= max_gap)].reset_index(drop=True)


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def neuropsych_columns(meta: dict) -> list:
    """Indices of neuropsych test inputs (ABC-history block, matched by name prefix)."""
    a, b = meta["category_blocks"]["ABC_HISTORY"]
    cols = meta["feature_columns"]
    return [i for i in range(a, b) if any(cols[i].upper().startswith(p) for p in NEURO_PREFIXES)]


def add_composites(pairs: pd.DataFrame, arrays: dict, meta: dict, min_tests: int = 2) -> tuple:
    """Adds function_comp_now and cognition_comp_now. Returns (pairs, table of the
    neuropsych tests used with their orientation sign and coverage)."""
    pairs = pairs.copy()
    idx = neuropsych_columns(meta)
    cols = meta["feature_columns"]
    rows, ts = pairs["row"].to_numpy(), pairs["t"].to_numpy()

    def tests_at(t_index):
        v = arrays["X"][rows, t_index][:, idx].astype(float)
        return np.where(arrays["M"][rows, t_index][:, idx] > 0, v, np.nan)

    x = tests_at(ts)
    worse_mmse = pairs["MMSE_now"].to_numpy(float)
    signs, info = np.ones(len(idx)), []
    for j, i in enumerate(idx):
        ok = ~np.isnan(x[:, j]) & ~np.isnan(worse_mmse)
        r = np.corrcoef(x[ok, j], worse_mmse[ok])[0, 1] if ok.sum() > 30 else np.nan
        signs[j] = -1.0 if (not np.isnan(r) and r < 0) else 1.0
        info.append({"test": cols[i], "corr_with_worse_MMSE": r, "sign": signs[j], "coverage": ok.mean()})
    def composite(values):
        oriented = values * signs
        n = (~np.isnan(oriented)).sum(1)
        return np.where(n >= min_tests, np.nansum(oriented, axis=1) / np.maximum(n, 1), np.nan), n

    np_mean, n_tests = composite(x)
    if "prev_gap" in pairs:
        prev_mean, _ = composite(tests_at(np.maximum(ts - 1, 0)))
        prev_mean = np.where(ts >= 1, prev_mean, np.nan)
        pairs["NEURO_prev_slope"] = (np_mean - prev_mean) / pairs["prev_gap"].to_numpy(float)
    pairs["NEURO_now"] = np_mean
    np_series = pd.Series(np_mean, index=pairs.index)
    cog = pd.concat([_z(pairs["MMSE_now"]), _z(np_series)], axis=1)
    pairs["cognition_comp_now"] = cog.mean(axis=1).where(cog.notna().all(axis=1))
    fun = pd.concat([_z(pairs["IADL_now"]), _z(pairs["PSMS_now"])], axis=1)
    pairs["function_comp_now"] = fun.mean(axis=1).where(fun.notna().all(axis=1))
    pairs["neuropsych_tests"] = n_tests
    return pairs, pd.DataFrame(info)


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


def fit_outcome(pairs: pd.DataFrame, outcome: str, scale: dict, predictors: dict = None,
                within: bool = False, extra: list = None) -> tuple:
    """Fit one outcome. predictors: {domain: column}; scale: {column: (mean, sd)}.
    Returns (coefficient table, n pairs, n patients). With within=True the table has
    part = "within" and part = "between" rows; otherwise part = "level"."""
    predictors = predictors or SPECS["single"]
    own = f"{outcome}_now"
    extra = extra or []
    need = list(predictors.values()) + [f"{outcome}_change", own, "age", "gap"] + list(extra)
    d = pairs.dropna(subset=list(dict.fromkeys(need))).copy()
    if len(d) < 30 or d["PATID"].nunique() < 10:
        return None, len(d), d["PATID"].nunique()
    parts = {"intercept": np.ones(len(d))}
    for domain, col in predictors.items():
        m, s = scale[col]
        z = (d[col] - m) / s
        if within:
            mean_i = z.groupby(d["PATID"]).transform("mean")
            parts[f"{domain}|within"] = z - mean_i
            parts[f"{domain}|between"] = mean_i
        else:
            parts[f"{domain}|level"] = z
    if own not in predictors.values():  # control for the outcome's own level
        parts["own_level"] = d[own] - d[own].mean()
    for col in extra:  # history covariates (z-scored), e.g. each domain's previous slope
        parts[f"x_{col}"] = (d[col] - d[col].mean()) / (d[col].std() or 1.0)
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
    for j, name in enumerate(names):
        if name.startswith("x_"):
            domain, part = name[2:], "covariate"
        elif "|" in name:
            domain, part = name.split("|")
        else:
            continue
        rows.append({"outcome": outcome, "predictor": domain, "part": part, "coef": beta[j], "se": se[j],
                     "ci_low": beta[j] - tcrit * se[j], "ci_high": beta[j] + tcrit * se[j], "p": pval[j]})
    return pd.DataFrame(rows), len(d), G


def predictor_scale(pairs: pd.DataFrame, predictors: dict) -> dict:
    return {c: (pairs[c].mean(), pairs[c].std()) for c in predictors.values()}


def cross_lagged_table(pairs: pd.DataFrame, scale: dict = None, predictors: dict = None,
                       within: bool = False, extra: list = None) -> pd.DataFrame:
    """All outcomes. scale defaults to these pairs' own mean/sd (pass the full split's scale
    when fitting subgroups, so coefficients stay comparable)."""
    predictors = predictors or SPECS["single"]
    scale = scale or predictor_scale(pairs, predictors)
    out = []
    for outcome in OUTCOMES:
        tab, n, G = fit_outcome(pairs, outcome, scale, predictors, within, extra)
        if tab is not None:
            out.append(tab.assign(pairs=n, patients=G))
    return pd.concat(out, ignore_index=True)


# What the deep model already knows beyond current levels: each domain's recent trend and time in study.
HISTORY_COVARIATES = ["IADL_prev_slope", "PSMS_prev_slope", "NPIQ_SEV_prev_slope", "MMSE_prev_slope",
                      "NEURO_prev_slope", "years_since_entry"]

# Cross-domain links (predictor domain -> outcome); own-domain cells are not ABC links.
OUTCOME_DOMAIN = {"PSMS": "function", "IADL": "function", "NPIQ_SEV": "behavior",
                  "NPIQ_COUNT": "behavior", "MMSE": "cognition"}


def cross_domain(tab: pd.DataFrame) -> pd.DataFrame:
    keep = (tab["predictor"] != tab["outcome"].map(OUTCOME_DOMAIN)) & (tab["part"] != "covariate")
    return tab[keep]
