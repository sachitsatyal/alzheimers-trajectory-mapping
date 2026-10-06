"""Turn cleaned visits into padded tensors for the model.

Order of operations matters for leakage:
  1. split patients into train / validation / test, grouped by STUDYID;
  2. fit detection-limit fills and standardization on TRAINING visits only;
  3. apply them to every split, compute time inputs, pad to T_max.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .outcomes import OUTCOMES
from .transforms import CATEGORIES, CleanData


def split_patients(d: CleanData, fractions=(0.70, 0.15, 0.15), seed=42) -> pd.Series:
    """Assign each PATID to train / val / test.

    Whole STUDYID groups go to one split. Groups are shuffled within strata of
    baseline dementia status so all splits have a similar case mix.
    """
    first = d.keys.groupby("PATID").head(1).index
    base = pd.DataFrame({"PATID": d.keys.loc[first, "PATID"].to_numpy(),
                         "STUDYID": d.keys.loc[first, "STUDYID"].to_numpy(),
                         "DEMENTED": d.features.loc[first, "D1_DEMENTED"].to_numpy()
                         if "D1_DEMENTED" in d.features else np.nan})
    groups = base.groupby("STUDYID").agg(n=("PATID", "size"), demented=("DEMENTED", "max"))
    groups["stratum"] = groups["demented"].fillna(-1)

    rng = np.random.default_rng(seed)
    assignment = {}
    for _, g in groups.groupby("stratum"):
        ids = g.index.to_numpy().copy()
        rng.shuffle(ids)
        cum = g.loc[ids, "n"].cumsum() / g["n"].sum()
        cut1, cut2 = fractions[0], fractions[0] + fractions[1]
        for sid, c in zip(ids, cum):
            assignment[sid] = "train" if c <= cut1 else ("val" if c <= cut2 else "test")
    return base.set_index("PATID")["STUDYID"].map(assignment).rename("split")


class Standardizer:
    """Detection-limit fills and z-scoring, fit on training visits only."""

    def fit(self, features, below, above):
        self.columns = list(features.columns)
        self.mean = features.mean()
        std = features.std()
        self.std = std.where(std > 1e-8, 1.0).fillna(1.0)
        self.mean = self.mean.fillna(0.0)
        # Below the assay limit -> lowest training value; above -> highest.
        self.low_fill = features[below.columns].min() if len(below.columns) else pd.Series(dtype=float)
        self.high_fill = features[above.columns].max() if len(above.columns) else pd.Series(dtype=float)
        return self

    def transform(self, features, below, above):
        x = features.copy()
        for c in below.columns:
            x.loc[below[c] & x[c].isna(), c] = self.low_fill[c]
        for c in above.columns:
            x.loc[above[c] & x[c].isna(), c] = self.high_fill[c]
        mask = x.notna()
        z = ((x - self.mean) / self.std).fillna(0.0)  # missing -> training mean (0)
        return z[self.columns], mask[self.columns]

    def to_dict(self):
        return {"mean": self.mean.to_dict(), "std": self.std.to_dict(),
                "low_fill": self.low_fill.to_dict(), "high_fill": self.high_fill.to_dict()}


def static_indices(static: pd.DataFrame) -> tuple:
    """Integer codes for the embedding tables; 0 is reserved for missing."""
    levels, out = {}, pd.DataFrame(index=static.index)
    for c in static.columns:
        lv = sorted(static[c].dropna().unique().tolist())
        levels[c] = lv
        out[c] = static[c].map({v: i + 1 for i, v in enumerate(lv)}).fillna(0).astype(int)
    return out, levels


def time_inputs(keys: pd.DataFrame) -> pd.DataFrame:
    years = lambda td: td.dt.days / 365.25
    g = keys.groupby("PATID")["DATE"]
    return pd.DataFrame({
        "t_abs": years(keys["DATE"] - g.transform("min")),
        "dt": years(keys["DATE"] - g.shift()).fillna(0.0),
        "dt_next": years(g.shift(-1) - keys["DATE"]).fillna(0.0),  # 0 at the last visit
    }, index=keys.index)


def _pad(frame_values, patient, t_max, fill=0.0):
    """Stack a per-visit array into (patients, t_max, ...)."""
    pids = pd.unique(patient)
    position = patient.groupby(patient).cumcount().to_numpy()
    row = pd.Series(np.arange(len(pids)), index=pids)[patient].to_numpy()
    shape = (len(pids), t_max) + frame_values.shape[1:]
    out = np.full(shape, fill, dtype=frame_values.dtype)
    out[row, position] = frame_values
    return out, pids


def build(d: CleanData, seed=42, t_max=None) -> tuple:
    """Return {split: arrays} and metadata."""
    split_of = split_patients(d, seed=seed)
    visit_split = d.keys["PATID"].map(split_of)
    t_max = t_max or int(d.keys.groupby("PATID").size().max())

    train = visit_split == "train"
    std = Standardizer().fit(d.features[train], d.below_detection[train], d.above_detection[train])
    x, m = std.transform(d.features, d.below_detection, d.above_detection)
    times = time_inputs(d.keys)
    s_idx, s_levels = static_indices(d.static)

    names = list(OUTCOMES)
    lows = np.array([OUTCOMES[k][1] for k in names], dtype=float)
    y_raw = d.outcomes[names].to_numpy(dtype=float)
    mu = ~np.isnan(y_raw)
    y = np.where(mu, y_raw - lows, 0).astype(np.int64)  # shifted to 0..n_k

    arrays = {}
    for split in ["train", "val", "test"]:
        sel = (visit_split == split).to_numpy()
        patient = d.keys.loc[sel, "PATID"].reset_index(drop=True)
        X, pids = _pad(x.to_numpy(np.float32)[sel], patient, t_max)
        arrays[split] = {
            "X": X,
            "M": _pad(m.to_numpy(np.float32)[sel], patient, t_max)[0],
            "tau": _pad(times[["t_abs", "dt"]].to_numpy(np.float32)[sel], patient, t_max)[0],
            "dt_next": _pad(times["dt_next"].to_numpy(np.float32)[sel], patient, t_max)[0],
            "Y": _pad(y[sel], patient, t_max)[0],
            "MU": _pad(mu[sel].astype(np.float32), patient, t_max)[0],
            "PI": _pad(np.ones(sel.sum(), np.float32), patient, t_max)[0],
            "S": s_idx.loc[pids].to_numpy(np.int64),
            "PATID": np.asarray(pids),
            "STUDYID": d.keys.loc[sel].groupby("PATID")["STUDYID"].first().loc[pids].to_numpy(),
        }

    start, blocks = 0, {}
    for cat in CATEGORIES:
        n = len(d.columns_by_category[cat])
        blocks[cat] = [start, start + n]
        start += n
    meta = {
        "feature_columns": list(x.columns),
        "category_blocks": blocks,
        "static_columns": list(d.static.columns),
        "static_levels": {k: [float(v) for v in lv] for k, lv in s_levels.items()},
        "outcomes": {k: {"domain": OUTCOMES[k][0], "low": OUTCOMES[k][1], "high": OUTCOMES[k][2],
                         "n": OUTCOMES[k][2] - OUTCOMES[k][1]} for k in names},
        "t_max": t_max, "seed": seed,
        "n_patients": {s: int(len(a["PATID"])) for s, a in arrays.items()},
        "standardizer": std.to_dict(),
    }
    return arrays, meta


def save(arrays, meta, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, a in arrays.items():
        np.savez_compressed(out / f"{split}.npz", **a)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))