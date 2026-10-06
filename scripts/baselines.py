"""Simple baselines: last value carried forward (LOCF) and training mean."""
import json
import sys

import numpy as np

split = sys.argv[1] if len(sys.argv) > 1 else "val"
meta = json.load(open("data/processed/meta.json"))
names = list(meta["outcomes"])


def pairs(path):
    d = np.load(path)
    fp = d["PI"][:, :-1] * d["PI"][:, 1:]
    m0 = d["MU"][:, :-1] * fp[..., None]
    m1 = d["MU"][:, 1:] * fp[..., None]
    return d["Y"][:, :-1], d["Y"][:, 1:], m0, m1


y0, y1, m0, m1 = pairs(f"data/processed/{split}.npz")
_, ty1, _, tm1 = pairs("data/processed/train.npz")
train_mean = (ty1 * tm1).sum((0, 1)) / tm1.sum((0, 1))
for k, name in enumerate(names):
    both = (m0[..., k] * m1[..., k]) == 1
    target = m1[..., k] == 1
    locf = np.abs(y0[..., k] - y1[..., k])[both].mean()
    mean_mae = np.abs(train_mean[k] - y1[..., k])[target].mean()
    print(f"{name:<11} LOCF={locf:.2f} (n={int(both.sum())})   train-mean={mean_mae:.2f}")