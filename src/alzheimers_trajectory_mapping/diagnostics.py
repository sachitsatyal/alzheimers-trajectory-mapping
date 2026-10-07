"""Is the latent state actually USED? (and which blocks)

"Active units" only says the posterior differs from the prior; it does not say
the decoders rely on it. Here each latent block's samples are SHUFFLED across
patients (each patient gets another patient's block), which keeps realistic
values but removes the link to this patient's history. If CRPS barely changes,
that block is not used. Expected for a healthy coupled ABC model:
  shuffle all -> every outcome worse
  shuffle S   -> every outcome worse (shared severity)
  shuffle A/B/C -> mainly that domain's outcomes worse (plus coupled domains)
"""

import pandas as pd
import torch

from .data import make_loader
from .evaluate import load_run, model_scores

# Which ABC-history inputs belong to which domain (matched as name prefixes).
# CDR boxes split by content; CDR global mixes domains and stays unassigned.
DOMAIN_INPUT_VARIANTS = {}  # filled below

DOMAIN_INPUTS = {
    "A": ["OUT_PSMS", "OUT_IADL", "A1_INDEPEND", "C1_CDRCA", "C1_CDRHOB", "C1_CDRPER"],
    "B": ["OUT_NPIQ", "C1_GDS"],
    "C": ["OUT_MMSE", "C1_CDRMEM", "C1_CDROR", "C1_CDRJU", "C1_SS_", "C1_ANIMAL", "C1_CLOCK",
          "C1_CERAD", "C1_TRAIL", "C1_BOSTON", "C1_COWAT", "C1_TOOIMPAIRED", "X1_DECLINE"],
}
# Robustness variants of the assignment:
#   no_cdr       CDR boxes unassigned (they are clinician ratings that mix domains)
#   scores_only  only the five outcome scores themselves
DOMAIN_INPUT_VARIANTS = {
    "main": DOMAIN_INPUTS,
    "no_cdr": {D: [p for p in prefixes if not p.startswith("C1_CDR")] for D, prefixes in DOMAIN_INPUTS.items()},
    "scores_only": {"A": ["OUT_PSMS", "OUT_IADL"], "B": ["OUT_NPIQ"], "C": ["OUT_MMSE"]},
}



def shuffle_block(sl, generator_seed=0):
    def perturb(z0):  # z0 (S, B, T, dz): permute patients within the batch for this block
        g = torch.Generator(device="cpu").manual_seed(generator_seed)
        perm = torch.randperm(z0.shape[1], generator=g).to(z0.device)
        z = z0.clone()
        z[..., sl] = z0[:, perm][..., sl]
        return z
    return perturb


def z_usage(run_dir, split="val", n_samples=32, device=None) -> pd.DataFrame:
    """CRPS per outcome with z intact and with each latent block shuffled.
    Returns a table of CRPS and its % increase over intact."""
    model, cfg, meta, device = load_run(run_dir, device)
    loader = make_loader(cfg["data"]["processed_dir"], split, cfg["training"]["batch_size"])
    settings = {"intact": None, "shuffle_all": shuffle_block(slice(None))}
    for name, sl in model.dynamics.slices.items():
        settings[f"shuffle_{name}"] = shuffle_block(sl)
    rows = {}
    for label, perturb in settings.items():
        scores = model_scores(model, loader, device, n_samples, perturb)
        rows[label] = {k: v["CRPS"] for k, v in scores.items()}
    crps = pd.DataFrame(rows).T
    change = (crps / crps.loc["intact"] - 1.0) * 100
    return crps, change


def domain_history_columns(meta: dict, domain_inputs: dict = None) -> tuple:
    """Indices of each domain's ABC-history inputs, plus the unassigned column names."""
    domain_inputs = domain_inputs or DOMAIN_INPUTS
    a, b = meta["category_blocks"]["ABC_HISTORY"]
    columns = meta["feature_columns"]
    assigned, unassigned = {D: [] for D in domain_inputs}, []
    for i in range(a, b):
        name = columns[i].upper()
        owner = [D for D, prefixes in domain_inputs.items() if any(name.startswith(p) for p in prefixes)]
        if owner:
            assigned[owner[0]].append(i)
        else:
            unassigned.append(columns[i])
    return assigned, unassigned


def shuffle_inputs(indices, seed=0):
    """Give each patient another patient's history for these input columns (values and masks)."""
    def transform(batch):
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(batch["X"].shape[0], generator=g).to(batch["X"].device)
        out = dict(batch)
        for key in ("X", "M"):
            x = batch[key].clone()
            x[..., indices] = batch[key][perm][..., indices]
            out[key] = x
        return out
    return transform


def history_influence(run_dir, split="val", n_samples=32, device=None, variant="main"):
    """Granger-style ABC map: % CRPS increase of every outcome when one domain's
    PAST scores are shuffled across patients. Off-diagonal entries = how much a
    domain's history helps forecast the other domains (predictive influence)."""
    model, cfg, meta, device = load_run(run_dir, device)
    loader = make_loader(cfg["data"]["processed_dir"], split, cfg["training"]["batch_size"])
    columns, _ = domain_history_columns(meta, DOMAIN_INPUT_VARIANTS[variant])
    rows = {"intact": {k: v["CRPS"] for k, v in model_scores(model, loader, device, n_samples).items()}}
    for D, idx in columns.items():
        scores = model_scores(model, loader, device, n_samples, batch_transform=shuffle_inputs(idx))
        rows[f"shuffle_history_{D}"] = {k: v["CRPS"] for k, v in scores.items()}
    crps = pd.DataFrame(rows).T
    return crps, (crps / crps.loc["intact"] - 1.0) * 100