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