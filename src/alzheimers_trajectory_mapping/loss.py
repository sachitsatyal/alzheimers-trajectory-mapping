"""Piece 7b: the training objective (paper Sec. III-I, adapted to patient subtypes).

Per patient i:
  l_t(c) = mean over samples s of sum_k MU_{t+1,k} log BetaBin(y_{t+1,k} | subtype c, sample s)
  log p(Y_i) ~ T * logsumexp_c [ (log P(c | baseline) + sum_t l_t(c)) / T ]   (T = temperature)
Loss = [ -sum_i log p(Y_i) + beta * KL + lambda * R ] / (number of forecast positions)

  KL  closed form KL(q(z_t | H_t) || N(0, I)) at forecast positions; optional free bits
  R   mean squared velocity at the start state (Eq. 15)
  + coupling_penalty * group lasso on the ABC interaction map (keeps it sparse)
Positions without a next real visit, and unobserved outcomes, contribute nothing.
"""

import torch

from .subtypes import SubtypeModel


def next_visit_targets(batch):
    """Targets at position t are the outcomes of visit t+1. Returns Y, MU (B, T, K)
    and the forecast-position mask fp (B, T) = real visit t followed by a real visit."""
    Y, MU, PI = batch["Y"], batch["MU"], batch["PI"]
    pad = lambda x: torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    fp = PI * pad(PI)
    return pad(Y), pad(MU) * fp.unsqueeze(-1), fp


def compute_loss(model, batch, out, beta=1.0, velocity_penalty=1e-3, temperature=1.0, free_bits=0.0,
                 coupling_penalty=1e-3):
    y_next, mu_next, fp = next_visit_targets(batch)
    n_pos = fp.sum().clamp_min(1.0)

    # Likelihood per subtype and position: (J, S, B, T, K) -> (B, T, J)
    ll = model.decoders.log_likelihood(y_next, out["p"], out["kappa"]) * mu_next
    ll_k = ll.mean(1)                                   # average over samples: (J, B, T, K)
    transition = ll_k.sum(-1).permute(1, 2, 0)          # (B, T, J)
    log_marginal, resp = SubtypeModel.patient_posterior(out["log_prior"], transition, temperature)

    # KL to N(0, I), per latent dimension, at forecast positions.
    mu, logvar = out["mu"], out["logvar"]
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar) * fp.unsqueeze(-1)   # (B, T, dz)
    kl_mean_dim = kl_dim.sum((0, 1)) / n_pos
    if model.posterior.deterministic:
        kl_term = mu.new_zeros(())
    elif free_bits > 0:
        kl_term = torch.clamp(kl_mean_dim, min=free_bits).sum() * n_pos
    else:
        kl_term = kl_dim.sum()

    vel = model.dynamics.velocity_penalty(out["z0"], model.subtypes.embedding.weight)  # (J, S, B, T)
    r_term = (vel.mean((0, 1)) * fp).sum()

    coupling = model.dynamics.coupling_norms().sum() if model.dynamics.coupling else mu.new_zeros(())
    loss = (-log_marginal.sum() + beta * kl_term + velocity_penalty * r_term) / n_pos \
        + coupling_penalty * coupling

    with torch.no_grad():
        # Per-outcome negative log-likelihood, weighted by each patient's subtype probabilities.
        weighted = (ll_k * resp.T[:, :, None, None]).sum(0)          # (B, T, K)
        nll_k = -weighted.sum((0, 1)) / mu_next.sum((0, 1)).clamp_min(1.0)
        stats = {
            "loss": loss.item(),
            "nll": (-log_marginal.sum() / n_pos).item(),
            "kl_per_position": (kl_dim.sum() / n_pos).item(),
            "active_units": int((kl_mean_dim > 0.01).sum()),
            "velocity": (r_term / n_pos).item(),
            "subtype_sizes": SubtypeModel.subtype_sizes(resp).tolist(),
            # how far apart the subtypes take the same start state (0 = subtypes ignored)
            "subtype_divergence": out["z_delta"].std(0).mean().item(),
            "nll_per_outcome": dict(zip(model.decoders.outcomes, nll_k.tolist())),
            "forecast_positions": int(n_pos.item()),
            "coupling_total": float(coupling),
        }
    return loss, stats