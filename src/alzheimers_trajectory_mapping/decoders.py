"""Piece 6 (coupled ABC version): domain decoders.

For each domain D in {A, B, C}, the decoder reads only its own latent block
plus the shared block, u = [z_D; z_S] (evolved to the next visit):
  q^D = W_Q^D u                               query from the LATENT state
  alpha^D = softmax(E_t q^D / sqrt(d))        category attention (Aim 2)
  c^D = sum_g alpha_g^D e^(g)
  [logit p, kappa~]_k = MLP^D([u; c^D])        -> beta-binomial per outcome

Why the query comes from z and not from the history h~: in the first full
run (base_v1) the latent state collapsed (0 active units). With an h~-based
query, attention weights are a side channel: the decoder can read the patient's
history through *how* it mixes tokens, bypassing z. A z-based query closes it,
so everything the decoder knows about the patient's past passes through z.

Anchor (decoder_anchor, default true): each domain's decoder also receives its
own outcomes' last observed value (up to and including visit t), whether one
exists, and how many years old it is. The forecast is parameterized as a change
from that value:  logit p = logit((y_last + 0.5) / (n + 1)) + MLP output.
So the model starts at "last value carried forward" and z only needs to carry
the CHANGE (the progression), which is what the dynamics, subtypes and ABC
coupling describe. Each decoder sees only its own domain's scores, so any
cross-domain influence still has to travel through the coupled latent dynamics.
Motivation: without the anchor, all models lost to LOCF on function and
behavior, even with no KL penalty (z is too narrow a path to carry exact scores).

Switches: decoder_exclude_categories (default [ABC_HISTORY]);
decoder_uses_history (default false; true re-opens a direct path from h~);
decoder_anchor (default true).
"""

import torch
from torch import nn

from .dynamics import SHARED
from .encoder import TokenAttention


def beta_binomial_log_prob(y, n, p, kappa):
    """log BetaBin(y | n, a = p * kappa, b = (1 - p) * kappa)   (Eq. 12)."""
    a, b = p * kappa, (1 - p) * kappa
    lbeta = lambda x, w: torch.lgamma(x) + torch.lgamma(w) - torch.lgamma(x + w)
    log_choose = torch.lgamma(n + 1) - torch.lgamma(y + 1) - torch.lgamma(n - y + 1)
    return log_choose + lbeta(y + a, n - y + b) - lbeta(a, b)


def last_observed(Y, MU, t_abs):
    """Last observed value of each outcome up to and including each visit.

    Y, MU (B, T, K); t_abs (B, T). Returns value (B, T, K) float, has (B, T, K)
    0/1, and age (B, T, K) = years since that observation (0 if none)."""
    B, T, K = Y.shape
    steps = torch.arange(T, device=Y.device).view(1, T, 1).expand(B, T, K)
    idx = torch.where(MU > 0, steps, torch.full_like(steps, -1))
    last = torch.cummax(idx, dim=1).values
    has = (last >= 0).to(t_abs.dtype)
    safe = last.clamp_min(0)
    value = Y.gather(1, safe).to(t_abs.dtype) * has
    when = t_abs.unsqueeze(-1).expand(B, T, K).gather(1, safe)
    age = (t_abs.unsqueeze(-1) - when) * has
    return value, has, age


class DomainDecoders(nn.Module):
    def __init__(self, meta: dict, categories: list, cfg: dict, blocks: dict):
        super().__init__()
        d = cfg["d"]
        self.outcomes = list(meta["outcomes"])
        self.domains = sorted({meta["outcomes"][k]["domain"] for k in self.outcomes})
        self.index = {D: [i for i, k in enumerate(self.outcomes) if meta["outcomes"][k]["domain"] == D]
                      for D in self.domains}
        self.register_buffer("n", torch.tensor([float(meta["outcomes"][k]["n"]) for k in self.outcomes]))

        start, self.slices = 0, {}
        for k, size in blocks.items():
            self.slices[k] = slice(start, start + size)
            start += size

        excluded = set(cfg.get("decoder_exclude_categories", ["ABC_HISTORY"]))
        self.context_categories = [c for c in categories if c not in excluded]
        assert self.context_categories, "decoders need at least one category to attend to"
        self.register_buffer("context_idx", torch.tensor(
            [i for i, c in enumerate(categories) if c not in excluded], dtype=torch.long))
        self.uses_history = cfg.get("decoder_uses_history", False)
        self.extra_dim = cfg.get("decoder_extra_dim", 0)
        self.kappa_min = cfg["kappa_min"]
        self.anchor = cfg.get("decoder_anchor", True)

        self.attention = TokenAttention()
        self.queries, self.heads = nn.ModuleDict(), nn.ModuleDict()
        for D in self.domains:
            u_dim = blocks[D] + blocks[SHARED]
            in_dim = (u_dim + d + (d if self.uses_history else 0) + self.extra_dim
                      + (3 * len(self.index[D]) if self.anchor else 0))
            self.queries[D] = nn.Linear(u_dim, d)
            self.heads[D] = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg["decoder_hidden"]), nn.GELU(),
                nn.Dropout(cfg["decoder_dropout"]), nn.Linear(cfg["decoder_hidden"], 2 * len(self.index[D])))

    def latent_input(self, z, D):
        return torch.cat([z[..., self.slices[D]], z[..., self.slices[SHARED]]], dim=-1)

    def forward(self, z_delta, tokens, h_tilde, extra=None, anchor=None):
        """z_delta (J, S, B, T, dz); tokens (B, T, G, d); h_tilde (B, T, d);
        anchor = (value, has, age) from last_observed(), required if decoder_anchor.
        Returns p, kappa (J, S, B, T, K) and attention {D: (B, T, G')} averaged
        over subtypes and samples."""
        lead = z_delta.shape[:-1]
        if self.anchor:
            value, has, age = anchor
            frac = (value + 0.5) / (self.n + 1.0)
            base_logit = torch.log(frac) - torch.log1p(-frac)      # (B, T, K)
            anchor_feats = torch.stack([frac, has, age], dim=-1)   # (B, T, K, 3)
        ctx_tokens = tokens.index_select(-2, self.context_idx)
        p = z_delta.new_zeros(lead + (len(self.outcomes),))
        kappa = torch.zeros_like(p)
        attention = {}
        for D in self.domains:
            u = self.latent_input(z_delta, D)
            c, alpha = self.attention(ctx_tokens, self.queries[D](u))   # (J, S, B, T, d), (J, S, B, T, G')
            attention[D] = alpha.mean((0, 1))
            parts = [u, c]
            if self.uses_history:
                parts.append(h_tilde.expand(lead + h_tilde.shape[-1:]))
            if self.extra_dim:
                parts.append(extra)
            k, idx = len(self.index[D]), self.index[D]
            if self.anchor:
                feats = anchor_feats[:, :, idx].flatten(-2)          # (B, T, 3k)
                parts.append(feats.expand(lead + feats.shape[-1:]))
            out = self.heads[D](torch.cat(parts, dim=-1))
            logit = out[..., :k]
            if self.anchor:  # change from the last observed value (0 if none observed)
                logit = logit + base_logit[..., idx] * has[..., idx]
            p[..., idx] = torch.sigmoid(logit).clamp(1e-5, 1 - 1e-5)
            kappa[..., idx] = nn.functional.softplus(out[..., k:]) + self.kappa_min
        return {"p": p, "kappa": kappa, "attention": attention}

    def log_likelihood(self, y, p, kappa):
        """Per-outcome log-likelihood, unmasked: y (B, T, K) -> (J, S, B, T, K)."""
        return beta_binomial_log_prob(y.to(p.dtype), self.n, p, kappa)

    def mean(self, p):
        """Expected shifted score n_k * p."""
        return self.n * p