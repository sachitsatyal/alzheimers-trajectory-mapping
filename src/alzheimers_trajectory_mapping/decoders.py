"""Piece 6: domain decoders with category attention and beta-binomial outputs
(paper Sec. III-H).

For each domain D in {A, B, C}:
  q_t^D = W_Q^D h~_t + b                     history-dependent query
  alpha_t^D = softmax(E_t q_t^D / sqrt(d))   category attention (Aim 2)
  c_t^D = sum_g alpha_t,g^D e_t^(g)          category context
  [logit p, kappa~]_k = MLP^D([z_delta; c_t^D])  for each outcome k in D
  y ~ BetaBinomial(n_k, p * kappa, (1 - p) * kappa)

Config switches (decide on validation):
  decoder_exclude_categories  categories the decoders may not attend to.
      Default [ABC_HISTORY]: the current outcome values then reach the forecast
      only through z, which guards against posterior collapse and keeps the
      subtypes and dynamics meaningful (Aims 1 and 3). [] = paper behavior.
  decoder_uses_history        also feed h~_t to the decoder (architecture figure).
      Default false (paper): the forecast must flow through the evolved state.
"""

import torch
from torch import nn

from .encoder import TokenAttention


def beta_binomial_log_prob(y, n, p, kappa):
    """log BetaBin(y | n, a = p * kappa, b = (1 - p) * kappa)   (Eq. 12)."""
    a, b = p * kappa, (1 - p) * kappa
    lbeta = lambda x, w: torch.lgamma(x) + torch.lgamma(w) - torch.lgamma(x + w)
    log_choose = torch.lgamma(n + 1) - torch.lgamma(y + 1) - torch.lgamma(n - y + 1)
    return log_choose + lbeta(y + a, n - y + b) - lbeta(a, b)


class DomainDecoders(nn.Module):
    def __init__(self, meta: dict, categories: list, cfg: dict):
        super().__init__()
        d, dz = cfg["d"], cfg["latent_dim"]
        self.outcomes = list(meta["outcomes"])
        self.domains = sorted({meta["outcomes"][k]["domain"] for k in self.outcomes})
        self.index = {D: [i for i, k in enumerate(self.outcomes) if meta["outcomes"][k]["domain"] == D]
                      for D in self.domains}
        self.register_buffer("n", torch.tensor([float(meta["outcomes"][k]["n"]) for k in self.outcomes]))

        excluded = set(cfg.get("decoder_exclude_categories", ["ABC_HISTORY"]))
        self.context_categories = [c for c in categories if c not in excluded]
        self.register_buffer("context_idx", torch.tensor(
            [i for i, c in enumerate(categories) if c not in excluded], dtype=torch.long))
        self.uses_history = cfg.get("decoder_uses_history", False)
        self.extra_dim = cfg.get("decoder_extra_dim", 0)  # e.g. dt + subtype when dynamics = none
        self.kappa_min = cfg["kappa_min"]

        in_dim = dz + d + (d if self.uses_history else 0) + self.extra_dim
        self.attention = TokenAttention()
        self.queries = nn.ModuleDict({D: nn.Linear(d, d) for D in self.domains})
        self.heads = nn.ModuleDict({D: nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg["decoder_hidden"]), nn.GELU(),
            nn.Dropout(cfg["decoder_dropout"]), nn.Linear(cfg["decoder_hidden"], 2 * len(self.index[D])))
            for D in self.domains})

    def forward(self, z_delta, tokens, h_tilde, extra=None):
        """z_delta (J, S, B, T, dz); tokens (B, T, G, d); h_tilde (B, T, d).
        Returns p, kappa (J, S, B, T, K) and attention {D: (B, T, G')}."""
        lead = z_delta.shape[:-1]
        ctx_tokens = tokens.index_select(-2, self.context_idx)
        p = z_delta.new_zeros(lead + (len(self.outcomes),))
        kappa = torch.zeros_like(p)
        attention = {}
        for D in self.domains:
            c, alpha = self.attention(ctx_tokens, self.queries[D](h_tilde))
            attention[D] = alpha
            parts = [z_delta, c.expand(lead + c.shape[-1:])]
            if self.uses_history:
                parts.append(h_tilde.expand(lead + h_tilde.shape[-1:]))
            if self.extra_dim:
                parts.append(extra)
            out = self.heads[D](torch.cat(parts, dim=-1))
            k = len(self.index[D])
            idx = self.index[D]
            p[..., idx] = torch.sigmoid(out[..., :k]).clamp(1e-5, 1 - 1e-5)
            kappa[..., idx] = nn.functional.softplus(out[..., k:]) + self.kappa_min
        return {"p": p, "kappa": kappa, "attention": attention}

    def log_likelihood(self, y, p, kappa):
        """Per-outcome log-likelihood, unmasked: y (B, T, K) -> (J, S, B, T, K)."""
        return beta_binomial_log_prob(y.to(p.dtype), self.n, p, kappa)

    def mean(self, p):
        """Expected shifted score n_k * p."""
        return self.n * p