"""Piece 3: history encoder and amortized posterior (paper Sec. III-D, III-E).

  h~_t = causal transformer over visits 1..t           (visit t never sees t+1)
  q(z_t | H_t) = N(mu_t, diag sigma_t^2),  mu_t, log sigma_t^2 linear in h~_t
  z_t^(s) = mu_t + sigma_t * eps^(s)                   (reparameterization, Eq. 5)
"""

import torch
from torch import nn

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0  # keeps sigma in [0.007, 12]: avoids exp() overflow / zero variance


class HistoryEncoder(nn.Module):
    """Pre-norm causal transformer; output at t summarizes visits <= t only."""

    def __init__(self, cfg: dict):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg["d"], nhead=cfg["encoder_heads"], dim_feedforward=cfg["encoder_ff"],
            dropout=cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
        # Nested tensors are a speed path that mishandles some mask combinations.
        self.layers = nn.TransformerEncoder(layer, cfg["encoder_layers"], enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(cfg["d"])  # pre-norm stacks need a closing norm

    def forward(self, h, pi):
        """h: (B, T, d) visit vectors; pi: (B, T) 1 = real visit -> h~ (B, T, d)."""
        t = h.shape[1]
        causal = torch.triu(torch.full((t, t), float("-inf"), device=h.device), diagonal=1)
        padding = torch.zeros_like(pi).masked_fill(pi == 0, float("-inf"))
        out = self.layers(h, mask=causal, src_key_padding_mask=padding)
        return self.final_norm(out)


class GaussianPosterior(nn.Module):
    """Amortized q(z_t | H_t). deterministic=True gives z = mu (paper ablation i)."""

    def __init__(self, d: int, dz: int, deterministic: bool = False, logvar_init: float = -2.0):
        super().__init__()
        self.mu = nn.Linear(d, dz)
        self.logvar = nn.Linear(d, dz)
        # Start with moderately confident samples (sigma ~ 0.37 at -2) instead of
        # sigma ~ 1: very noisy early samples teach the decoder to ignore z.
        nn.init.constant_(self.logvar.bias, logvar_init)
        self.deterministic = deterministic

    def forward(self, h_tilde, n_samples: int):
        """-> mu, logvar (B, T, dz) and samples z0 (S, B, T, dz)."""
        mu = self.mu(h_tilde)
        logvar = self.logvar(h_tilde).clamp(LOGVAR_MIN, LOGVAR_MAX)
        if self.deterministic:
            return mu, logvar, mu.unsqueeze(0).expand(n_samples, *mu.shape)
        eps = torch.randn((n_samples,) + mu.shape, device=mu.device, dtype=mu.dtype)
        return mu, logvar, mu + torch.exp(0.5 * logvar) * eps

    @staticmethod
    def log_prob(z, mu, logvar):
        """log q(z) summed over latent dims; z (S, B, T, dz) -> (S, B, T)."""
        return torch.distributions.Normal(mu, torch.exp(0.5 * logvar)).log_prob(z).sum(-1)