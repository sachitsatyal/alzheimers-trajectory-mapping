"""Piece 7a: the full model, assembling pieces 2-6.

  visit encoder -> causal history -> posterior q(z_t | H_t) -> samples z0
  -> dynamics under every subtype across the gap -> domain decoders
  plus the subtype membership prior P(c | baseline visit).
"""

import torch
from torch import nn

from .decoders import DomainDecoders
from .dynamics import LatentDynamics
from .encoder import VisitEncoder
from .history import GaussianPosterior, HistoryEncoder
from .subtypes import SubtypeModel


class ABCModel(nn.Module):
    def __init__(self, meta: dict, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d, dz, e = cfg["d"], cfg["latent_dim"], cfg["subtype_embedding_dim"]
        self.encoder = VisitEncoder(meta, cfg)
        self.history = HistoryEncoder(cfg)
        self.posterior = GaussianPosterior(d, dz, cfg.get("deterministic", False))
        self.subtypes = SubtypeModel(d, cfg["n_subtypes"], e, cfg["membership_hidden"], cfg["dropout"])
        self.dynamics = LatentDynamics(dz, e, cfg["velocity_hidden"], cfg["ode_steps"],
                                       cfg.get("dynamics", "ode"))
        # Without dynamics, the gap and the subtype must reach the decoder directly.
        extra = 1 + e if cfg.get("dynamics", "ode") == "none" else 0
        self.decoders = DomainDecoders(meta, self.encoder.categories, {**cfg, "decoder_extra_dim": extra})

    def forward(self, batch, n_samples: int):
        enc = self.encoder(batch)
        h_tilde = self.history(enc["h"], batch["PI"])
        mu, logvar, z0 = self.posterior(h_tilde, n_samples)
        embeddings = self.subtypes.embedding.weight
        z_delta = self.dynamics(z0, batch["dt_next"], embeddings)
        extra = None
        if self.decoders.extra_dim:
            J, S, B, T, _ = z_delta.shape
            dt = batch["dt_next"][None, None, :, :, None].expand(J, S, B, T, 1)
            emb = embeddings[:, None, None, None, :].expand(J, S, B, T, embeddings.shape[-1])
            extra = torch.cat([dt, emb], dim=-1)
        dec = self.decoders(z_delta, enc["tokens"], h_tilde, extra)
        return {"h_tilde": h_tilde, "mu": mu, "logvar": logvar, "z0": z0, "z_delta": z_delta,
                "log_prior": self.subtypes.baseline_log_prior(h_tilde),
                "alpha_fuse": enc["alpha_fuse"], **dec}