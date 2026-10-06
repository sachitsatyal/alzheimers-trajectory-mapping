"""Piece 5: latent dynamics across the forecast gap (paper Sec. III-G),
conditioned on the patient's subtype.

  dz/dr = dt_{t->t+1} * f(z, e_c),   r in [0, 1],  z(0) = z0     (Eq. 8, + subtype)
  fixed-step RK4; scaling by the gap means a longer gap -> more evolution.
  Optional latent SDE (Eq. 9): + sqrt(dt) * g(z) dW, Euler-Maruyama.

f is autonomous: it depends on the current state and the subtype, not on
calendar time. Two patients in the same state progress differently only
through their subtype embedding e_c.
"""

import torch
from torch import nn


class VelocityField(nn.Module):
    """f(z, e_c): one tanh hidden layer (paper: 2-layer MLP, tanh)."""

    def __init__(self, dz: int, e_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dz + e_dim, hidden), nn.Tanh(), nn.Linear(hidden, dz))
        with torch.no_grad():  # start near "no change" so early training is stable
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, z, e):
        return self.net(torch.cat([z, e], dim=-1))


class Diffusion(nn.Module):
    """g(z) > 0, diagonal, for the SDE variant."""

    def __init__(self, dz: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dz, hidden), nn.Tanh(), nn.Linear(hidden, dz), nn.Softplus())

    def forward(self, z):
        return self.net(z)


class LatentDynamics(nn.Module):
    """mode: "ode" (paper default), "sde" (paper variant v), "none" (ablation iii)."""

    def __init__(self, dz: int, e_dim: int, hidden: int, steps: int = 8, mode: str = "ode",
                 field: nn.Module = None):
        super().__init__()
        assert mode in ("ode", "sde", "none"), mode
        self.mode, self.steps = mode, steps
        self.field = field if field is not None else VelocityField(dz, e_dim, hidden)
        self.diffusion = Diffusion(dz, hidden) if mode == "sde" else None

    @staticmethod
    def _expand(z0, dt, embeddings):
        """z0 (S, B, T, dz), dt (B, T), embeddings (J, e) ->
        z (J, S, B, T, dz), e (J, S, B, T, e), scale (1, 1, B, T, 1)."""
        J = embeddings.shape[0]
        z = z0.unsqueeze(0).expand(J, *z0.shape)
        e = embeddings[:, None, None, None, :].expand(*z.shape[:-1], embeddings.shape[-1])
        return z, e, dt[None, None, :, :, None]

    def forward(self, z0, dt, embeddings):
        """Evolve every posterior sample under every subtype across its own gap.
        Returns z_delta (J, S, B, T, dz). At the last visit dt = 0, so z_delta = z0."""
        z, e, scale = self._expand(z0, dt, embeddings)
        if self.mode == "none":
            return z
        h = 1.0 / self.steps
        f = lambda state: scale * self.field(state, e)
        for _ in range(self.steps):
            if self.mode == "ode":
                k1 = f(z)
                k2 = f(z + 0.5 * h * k1)
                k3 = f(z + 0.5 * h * k2)
                k4 = f(z + h * k3)
                z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:  # Euler-Maruyama; variance over the unit interval matches dt years
                noise = torch.randn_like(z)
                z = z + h * f(z) + torch.sqrt(h * scale) * self.diffusion(z) * noise
        return z

    def velocity_penalty(self, z0, embeddings):
        """R_t (Eq. 15): mean squared velocity at the start state, per (J, S, B, T)."""
        z, e, _ = self._expand(z0, torch.zeros(z0.shape[1:3], device=z0.device), embeddings)
        return self.field(z, e).pow(2).mean(-1)