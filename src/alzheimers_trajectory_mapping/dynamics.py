"""Piece 5 (coupled ABC version): latent dynamics with explicit cross-domain coupling.

The latent state is split into blocks:
  z_S  shared severity  (common disease process, drives all domains)
  z_A  function, z_B behavior, z_C cognition  (each read only by its own decoder)

For subtype c, across the gap to the next visit (r in [0, 1], scaled by dt):
  dz_S/dt = f_S(z_S, e_c)                                    S is autonomous
  dz_D/dt = f_D(z_D, z_S, e_c) + sum_{D' != D} W_{D<-D'}(c) tanh(z_D')

W_{D<-D'}(c) is the ABC interaction map: how strongly domain D' drives change
in domain D for subtype c, beyond the shared severity. W starts at zero
(uncoupled) and a group-lasso penalty keeps it sparse, so a link only appears
if the data support it. Interpretation: directed predictive influence, not
proven causation.
"""

import torch
from torch import nn

SHARED = "S"


def _mlp(inp, hidden, out):
    net = nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, out))
    with torch.no_grad():  # start near "no change"
        net[-1].weight.mul_(0.1)
        net[-1].bias.zero_()
    return net


def integrate(velocity, z, scale, steps, mode="ode", diffusion=None):
    """Fixed-step solver over r in [0, 1] for dz/dr = scale * velocity(z).
    ode: RK4. sde: Euler-Maruyama with noise sqrt(scale * h) * g(z)."""
    h = 1.0 / steps
    f = lambda state: scale * velocity(state)
    for _ in range(steps):
        if mode == "ode":
            k1 = f(z)
            k2 = f(z + 0.5 * h * k1)
            k3 = f(z + 0.5 * h * k2)
            k4 = f(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            z = z + h * f(z) + torch.sqrt(h * scale) * diffusion(z) * torch.randn_like(z)
    return z


class CoupledDynamics(nn.Module):
    """mode: "ode" (default), "sde", "none" (ablation: no evolution).
    coupling=False removes all cross-domain terms (ablation)."""

    def __init__(self, blocks: dict, e_dim: int, hidden: int, n_subtypes: int,
                 steps: int = 8, mode: str = "ode", coupling: bool = True):
        super().__init__()
        assert mode in ("ode", "sde", "none"), mode
        assert list(blocks)[0] == SHARED, "first block must be the shared block S"
        self.blocks, self.mode, self.steps, self.coupling = dict(blocks), mode, steps, coupling
        self.domains = [k for k in blocks if k != SHARED]
        start, self.slices = 0, {}
        for k, n in blocks.items():
            self.slices[k] = slice(start, start + n)
            start += n
        self.dz = start
        dS = blocks[SHARED]
        self.fields = nn.ModuleDict({SHARED: _mlp(dS + e_dim, hidden, dS)})
        for D in self.domains:
            self.fields[D] = _mlp(blocks[D] + dS + e_dim, hidden, blocks[D])
        self.pairs = [(D, E) for D in self.domains for E in self.domains if E != D]
        if coupling:
            self.W = nn.ParameterDict({f"{D}_from_{E}": nn.Parameter(torch.zeros(n_subtypes, blocks[D], blocks[E]))
                                       for D, E in self.pairs})
        self.diffusion = (nn.Sequential(nn.Linear(self.dz, hidden), nn.Tanh(), nn.Linear(hidden, self.dz),
                                        nn.Softplus()) if mode == "sde" else None)

    def block(self, z, name):
        return z[..., self.slices[name]]

    def velocity(self, z, e):
        """z (J, S, B, T, dz), e (J, S, B, T, e_dim) -> dz/dt, same shape as z."""
        zS = self.block(z, SHARED)
        parts = {SHARED: self.fields[SHARED](torch.cat([zS, e], -1))}
        for D in self.domains:
            v = self.fields[D](torch.cat([self.block(z, D), zS, e], -1))
            if self.coupling:
                for E in self.domains:
                    if E != D:  # W[c] applies to the trajectories of subtype c (leading dim J)
                        v = v + torch.einsum("jdk,jsbtk->jsbtd", self.W[f"{D}_from_{E}"], torch.tanh(self.block(z, E)))
            parts[D] = v
        return torch.cat([parts[k] for k in self.blocks], -1)

    @staticmethod
    def _expand(z0, dt, embeddings):
        J = embeddings.shape[0]
        z = z0.unsqueeze(0).expand(J, *z0.shape)
        e = embeddings[:, None, None, None, :].expand(*z.shape[:-1], embeddings.shape[-1])
        return z, e, dt[None, None, :, :, None]

    def forward(self, z0, dt, embeddings):
        """z0 (S, B, T, dz), dt (B, T), embeddings (J, e) -> z_delta (J, S, B, T, dz)."""
        z, e, scale = self._expand(z0, dt, embeddings)
        if self.mode == "none":
            return z
        return integrate(lambda s: self.velocity(s, e), z, scale, self.steps, self.mode, self.diffusion)

    def velocity_penalty(self, z0, embeddings):
        """R (Eq. 15): mean squared velocity at the start state, (J, S, B, T)."""
        z, e, _ = self._expand(z0, torch.zeros(z0.shape[1:3], device=z0.device), embeddings)
        return self.velocity(z, e).pow(2).mean(-1)

    def coupling_norms(self):
        """Group-lasso terms: Frobenius norm of each W_{D<-D'}(c), shape (J, n_pairs)."""
        if not self.coupling:
            return torch.zeros(1)
        return torch.stack([torch.sqrt(self.W[f"{D}_from_{E}"].pow(2).sum((1, 2)) + 1e-12)
                            for D, E in self.pairs], dim=1)

    @torch.no_grad()
    def interaction_map(self):
        """{subtype: {"A<-B": strength, ...}}: the learned ABC interaction map."""
        norms = self.coupling_norms()
        if not self.coupling:
            return {}
        return {c: {f"{D}<-{E}": float(norms[c, i]) for i, (D, E) in enumerate(self.pairs)}
                for c in range(norms.shape[0])}