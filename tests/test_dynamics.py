import math

import torch
from torch import nn

from alzheimers_trajectory_mapping.dynamics import LatentDynamics


class Decay(nn.Module):
    """f(z) = -z: exact solution z(t) = z0 * exp(-t)."""

    def forward(self, z, e):
        return -z


def setup(J=3, S=2, B=4, T=5, dz=6):
    torch.manual_seed(0)
    return torch.randn(S, B, T, dz), torch.rand(B, T) * 3, torch.randn(J, 4)


def test_rk4_matches_exact_solution():
    z0, dt, emb = setup()
    dyn = LatentDynamics(6, 4, 16, steps=8, field=Decay())
    z = dyn(z0, dt, emb)
    exact = z0 * torch.exp(-dt)[None, :, :, None]
    assert torch.allclose(z[0], exact, rtol=2e-3, atol=1e-5)  # RK4, 8 steps, gap up to 3 years


def test_zero_gap_leaves_state_unchanged():
    z0, dt, emb = setup()
    dt[:, -1] = 0.0  # last visit
    z = LatentDynamics(6, 4, 16)(z0, dt, emb)
    assert torch.allclose(z[:, :, :, -1], z0[:, :, -1].expand_as(z[:, :, :, -1]))


def test_subtypes_change_the_trajectory():
    z0, dt, emb = setup()
    z = LatentDynamics(6, 4, 16)(z0, dt, emb)
    assert z.shape == (3, 2, 4, 5, 6)
    assert not torch.allclose(z[0], z[1])  # same start, different subtype -> different end


def test_longer_gap_more_change():
    z0, _, emb = setup()
    dyn = LatentDynamics(6, 4, 16, field=Decay())
    short = dyn(z0, torch.full((4, 5), 0.5), emb)
    long = dyn(z0, torch.full((4, 5), 2.0), emb)
    assert ((long - z0).norm() > (short - z0).norm())


def test_sde_is_stochastic_and_none_is_identity():
    z0, dt, emb = setup()
    sde = LatentDynamics(6, 4, 16, mode="sde")
    assert not torch.allclose(sde(z0, dt, emb), sde(z0, dt, emb))
    none = LatentDynamics(6, 4, 16, mode="none")
    assert torch.allclose(none(z0, dt, emb)[1], z0)


def test_penalty_and_gradients():
    z0, dt, emb = setup()
    emb.requires_grad_(True)
    dyn = LatentDynamics(6, 4, 16)
    pen = dyn.velocity_penalty(z0, emb)
    assert pen.shape == (3, 2, 4, 5) and (pen >= 0).all()
    (dyn(z0, dt, emb).sum() + pen.sum()).backward()
    assert emb.grad is not None and dyn.field.net[0].weight.grad is not None