import torch

from alzheimers_trajectory_mapping.dynamics import CoupledDynamics, integrate

BLOCKS = {"S": 3, "A": 2, "B": 2, "C": 2}


def setup(J=3, S=2, B=4, T=5, dz=9):
    torch.manual_seed(0)
    return torch.randn(S, B, T, dz), torch.rand(B, T) * 3, torch.randn(J, 4)


def dyn(**kw):
    torch.manual_seed(1)
    return CoupledDynamics(BLOCKS, e_dim=4, hidden=16, n_subtypes=3, **kw)


def test_rk4_matches_exact_solution():
    z0 = torch.randn(5, 3)
    dt = torch.rand(5, 1) * 3
    z = integrate(lambda s: -s, z0, dt, steps=8)
    assert torch.allclose(z, z0 * torch.exp(-dt), rtol=2e-3, atol=1e-5)  # RK4, gap up to 3 y


def test_zero_gap_and_shapes():
    z0, dt, emb = setup()
    dt[:, -1] = 0.0
    z = dyn()(z0, dt, emb)
    assert z.shape == (3, 2, 4, 5, 9)
    assert torch.allclose(z[:, :, :, -1], z0[:, :, -1].expand_as(z[:, :, :, -1]))


def test_starts_uncoupled_and_shared_block_is_autonomous():
    z0, dt, emb = setup()
    d = dyn()
    z2 = z0.clone()
    z2[..., 5:7] += 3.0  # change behavior block B only
    out1, out2 = d(z0, dt, emb), d(z2, dt, emb)
    assert torch.allclose(d.block(out1, "A"), d.block(out2, "A"))  # W = 0: A unaffected by B
    assert torch.allclose(d.block(out1, "S"), d.block(out2, "S"))  # S never depends on domains


def test_coupling_carries_influence_between_domains():
    z0, dt, emb = setup()
    d = dyn()
    with torch.no_grad():
        d.W["A_from_B"].fill_(0.5)
    z2 = z0.clone()
    z2[..., 5:7] += 3.0
    assert not torch.allclose(d.block(d(z0, dt, emb), "A"), d.block(d(z2, dt, emb), "A"))
    assert d.interaction_map()[0]["A<-B"] > 0 and d.interaction_map()[0]["B<-A"] < 1e-5


def test_subtypes_differ_and_ablations_run():
    z0, dt, emb = setup()
    z = dyn()(z0, dt, emb)
    assert not torch.allclose(z[0], z[1])
    assert torch.allclose(dyn(mode="none")(z0, dt, emb)[1], z0)
    assert dyn(mode="sde")(z0, dt, emb).shape == z.shape
    assert dyn(coupling=False)(z0, dt, emb).shape == z.shape


def test_penalties_and_gradients():
    z0, dt, emb = setup()
    emb.requires_grad_(True)
    d = dyn()
    pen = d.velocity_penalty(z0, emb)
    norms = d.coupling_norms()
    assert pen.shape == (3, 2, 4, 5) and norms.shape == (3, 6)
    (d(z0, dt, emb).sum() + pen.sum() + norms.sum()).backward()
    assert emb.grad is not None and d.W["A_from_B"].grad is not None