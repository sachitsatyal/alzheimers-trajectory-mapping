import torch

from alzheimers_trajectory_mapping.history import GaussianPosterior, HistoryEncoder

CFG = {"d": 16, "encoder_heads": 4, "encoder_ff": 32, "dropout": 0.0, "encoder_layers": 2}


def inputs(b=3, t=6):
    torch.manual_seed(0)
    pi = torch.ones(b, t)
    pi[1, 4:] = 0  # patient 1 has 4 visits
    pi[2, 1:] = 0  # patient 2 has 1 visit
    return torch.randn(b, t, 16), pi


def test_no_future_leakage():
    enc = HistoryEncoder(CFG).eval()
    h, pi = inputs()
    out = enc(h, pi)
    h2 = h.clone()
    h2[:, 3] += 10.0  # change visit 4 only
    out2 = enc(h2, pi)
    assert torch.allclose(out[:, :3], out2[:, :3], atol=1e-5)  # visits 1-3 unaffected
    assert not torch.allclose(out[0, 3], out2[0, 3])


def test_padding_does_not_affect_real_visits_and_no_nans():
    enc = HistoryEncoder(CFG).eval()
    h, pi = inputs()
    out = enc(h, pi)
    h2 = h.clone()
    h2[2, 1:] = 99.0  # garbage in patient 2's padding
    out2 = enc(h2, pi)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[2, 0], out2[2, 0], atol=1e-5)


def test_posterior_shapes_sampling_and_deterministic():
    h, pi = inputs()
    post = GaussianPosterior(16, 8)
    mu, logvar, z = post(h, n_samples=5)
    assert mu.shape == (3, 6, 8) and z.shape == (5, 3, 6, 8)
    assert not torch.allclose(z[0], z[1])  # samples differ
    det = GaussianPosterior(16, 8, deterministic=True)
    _, _, zd = det(h, 5)
    assert torch.allclose(zd[0], zd[1])


def test_logvar_clamped_and_gradients_flow():
    post = GaussianPosterior(16, 8)
    h = torch.randn(2, 3, 16) * 1e4  # extreme input
    mu, logvar, z = post(h, 4)
    assert logvar.max() <= 5.0 and logvar.min() >= -10.0
    post.log_prob(z, mu, logvar).sum().backward()
    assert post.mu.weight.grad is not None