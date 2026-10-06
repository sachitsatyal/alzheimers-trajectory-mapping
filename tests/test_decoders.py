import torch
from scipy.stats import betabinom

from alzheimers_trajectory_mapping.decoders import DomainDecoders, beta_binomial_log_prob

META = {"outcomes": {"PSMS": {"domain": "A", "n": 24}, "IADL": {"domain": "A", "n": 23},
                     "NPIQ_SEV": {"domain": "B", "n": 36}, "NPIQ_COUNT": {"domain": "B", "n": 12},
                     "MMSE": {"domain": "C", "n": 30}}}
CATS = ["DEMOGRAPHIC", "CLINICAL", "ABC_HISTORY"]
CFG = {"d": 16, "latent_dim": 8, "decoder_hidden": 32, "decoder_dropout": 0.0, "kappa_min": 0.01}


def inputs(J=2, S=3, B=4, T=5):
    torch.manual_seed(0)
    return torch.randn(J, S, B, T, 8), torch.randn(B, T, 3, 16), torch.randn(B, T, 16)


def test_beta_binomial_matches_scipy_and_sums_to_one():
    n, p, kappa = torch.tensor(30.0), torch.tensor(0.7), torch.tensor(5.0)
    y = torch.arange(31.0)
    lp = beta_binomial_log_prob(y, n, p, kappa)
    ref = betabinom.logpmf(y.numpy(), 30, 0.7 * 5, 0.3 * 5)
    assert torch.allclose(lp, torch.tensor(ref, dtype=lp.dtype), atol=1e-4)
    assert abs(lp.exp().sum().item() - 1.0) < 1e-4


def test_shapes_attention_and_excluded_category():
    z, tokens, h = inputs()
    dec = DomainDecoders(META, CATS, CFG)  # default excludes ABC_HISTORY
    out = dec(z, tokens, h)
    assert out["p"].shape == (2, 3, 4, 5, 5) and (out["kappa"] > 0).all()
    assert set(out["attention"]) == {"A", "B", "C"}
    assert out["attention"]["C"].shape == (4, 5, 2)  # 3 categories minus ABC_HISTORY
    assert torch.allclose(out["attention"]["A"].sum(-1), torch.ones(4, 5))
    mean = dec.mean(out["p"])
    assert (mean >= 0).all() and (mean <= dec.n).all()


def test_paper_mode_and_history_switch():
    z, tokens, h = inputs()
    cfg = {**CFG, "decoder_exclude_categories": [], "decoder_uses_history": True}
    out = DomainDecoders(META, CATS, cfg)(z, tokens, h)
    assert out["attention"]["A"].shape == (4, 5, 3)


def test_excluded_category_cannot_influence_forecast():
    z, tokens, h = inputs()
    dec = DomainDecoders(META, CATS, CFG)
    t2 = tokens.clone()
    t2[..., 2, :] += 50.0  # change only the ABC_HISTORY token
    assert torch.allclose(dec(z, tokens, h)["p"], dec(z, t2, h)["p"])


def test_log_likelihood_and_gradients():
    z, tokens, h = inputs()
    z.requires_grad_(True)
    dec = DomainDecoders(META, CATS, CFG)
    out = dec(z, tokens, h)
    y = torch.randint(0, 12, (4, 5, 5))
    ll = dec.log_likelihood(y, out["p"], out["kappa"])
    assert ll.shape == (2, 3, 4, 5, 5) and torch.isfinite(ll).all() and (ll <= 0).all()
    ll.sum().backward()
    assert z.grad is not None and dec.queries["A"].weight.grad is not None