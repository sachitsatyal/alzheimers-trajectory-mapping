import torch
from scipy.stats import betabinom

from alzheimers_trajectory_mapping.decoders import DomainDecoders, beta_binomial_log_prob, last_observed

META = {"outcomes": {"PSMS": {"domain": "A", "n": 24}, "IADL": {"domain": "A", "n": 23},
                     "NPIQ_SEV": {"domain": "B", "n": 36}, "NPIQ_COUNT": {"domain": "B", "n": 12},
                     "MMSE": {"domain": "C", "n": 30}}}
CATS = ["DEMOGRAPHIC", "CLINICAL", "ABC_HISTORY"]
BLOCKS = {"S": 2, "A": 2, "B": 2, "C": 2}
CFG = {"d": 16, "decoder_hidden": 32, "decoder_dropout": 0.0, "kappa_min": 0.01, "decoder_anchor": False}


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
    dec = DomainDecoders(META, CATS, CFG, BLOCKS)
    out = dec(z, tokens, h)
    assert out["p"].shape == (2, 3, 4, 5, 5) and (out["kappa"] > 0).all()
    assert out["attention"]["C"].shape == (4, 5, 2)
    assert torch.allclose(out["attention"]["A"].sum(-1), torch.ones(4, 5))
    assert ((dec.mean(out["p"]) >= 0) & (dec.mean(out["p"]) <= dec.n)).all()


def test_no_path_from_history_except_through_z():
    z, tokens, h = inputs()
    dec = DomainDecoders(META, CATS, CFG, BLOCKS)
    assert torch.allclose(dec(z, tokens, h)["p"], dec(z, tokens, h + 10.0)["p"])
    t2 = tokens.clone()
    t2[..., 2, :] += 50.0  # excluded ABC_HISTORY token
    assert torch.allclose(dec(z, tokens, h)["p"], dec(z, t2, h)["p"])


def test_each_domain_reads_only_its_own_block_and_shared():
    z, tokens, h = inputs()
    dec = DomainDecoders(META, CATS, CFG, BLOCKS)
    z2 = z.clone()
    z2[..., 4:6] += 5.0  # behavior block only
    p1, p2 = dec(z, tokens, h)["p"], dec(z2, tokens, h)["p"]
    assert torch.allclose(p1[..., [0, 1, 4]], p2[..., [0, 1, 4]])  # function and cognition unchanged
    assert not torch.allclose(p1[..., 2:4], p2[..., 2:4])           # behavior changed


def test_switches_log_likelihood_and_gradients():
    z, tokens, h = inputs()
    z.requires_grad_(True)
    cfg = {**CFG, "decoder_exclude_categories": [], "decoder_uses_history": True}
    dec = DomainDecoders(META, CATS, cfg, BLOCKS)
    out = dec(z, tokens, h)
    assert out["attention"]["A"].shape == (4, 5, 3)
    ll = dec.log_likelihood(torch.randint(0, 12, (4, 5, 5)), out["p"], out["kappa"])
    assert torch.isfinite(ll).all()
    ll.sum().backward()
    assert z.grad is not None


def test_last_observed_carries_forward_with_age():
    Y = torch.tensor([[[10], [0], [12], [0]]])
    MU = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])
    MU[0, 0, 0] = 0.0
    Y[0, 1, 0] = 7
    t_abs = torch.tensor([[0.0, 1.0, 2.5, 4.0]])
    value, has, age = last_observed(Y, MU, t_abs)
    assert has[0, :, 0].tolist() == [0.0, 1.0, 1.0, 1.0]       # nothing observed at visit 1
    assert value[0, :, 0].tolist() == [0.0, 7.0, 7.0, 0.0]     # visit 3 carries 7 forward; visit 4 observed 0
    assert age[0, :, 0].tolist() == [0.0, 0.0, 1.5, 0.0]


def anchored_inputs():
    z, tokens, h = inputs()
    Y = torch.randint(0, 12, (4, 5, 5))
    MU = torch.ones(4, 5, 5)
    t_abs = torch.arange(5.0).expand(4, 5)
    return z, tokens, h, Y, MU, t_abs


def test_anchor_starts_at_last_value_carried_forward():
    z, tokens, h, Y, MU, t_abs = anchored_inputs()
    dec = DomainDecoders(META, CATS, {**CFG, "decoder_anchor": True}, BLOCKS)
    with torch.no_grad():  # an untrained "change" of exactly zero
        for head in dec.heads.values():
            head[-1].weight.zero_()
            head[-1].bias.zero_()
    out = dec(z, tokens, h, anchor=last_observed(Y, MU, t_abs))
    expected = (Y.float() + 0.5) * dec.n / (dec.n + 1)
    assert torch.allclose(dec.mean(out["p"])[0, 0], expected, atol=1e-3)


def test_anchor_never_sees_the_future():
    z, tokens, h, Y, MU, t_abs = anchored_inputs()
    dec = DomainDecoders(META, CATS, {**CFG, "decoder_anchor": True}, BLOCKS)
    Y2 = Y.clone()
    Y2[:, 3:] = 0  # change visits 4-5 only
    p1 = dec(z, tokens, h, anchor=last_observed(Y, MU, t_abs))["p"]
    p2 = dec(z, tokens, h, anchor=last_observed(Y2, MU, t_abs))["p"]
    assert torch.allclose(p1[..., :3, :], p2[..., :3, :])  # forecasts made at visits 1-3 unchanged