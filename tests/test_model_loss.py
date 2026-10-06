import torch

from alzheimers_trajectory_mapping.loss import compute_loss, next_visit_targets
from alzheimers_trajectory_mapping.model import ABCModel

META = {
    "category_blocks": {"DEMOGRAPHIC": [0, 2], "CLINICAL": [2, 5], "ABC_HISTORY": [5, 8]},
    "static_columns": ["A1_SEX"], "static_levels": {"A1_SEX": [0.0, 1.0]},
    "outcomes": {"PSMS": {"domain": "A", "n": 24}, "IADL": {"domain": "A", "n": 23},
                 "NPIQ_SEV": {"domain": "B", "n": 36}, "NPIQ_COUNT": {"domain": "B", "n": 12},
                 "MMSE": {"domain": "C", "n": 30}},
}
CFG = {"d": 16, "encoder_layers": 1, "encoder_heads": 2, "encoder_ff": 32, "dropout": 0.0,
       "time2vec_k": 4, "latent_blocks": {"S": 2, "A": 2, "B": 2, "C": 2}, "velocity_hidden": 16, "ode_steps": 4,
       "decoder_hidden": 16, "decoder_dropout": 0.0, "kappa_min": 0.01,
       "n_subtypes": 3, "subtype_embedding_dim": 4, "membership_hidden": 8}


def batch(B=4, T=5):
    g = torch.Generator().manual_seed(0)
    pi = torch.ones(B, T)
    pi[1, 3:] = 0
    pi[2, 1:] = 0  # single-visit patient: no forecast positions
    return {"X": torch.randn(B, T, 8, generator=g), "M": torch.ones(B, T, 8),
            "tau": torch.rand(B, T, 2, generator=g), "dt_next": torch.rand(B, T, generator=g) * pi,
            "Y": torch.randint(0, 12, (B, T, 5), generator=g), "MU": torch.ones(B, T, 5) * pi[..., None],
            "PI": pi, "S": torch.tensor([[1], [2], [0], [1]])}


def run(cfg=CFG, b=None, seed=0):
    torch.manual_seed(seed)
    model = ABCModel(META, cfg)
    b = b or batch()
    torch.manual_seed(1)
    out = model(b, n_samples=3)
    return model, b, out, compute_loss(model, b, out, beta=1.0)


def test_targets_are_next_visit():
    b = batch()
    y, mu, fp = next_visit_targets(b)
    assert torch.equal(y[:, 0], b["Y"][:, 1])
    assert fp[2].sum() == 0 and fp[1].sum() == 2 and fp[0].sum() == 4


def test_full_forward_and_finite_loss():
    model, b, out, (loss, stats) = run()
    assert out["p"].shape == (3, 3, 4, 5, 5)
    assert torch.isfinite(loss) and stats["forecast_positions"] == 10
    assert abs(sum(stats["subtype_sizes"]) - 1.0) < 1e-5
    loss.backward()
    assert model.subtypes.membership[1].weight.grad is not None
    assert model.encoder.tokenizer.encoders[0].weight.grad is not None


def test_first_visit_outcomes_are_never_targets():
    # With the anchor on, visit-1 scores are an input (the anchor for forecasting visit 2),
    # so check the target logic with the anchor off.
    cfg = {**CFG, "decoder_anchor": False}
    _, b, _, (loss1, _) = run(cfg)
    b2 = {**b, "Y": b["Y"].clone()}
    b2["Y"][:, 0] = 0
    _, _, _, (loss2, _) = run(cfg, b=b2)
    assert torch.allclose(loss1, loss2)


def test_unobserved_outcome_contributes_nothing():
    _, b, _, (loss1, _) = run()
    b2 = {**b, "Y": b["Y"].clone(), "MU": b["MU"].clone()}
    b2["MU"][0, 2, 4] = 0
    b2["Y"][0, 2, 4] = 11  # changing a masked target must not matter
    _, _, _, (l_masked, _) = run(b=b2)
    b3 = {**b2, "Y": b2["Y"].clone()}
    b3["Y"][0, 2, 4] = 0
    _, _, _, (l_masked2, _) = run(b=b3)
    assert torch.allclose(l_masked, l_masked2)


def test_variants_run():
    for extra in [{"dynamics": "none"}, {"dynamics": "sde"}, {"deterministic": True}, {"coupling": False},
                  {"decoder_exclude_categories": []}, {"decoder_uses_history": True}]:
        _, _, _, (loss, _) = run({**CFG, **extra})
        assert torch.isfinite(loss), extra


def test_free_bits_floor():
    model, b, out, _ = run()
    loss_fb, _ = compute_loss(model, b, out, beta=1.0, free_bits=10.0)
    loss_0, _ = compute_loss(model, b, out, beta=1.0, free_bits=0.0)
    assert loss_fb >= loss_0  # a high floor can only raise the KL term


def test_coupling_penalty_adds_to_loss():
    model, b, out, _ = run()
    with torch.no_grad():
        model.dynamics.W["A_from_B"].fill_(1.0)
    loss_on, stats = compute_loss(model, b, out, coupling_penalty=1.0)
    loss_off, _ = compute_loss(model, b, out, coupling_penalty=0.0)
    assert loss_on > loss_off and stats["coupling_total"] > 0