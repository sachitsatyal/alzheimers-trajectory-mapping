import torch

from alzheimers_trajectory_mapping.encoder import CategoryTokenizer, VisitEncoder

META = {
    "category_blocks": {"DEMOGRAPHIC": [0, 2], "CLINICAL": [2, 5], "ABC_HISTORY": [5, 8]},
    "static_columns": ["A1_SEX", "APOE_E4_COUNT"],
    "static_levels": {"A1_SEX": [0.0, 1.0], "APOE_E4_COUNT": [0.0, 1.0, 2.0]},
}
CFG = {"d": 16, "time2vec_k": 4, "dropout": 0.0}


def batch(b=3, t=5, p=8):
    torch.manual_seed(0)
    return {"X": torch.randn(b, t, p), "M": torch.ones(b, t, p),
            "tau": torch.rand(b, t, 2) * 5, "S": torch.tensor([[1, 0], [2, 3], [0, 1]])}


def test_shapes_and_fusion_weights():
    out = VisitEncoder(META, CFG)(batch())
    assert out["h"].shape == (3, 5, 16)
    assert out["tokens"].shape == (3, 5, 3, 16)
    assert torch.allclose(out["alpha_fuse"].sum(-1), torch.ones(3, 5))


def test_fully_missing_category_gives_absent_token():
    tok = CategoryTokenizer({"A": [0, 3]}, d=4)
    x, m = torch.zeros(1, 1, 3), torch.zeros(1, 1, 3)
    expected = tok.norms[0](tok.encoders[0].bias)
    assert torch.allclose(tok(x, m)[0, 0, 0], expected, atol=1e-6)


def test_token_norm_makes_token_sizes_comparable():
    tok = CategoryTokenizer({"SMALL": [0, 2], "LARGE": [2, 202]}, d=16)
    x, m = torch.randn(4, 3, 202) * 5, torch.ones(4, 3, 202)
    norms = tok(x, m).norm(dim=-1)
    assert torch.allclose(norms[..., 0], norms[..., 1], rtol=0.05)


def test_category_dropout_only_in_training():
    cfg = {**CFG, "category_dropout": 1.0}  # hide every category
    enc = VisitEncoder(META, cfg)
    b = batch()
    absent = enc.tokenizer(torch.zeros_like(b["X"]), torch.zeros_like(b["M"]))
    assert torch.allclose(enc(b)["tokens"], absent, atol=1e-6)
    enc.eval()
    assert not torch.allclose(enc(b)["tokens"], absent)


def test_ablation_switches():
    cfg = {**CFG, "drop_categories": ["CLINICAL"], "use_masks": False,
           "use_time2vec": False, "token_norm": False}
    out = VisitEncoder(META, cfg)(batch())
    assert out["tokens"].shape[-2] == 2 and out["h"].shape == (3, 5, 16)


def test_time_changes_output_and_gradients_flow():
    enc = VisitEncoder(META, CFG)
    b = batch()
    h1 = enc(b)["h"]
    assert not torch.allclose(h1, enc({**b, "tau": b["tau"] + 1.0})["h"])
    h1.sum().backward()
    assert enc.fusion_query.grad is not None and enc.static.tables[0].weight.grad is not None