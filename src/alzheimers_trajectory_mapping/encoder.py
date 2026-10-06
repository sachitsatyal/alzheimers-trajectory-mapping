"""Piece 2: the visit encoder (paper Sec. III-B and III-C).

Turns one visit's inputs into a single vector h_t:
  category tokens  e_t^(g) = LN_g(W_g [x_t^(g); m_t^(g)] + b_g)    (Eq. 1, + LayerNorm)
  fusion           h_t = sum_g softmax(E_t q_f / sqrt(d))_g e_t^(g) (Eq. 2)
  + Time2Vec(t_abs, dt)                                         (Eq. 3)
  + sum of static embeddings (index 0 = missing)
Each visit is encoded independently; visits interact only in the history encoder.

Two additions to the paper, both switchable in the config:
  token_norm         LayerNorm per category token, so attention weights reflect
                     relevance rather than token size (168 biomarkers vs 4 genomic).
  category_dropout   during training, hide a whole category at random (as if
                     not measured) so the model cannot lean on any one category.
"""

import math

import torch
from torch import nn


class CategoryTokenizer(nn.Module):
    """One affine encoder per category: values (+ masks) -> a d-dim token."""

    def __init__(self, blocks: dict, d: int, use_masks: bool = True, token_norm: bool = True):
        super().__init__()
        self.names = list(blocks)
        self.slices = [tuple(blocks[n]) for n in self.names]
        self.use_masks = use_masks
        width = lambda a, b: (b - a) * (2 if use_masks else 1)
        self.encoders = nn.ModuleList(nn.Linear(width(a, b), d) for a, b in self.slices)
        self.norms = nn.ModuleList(nn.LayerNorm(d) if token_norm else nn.Identity()
                                   for _ in self.slices)

    def forward(self, x, m, keep=None):
        """x, m: (B, T, p); keep: optional (B, T, G) 0/1 -> tokens E: (B, T, G, d).

        A category with keep = 0 is encoded as fully unobserved (values and
        masks zero), which yields its learned "absent" token.
        """
        tokens = []
        for g, ((a, b), enc, norm) in enumerate(zip(self.slices, self.encoders, self.norms)):
            xg, mg = x[..., a:b], m[..., a:b]
            if keep is not None:
                k = keep[..., g:g + 1]
                xg, mg = xg * k, mg * k
            inp = torch.cat([xg, mg], dim=-1) if self.use_masks else xg
            tokens.append(norm(enc(inp)))
        return torch.stack(tokens, dim=-2)


class TokenAttention(nn.Module):
    """softmax(E q / sqrt(d)) over categories; used for fusion (global query)
    and, in the decoders, with a history-dependent query."""

    def forward(self, tokens, query):
        """tokens: (B, T, G, d); query: (d,) or (B, T, d) -> context (B, T, d), weights (B, T, G)."""
        d = tokens.shape[-1]
        if query.dim() == 1:
            scores = tokens @ query / math.sqrt(d)
        else:
            scores = (tokens * query.unsqueeze(-2)).sum(-1) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * tokens).sum(-2), weights


class Time2Vec(nn.Module):
    """Eq. (3): linear trend + (k-1) learned sinusoids per time input, projected to d."""

    def __init__(self, n_inputs: int, k: int, d: int):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.periodic = nn.Linear(n_inputs, n_inputs * (k - 1))
        self.proj = nn.Linear(n_inputs * k, d)

    def forward(self, tau):
        """tau: (B, T, 2) -> (B, T, d)."""
        return self.proj(torch.cat([self.linear(tau), torch.sin(self.periodic(tau))], dim=-1))


class StaticEmbedding(nn.Module):
    """One table per static covariate (size = levels + 1, index 0 = missing); summed.

    Initialized small (std 0.02) so the static sum does not swamp the visit
    vector at the start of training (PyTorch's default std is 1).
    """

    def __init__(self, n_levels: list, d: int):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(n + 1, d) for n in n_levels)
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)

    def forward(self, s):
        """s: (B, n_static) long -> (B, d)."""
        return sum(table(s[:, i]) for i, table in enumerate(self.tables))


class VisitEncoder(nn.Module):
    def __init__(self, meta: dict, cfg: dict):
        super().__init__()
        d = cfg["d"]
        dropped = set(cfg.get("drop_categories", []))
        self.blocks = {k: v for k, v in meta["category_blocks"].items() if k not in dropped}
        self.categories = list(self.blocks)
        self.tokenizer = CategoryTokenizer(self.blocks, d, cfg.get("use_masks", True),
                                           cfg.get("token_norm", True))
        self.category_dropout = cfg.get("category_dropout", 0.0)
        self.fusion_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.attention = TokenAttention()
        self.time2vec = Time2Vec(2, cfg["time2vec_k"], d) if cfg.get("use_time2vec", True) else None
        n_levels = [len(meta["static_levels"][c]) for c in meta["static_columns"]]
        self.static = StaticEmbedding(n_levels, d)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, batch):
        """Returns h (B, T, d), the token bank E (B, T, G, d) and fusion weights (B, T, G)."""
        x, m = batch["X"], batch["M"]
        keep = None
        if self.training and self.category_dropout > 0:
            shape = x.shape[:-1] + (len(self.categories),)
            keep = (torch.rand(shape, device=x.device) >= self.category_dropout).to(x.dtype)
        tokens = self.tokenizer(x, m, keep)
        h, alpha = self.attention(tokens, self.fusion_query)
        if self.time2vec is not None:
            h = h + self.time2vec(batch["tau"])
        h = h + self.static(batch["S"]).unsqueeze(1)
        return {"h": self.dropout(h), "tokens": tokens, "alpha_fuse": alpha}