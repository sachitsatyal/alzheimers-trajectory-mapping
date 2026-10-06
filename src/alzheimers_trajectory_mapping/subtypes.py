"""Piece 4 (redesigned): patient-level progression subtypes.

Each patient has ONE discrete subtype c in {1..J}, fixed over time. The subtype
does not describe where a patient is (that is the latent state z), but how they
progress: it conditions the latent dynamics f(z, c) (piece 5).

  membership   pi(c | baseline) = softmax(g(h~_1))       assigns new patients at enrollment
  trajectory   log p(Y_i) = logsumexp_c [ log pi(c | baseline) + sum_t l_t(c) ]
  posterior    r_ic = softmax_c(...)                      patient's subtype probabilities
  online       P(c | data up to visit t) uses only transitions before t (no leakage)

l_t(c) is the masked log-likelihood of the outcomes at visit t+1 forecast from
visit t under subtype c (zero at positions without a forecast target).
This is a continuous-time, neural latent class trajectory model.
"""

import math

import torch
from torch import nn


def standard_normal_log_prob(z):
    """Prior on the continuous state z: N(0, I). Sum over the last dim."""
    return -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)


class SubtypeModel(nn.Module):
    def __init__(self, d: int, n_subtypes: int, embedding_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.J = n_subtypes
        self.membership = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_subtypes))
        # One vector per subtype, read by the velocity field f(z, c).
        self.embedding = nn.Embedding(n_subtypes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=1.0)  # distinct from the start

    def baseline_log_prior(self, h_tilde):
        """log pi(c | baseline visit), from the history summary at visit 1: (B, J)."""
        return torch.log_softmax(self.membership(h_tilde[:, 0]), dim=-1)

    @staticmethod
    def patient_posterior(log_prior, transition_loglik, temperature: float = 1.0):
        """Marginal log-likelihood per patient and subtype probabilities.

        log_prior: (B, J); transition_loglik: (B, T, J), already masked.
        temperature > 1 softens assignments early in training so one subtype
        cannot capture every patient before the others have learned anything.
        Returns log_marginal (B,) and responsibilities (B, J).
        """
        joint = log_prior + transition_loglik.sum(1)
        log_marginal = temperature * torch.logsumexp(joint / temperature, dim=-1)
        return log_marginal, torch.softmax(joint / temperature, dim=-1)

    @staticmethod
    def online_posterior(log_prior, transition_loglik):
        """P(c | data up to visit t), for forecasting at visit t: (B, T, J).

        Uses the baseline prior and the transitions BEFORE t (u -> u+1, u < t);
        the transition from t to t+1 is the one being forecast, so it is excluded.
        """
        before = transition_loglik.cumsum(1) - transition_loglik
        return torch.softmax(log_prior.unsqueeze(1) + before, dim=-1)

    @staticmethod
    def subtype_sizes(responsibilities):
        """Expected share of patients per subtype: (J,)."""
        return responsibilities.mean(0)