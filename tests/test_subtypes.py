import math

import torch

from alzheimers_trajectory_mapping.subtypes import SubtypeModel, standard_normal_log_prob


def model(j=3):
    torch.manual_seed(0)
    return SubtypeModel(d=8, n_subtypes=j, embedding_dim=4, hidden=16)


def test_baseline_prior_uses_first_visit_only():
    m = model()
    h = torch.randn(2, 5, 8)
    lp = m.baseline_log_prior(h)
    assert lp.shape == (2, 3) and torch.allclose(lp.exp().sum(-1), torch.ones(2))
    h2 = h.clone()
    h2[:, 1:] += 5.0  # later visits change
    assert torch.allclose(lp, m.baseline_log_prior(h2))


def test_patient_posterior_follows_evidence():
    log_prior = torch.log(torch.full((1, 3), 1 / 3))
    loglik = torch.zeros(1, 4, 3)
    loglik[0, :, 1] = 2.0  # subtype 2 explains every transition better
    log_marginal, r = SubtypeModel.patient_posterior(log_prior, loglik)
    assert r[0, 1] > 0.99
    expected = torch.logsumexp(log_prior + loglik.sum(1), -1)
    assert torch.allclose(log_marginal, expected)


def test_temperature_softens_assignments():
    log_prior = torch.log(torch.full((1, 3), 1 / 3))
    loglik = torch.zeros(1, 4, 3)
    loglik[0, :, 0] = 1.0
    _, sharp = SubtypeModel.patient_posterior(log_prior, loglik, temperature=1.0)
    _, soft = SubtypeModel.patient_posterior(log_prior, loglik, temperature=10.0)
    entropy = lambda p: -(p * p.log()).sum()
    assert entropy(soft) > entropy(sharp)


def test_online_posterior_uses_only_past_transitions():
    log_prior = torch.log(torch.tensor([[0.5, 0.5]]))
    loglik = torch.zeros(1, 4, 2)
    loglik[0, 2, 0] = 5.0  # evidence from the transition 3 -> 4
    post = SubtypeModel.online_posterior(log_prior, loglik)
    assert torch.allclose(post[0, 0], torch.tensor([0.5, 0.5]))  # visit 1 = baseline prior
    assert torch.allclose(post[0, 2], torch.tensor([0.5, 0.5]))  # forecasting 3 -> 4: not yet seen
    assert post[0, 3, 0] > 0.99                                  # visit 4 has seen it


def test_gradients_reach_membership_network():
    m = model()
    h = torch.randn(4, 5, 8)
    loglik = torch.randn(4, 5, 3)
    log_marginal, _ = m.patient_posterior(m.baseline_log_prior(h), loglik)
    (-log_marginal.sum()).backward()
    assert m.membership[1].weight.grad is not None


def test_standard_normal_prior():
    z = torch.zeros(2, 3)
    assert torch.allclose(standard_normal_log_prob(z), torch.full((2,), -1.5 * math.log(2 * math.pi)))