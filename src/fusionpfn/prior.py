"""Prior over model committees.

Instead of a prior over datasets (as in TabPFN), we place a prior over how a *committee* of
base models behaves on a task. A synthetic task is generated as:

  latent inputs   u_i ~ mixture of Gaussians in R^m
  true logits     g(u) = s * RFF(u)                      (random smooth function, s = separability)
  labels          y_i ~ Cat(softmax(g(u_i)))
  model k logits  l_k(u) = tau_k * ( a_k g(u) + sigma_k(u) * eps_k(u) + b_k )
     a_k        signal retention (underfitting / junk models: a_k ~ 0)
     sigma_k(u) local error scale = competence map (smooth in u)
     eps_k(u)   rho_k * shared group error (correlated mistakes) + independent error
                (smooth part + row-wise noise part)
     tau_k      temperature (over-/under-confidence), b_k class bias
  plus duplicated models, "hard" (near one-hot) models such as trees/kNN.
Query rows mimic refit-on-all-data models with a slightly smaller error scale than the
out-of-fold context rows. Everything is sampled on the GPU in batches.
"""
from __future__ import annotations

import math

import torch


# Prior hyper-parameters (ranges are sampled uniformly / log-uniformly per task).
# Ranges are chosen so that the statistics of sampled committees (agreement, accuracy spread,
# oracle gap) cover those of typical real committees; see scripts/prior_stats.py.
PRIOR = {
    "sep": (1.0, 10.0),          # log-uniform, class separability
    "base_err": (0.1, 1.0),      # log-uniform, model error scale
    "signal": (0.5, 1.1),        # signal retention a_k
    "p_junk": 0.05,
    "comp_amp": (0.0, 0.8),      # strength of local competence variation
    "rho": (0.3, 0.95),          # share of error that is common within a model family
    "tau": (0.3, 3.0),           # log-uniform temperature
    "p_dup": 0.1,
    "p_hard": 0.25,
    "p_confuse": 0.3,            # a model leaks evidence of class a into class b systematically
    "confuse_strength": (0.2, 1.0),
}


def _rff(u, out_dim, lengthscale, n_feat=64, gen=None):
    """Random Fourier-feature function u:(B,n,m) -> (B,n,out_dim), roughly unit variance."""
    B, n, m = u.shape
    dev = u.device
    W = torch.randn(B, m, n_feat, device=dev, generator=gen) / lengthscale[:, None, None]
    b = torch.rand(B, 1, n_feat, device=dev, generator=gen) * 2 * math.pi
    A = torch.randn(B, n_feat, out_dim, device=dev, generator=gen)
    phi = torch.cos(u @ W + b) * math.sqrt(2.0 / n_feat)
    return phi @ A


def _u(B, shape, lo, hi, dev, gen):
    return lo + (hi - lo) * torch.rand(B, *shape, device=dev, generator=gen)


@torch.no_grad()
def sample_committee_tasks(B, n_ctx, n_qry, K, C, device="cuda", gen=None):
    """Return dict with P_ctx (B,n,K,C), y_ctx, P_qry (B,m,K,C), y_qry, X_ctx, X_qry."""
    dev = device
    N = n_ctx + n_qry
    m = int(torch.randint(1, 6, (1,), generator=None).item())
    # ---- inputs: Gaussian mixture in latent space
    n_clu = 4
    centers = torch.randn(B, n_clu, m, device=dev, generator=gen) * _u(B, (1, 1), 0.0, 2.0, dev, gen)
    assign = torch.randint(0, n_clu, (B, N), device=dev, generator=gen)
    u = torch.gather(centers, 1, assign[..., None].expand(-1, -1, m)) + torch.randn(B, N, m, device=dev, generator=gen)
    # ---- ground truth
    ls = _u(B, (), 0.5, 3.0, dev, gen)
    sep = torch.exp(_u(B, (1, 1), *map(math.log, PRIOR["sep"]), dev, gen))
    g = _rff(u, C, ls, gen=gen)
    g = g + _u(B, (1, C), -1.0, 1.0, dev, gen)  # class imbalance
    g = sep * g
    y = torch.distributions.Categorical(logits=g).sample()

    # ---- committee
    G = int(torch.randint(1, K + 1, (1,)).item())  # error groups ("model families")
    group = torch.randint(0, G, (B, K), device=dev, generator=gen)
    shared = torch.stack([_rff(u, C, ls * _u(B, (), 0.5, 2.0, dev, gen), gen=gen) for _ in range(G)], 2)  # (B,N,G,C)
    shared = torch.gather(shared, 2, group[:, None, :, None].expand(-1, N, -1, C))  # (B,N,K,C)

    a = _u(B, (K,), *PRIOR["signal"], dev, gen)
    junk = torch.rand(B, K, device=dev, generator=gen) < PRIOR["p_junk"]
    a = torch.where(junk, torch.zeros_like(a), a)
    base_err = torch.exp(_u(B, (K,), *map(math.log, PRIOR["base_err"]), dev, gen))
    comp_amp = _u(B, (K,), *PRIOR["comp_amp"], dev, gen)  # strength of local competence variation
    comp = torch.stack([_rff(u, 1, ls, gen=gen).squeeze(-1) for _ in range(K)], -1)  # (B,N,K)
    sigma = base_err[:, None, :] * torch.exp(comp_amp[:, None, :] * comp)
    rho = _u(B, (K,), *PRIOR["rho"], dev, gen)
    indep_smooth = torch.stack([_rff(u, C, ls, gen=gen) for _ in range(K)], 2)
    row_noise_frac = _u(B, (K,), 0.0, 1.0, dev, gen)
    indep = (1 - row_noise_frac[:, None, :, None]).sqrt() * indep_smooth + \
        row_noise_frac[:, None, :, None].sqrt() * torch.randn(B, N, K, C, device=dev, generator=gen)
    eps = rho[:, None, :, None].sqrt() * shared + (1 - rho[:, None, :, None]).sqrt() * indep
    # refit-on-all-data models (query rows) are a bit more accurate than OOF models
    refit = torch.ones(B, N, 1, 1, device=dev)
    refit[:, n_ctx:] = _u(B, (1, 1, 1), 0.8, 1.0, dev, gen)
    tau = torch.exp(_u(B, (K,), *map(math.log, PRIOR["tau"]), dev, gen))
    bias = 0.5 * torch.randn(B, 1, K, C, device=dev, generator=gen) * (torch.rand(B, 1, K, 1, device=dev, generator=gen) < 0.3)
    logits = tau[:, None, :, None] * (a[:, None, :, None] * g[:, :, None, :] + refit * sigma[..., None] * eps * sep.unsqueeze(-1).sqrt() + bias)
    # systematic confusions: for a subset of models, evidence for class a leaks into a fixed class b
    if C > 1 and PRIOR["p_confuse"] > 0:
        tgt = torch.randint(0, C, (B, K, C), device=dev, generator=gen)  # a -> b map
        A = torch.nn.functional.one_hot(tgt, C).float()  # (B,K,C,C)
        A = A * (1 - torch.eye(C, device=dev))  # no self-leak
        strength = _u(B, (K,), *PRIOR["confuse_strength"], dev, gen)
        on = (torch.rand(B, K, device=dev, generator=gen) < PRIOR["p_confuse"]).float()
        leak = torch.einsum("bnc,bkcd->bnkd", torch.relu(g), A)
        logits = logits + (on * strength * tau)[:, None, :, None] * leak
    # duplicates: copy another model with small perturbation
    dup = torch.rand(B, K, device=dev, generator=gen) < PRIOR["p_dup"]
    src = torch.randint(0, K, (B, K), device=dev, generator=gen)
    copied = torch.gather(logits, 2, src[:, None, :, None].expand(-1, N, -1, C)) + 0.1 * torch.randn_like(logits)
    logits = torch.where(dup[:, None, :, None], copied, logits)
    P = torch.softmax(logits, -1)
    # hard models (tree / kNN-like): quantize probabilities
    hard = torch.rand(B, K, device=dev, generator=gen) < PRIOR["p_hard"]
    levels = torch.randint(2, 20, (B, 1, K, 1), device=dev, generator=gen).float()
    Pq = torch.round(P * levels) / levels
    Pq = (Pq + 1e-3) / (Pq + 1e-3).sum(-1, keepdim=True)
    P = torch.where(hard[:, None, :, None], Pq, P)

    return {
        "P_ctx": P[:, :n_ctx], "y_ctx": y[:, :n_ctx], "X_ctx": u[:, :n_ctx],
        "P_qry": P[:, n_ctx:], "y_qry": y[:, n_ctx:], "X_qry": u[:, n_ctx:],
    }
