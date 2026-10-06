"""Class-agnostic cell features for (row, model) pairs.

Every feature is invariant to class relabelling, so a single network handles any number of
classes C, models K and input dimensionality d. Label-dependent features are only filled for
context rows (whose labels are known); query rows get zeros plus a mask flag.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-4
N_FEATURES = 16
KNN = 10


def _knn_index(Xq: torch.Tensor, Xc: torch.Tensor, k: int, exclude_self: bool) -> torch.Tensor:
    """Indices (B, nq, k) of the k nearest context rows for each query row."""
    d = torch.cdist(Xq, Xc)
    if exclude_self:
        d = d + torch.diag_embed(torch.full(d.shape[:-1], float("inf"), device=d.device))
    k = min(k, Xc.shape[1] - int(exclude_self))
    return d.topk(k, dim=-1, largest=False).indices


def _gather_rows(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """t: (B, n, K) ; idx: (B, nq, k) -> (B, nq, k, K)"""
    B, nq, k = idx.shape
    flat = idx.reshape(B, nq * k)
    out = torch.gather(t, 1, flat.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
    return out.reshape(B, nq, k, t.shape[-1])


@torch.no_grad()
def cell_features(P_ctx, y_ctx, P_qry, X_ctx=None, X_qry=None, class_mask=None):
    """Build features.

    P_ctx: (B, n, K, C) probabilities on context rows (out-of-fold)
    y_ctx: (B, n) long labels of context rows
    P_qry: (B, m, K, C) probabilities on query rows
    X_ctx, X_qry: (B, n, d) / (B, m, d) inputs used for locality features (optional)
    class_mask: (B, C) bool, True for valid classes (for padded class dims)
    Returns F_ctx (B, n, K, F), F_qry (B, m, K, F), logP_ctx, logP_qry
    """
    B, n, K, C = P_ctx.shape
    if class_mask is None:
        class_mask = torch.ones(B, C, dtype=torch.bool, device=P_ctx.device)
    n_cls = class_mask.sum(-1).float()  # (B,)
    logC = torch.log(n_cls).clamp_min(math.log(2))

    def prep(P):
        P = P.clamp_min(EPS) * class_mask[:, None, None, :]
        P = P / P.sum(-1, keepdim=True)
        logP = torch.where(class_mask[:, None, None, :], P.log(), torch.full_like(P, -1e4))
        return P, logP

    P_ctx, logP_ctx = prep(P_ctx)
    P_qry, logP_qry = prep(P_qry)

    def label_free(P, logP):
        lc = logC[:, None, None]
        ent = -(P * logP.clamp_min(-30)).sum(-1) / lc
        top2 = P.topk(2, dim=-1).values
        maxp, margin = top2[..., 0], top2[..., 0] - top2[..., 1]
        pbar = P.mean(2, keepdim=True)  # consensus (B, n, 1, C)
        cons_arg = pbar.argmax(-1)  # (B, n, 1)
        arg = P.argmax(-1)  # (B, n, K)
        agree = (arg == cons_arg).float()
        kl = (P * (logP.clamp_min(-30) - pbar.clamp_min(EPS).log())).sum(-1) / lc
        lp_cons = torch.gather(logP, -1, cons_arg.unsqueeze(-1).expand(-1, -1, K, -1)).squeeze(-1)
        onehot = torch.nn.functional.one_hot(arg, C).float()  # (B,n,K,C)
        votes = onehot.sum(2, keepdim=True)  # (B,n,1,C)
        vote_share = ((votes * onehot).sum(-1) - 1) / max(K - 1, 1)
        # model-level deviation from consensus confidence
        conf_gap = maxp - pbar.squeeze(2).max(-1).values.unsqueeze(-1)
        return [ent, maxp, margin, agree, kl.clamp(max=5), lp_cons.clamp_min(-10) / 5, vote_share, conf_gap]

    f_ctx = label_free(P_ctx, logP_ctx)
    f_qry = label_free(P_qry, logP_qry)

    # label-dependent features (context only)
    lp_true = torch.gather(logP_ctx, -1, y_ctx[:, :, None, None].expand(-1, -1, K, 1)).squeeze(-1)
    correct = (P_ctx.argmax(-1) == y_ctx[:, :, None]).float()
    zeros_q = torch.zeros_like(f_qry[0])
    f_ctx += [lp_true.clamp_min(-10) / 5, correct]
    f_qry += [zeros_q, zeros_q]

    # locality features: performance of model k on nearest context neighbours
    if X_ctx is not None and n > 1:
        idx_c = _knn_index(X_ctx, X_ctx, KNN, exclude_self=True)
        idx_q = _knn_index(X_qry, X_ctx, KNN, exclude_self=False)
        loc = []
        for idx in (idx_c, idx_q):
            acc = _gather_rows(correct, idx).mean(2)
            lpt = _gather_rows(lp_true.clamp_min(-10) / 5, idx).mean(2)
            loc.append((acc, lpt))
        f_ctx += list(loc[0])
        f_qry += list(loc[1])
    else:
        f_ctx += [torch.zeros_like(f_ctx[0])] * 2
        f_qry += [torch.zeros_like(f_qry[0])] * 2

    # global context: model's overall context accuracy, K and C descriptors, context flag
    g_acc = correct.mean(1, keepdim=True)  # (B,1,K)
    g_lpt = (lp_true.clamp_min(-10) / 5).mean(1, keepdim=True)
    for f, rows, is_ctx in ((f_ctx, n, 1.0), (f_qry, P_qry.shape[1], 0.0)):
        shape = f[0].shape
        f += [g_acc.expand(shape), g_lpt.expand(shape),
              (logC[:, None, None] / math.log(10)).expand(shape),
              torch.full(shape, is_ctx, device=P_ctx.device)]

    F_ctx = torch.stack(f_ctx, -1)
    F_qry = torch.stack(f_qry, -1)
    assert F_ctx.shape[-1] == N_FEATURES, F_ctx.shape
    return F_ctx, F_qry, logP_ctx, logP_qry


N_CLASS_FEATURES = 9


@torch.no_grad()
def class_features(P_ctx, y_ctx, P_qry, class_mask=None):
    """Class-conditional (row, model, class) features for QUERY rows, computed from the context.

    Every feature is computed with the same function for every class, so the residual head that
    consumes them stays class-permutation-equivariant. Returns (B, m, K, C, N_CLASS_FEATURES).
    These carry the information a stacking meta-learner exploits with plenty of data: per-class
    precision / recall of each model, per-class calibration and class priors.
    """
    B, n, K, C = P_ctx.shape
    if class_mask is None:
        class_mask = torch.ones(B, C, dtype=torch.bool, device=P_ctx.device)
    cm = class_mask.float()

    def prep(P):
        P = P.clamp_min(EPS) * cm[:, None, None, :]
        return P / P.sum(-1, keepdim=True)

    P_ctx, P_qry = prep(P_ctx), prep(P_qry)
    Y = torch.nn.functional.one_hot(y_ctx, C).float()  # (B,n,C)
    A_ctx = torch.nn.functional.one_hot(P_ctx.argmax(-1), C).float()  # (B,n,K,C)
    hits = torch.einsum("bnkc,bnc->bkc", A_ctx, Y)
    pred_cnt = A_ctx.sum(1)  # (B,K,C)
    cls_cnt = Y.sum(1)[:, None, :]  # (B,1,C)
    precision = (hits + 1) / (pred_cnt + 2)
    recall = (hits + 1) / (cls_cnt + 2)
    L_ctx = P_ctx.log().clamp_min(-10) / 5
    lp_pos = torch.einsum("bnkc,bnc->bkc", L_ctx, Y) / cls_cnt.clamp_min(1)
    lp_neg = torch.einsum("bnkc,bnc->bkc", L_ctx, 1 - Y) / (n - cls_cnt).clamp_min(1)
    prior = torch.log((cls_cnt + 1) / (n + C)) / 3  # (B,1,C)

    m = P_qry.shape[1]
    L_q = P_qry.log().clamp_min(-10) / 5
    is_arg = torch.nn.functional.one_hot(P_qry.argmax(-1), C).float()
    pbar = P_qry.mean(2, keepdim=True).expand(-1, -1, K, -1)
    ex = lambda t: t[:, None].expand(-1, m, -1, -1)  # (B,K,C) -> (B,m,K,C)
    feats = [L_q, is_arg, P_qry, ex(precision), ex(recall), ex(lp_pos), ex(lp_neg),
             ex(prior.expand(-1, K, -1)), pbar]
    G = torch.stack(feats, -1) * cm[:, None, None, :, None]
    assert G.shape[-1] == N_CLASS_FEATURES
    return G
