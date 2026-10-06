"""Inference wrapper: fuse a real committee with a pretrained FusionPFN (no training)."""
from __future__ import annotations

import numpy as np
import torch

from .features import cell_features, class_features
from .model import FusionPFN


def _default_device():
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def load_model(path, device=None):
    """Load a pretrained FusionPFN checkpoint (GPU if available, otherwise CPU)."""
    device = device or _default_device()
    st = torch.load(path, map_location=device, weights_only=False)
    a = st["args"]
    m = FusionPFN(d=a["d"], layers=a["layers"], heads=a["heads"], residual=a.get("residual", False)).to(device).eval()
    m.load_state_dict(st["model"])
    m.use_loc = not a.get("no_loc", False)
    m.device = device
    return m


@torch.no_grad()
def fusionpfn_predict(model, P_oof, y_tr, P_te, X_tr=None, X_te=None, max_ctx=2048, chunk=1024,
                      n_perm=1, seed=0, device=None, return_weights=False):
    """Fuse a committee in context -- no training.

    P_oof : (n, K, C) out-of-fold class probabilities of the K members on the n training rows
    y_tr  : (n,) integer labels in 0..C-1
    P_te  : (m, K, C) probabilities of the members (refitted on all training rows) on the m query rows
    X_tr, X_te : optional standardised inputs (only used by checkpoints trained with locality features)
    n_perm > 1 averages predictions over random context sub-samples and member orders.
    Returns fused probabilities (m, C) [and instance-wise member weights (m, K)].
    """
    device = device or getattr(model, "device", None) or _default_device()
    use_amp = str(device).startswith("cuda")
    rng = np.random.default_rng(seed)
    n, K, C = P_oof.shape
    if not getattr(model, "use_loc", True):
        X_tr = X_te = None
    outs, ws = [], []
    for r in range(n_perm):
        ctx = np.arange(n) if n <= max_ctx else np.sort(rng.choice(n, max_ctx, replace=False))
        perm = np.arange(K) if r == 0 else rng.permutation(K)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)
        Pc = t(P_oof[ctx][:, perm]).float()[None]
        yc = t(y_tr[ctx]).long()[None]
        Xc = t(X_tr[ctx]).float()[None] if X_tr is not None else None
        out = np.zeros((len(P_te), C)); w_all = np.zeros((len(P_te), K))
        for s in range(0, len(P_te), chunk):
            sl = slice(s, s + chunk)
            Pq = t(P_te[sl][:, perm]).float()[None]
            Xq = t(X_te[sl]).float()[None] if X_te is not None else None
            F_ctx, F_qry, _, logP_qry = cell_features(Pc, yc, Pq, Xc, Xq)
            G = class_features(Pc, yc, Pq) if model.residual else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logp, w = model(F_ctx, F_qry, logP_qry, G_qry=G)
            out[sl] = logp.float().exp()[0].cpu().numpy()
            w_all[sl][:, perm] = w.float()[0].cpu().numpy()
        outs.append(out); ws.append(w_all)
    P = np.mean(outs, 0)
    P = P / P.sum(-1, keepdims=True)
    return (P, np.mean(ws, 0)) if return_weights else P
