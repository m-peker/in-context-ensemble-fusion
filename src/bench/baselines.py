"""Fusion baselines operating on precomputed committees.

Each baseline receives (P_oof, y_tr, P_te, X_tr, X_te) and returns test probabilities (n_te, C).
P_oof / P_te: (n, K, C).
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors

EPS = 1e-6


def _norm(P):
    P = np.clip(P, EPS, None)
    return P / P.sum(-1, keepdims=True)


def _logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean()


def single_best(P_oof, y, P_te, **_):
    k = int(np.argmin([_logloss(_norm(P_oof[:, j]), y) for j in range(P_oof.shape[1])]))
    return _norm(P_te[:, k])


def single_best_acc(P_oof, y, P_te, **_):
    k = int(np.argmax([(P_oof[:, j].argmax(-1) == y).mean() for j in range(P_oof.shape[1])]))
    return _norm(P_te[:, k])


def arith_mean(P_oof, y, P_te, **_):
    return _norm(P_te.mean(1))


def geo_mean(P_oof, y, P_te, **_):
    L = np.log(_norm(P_te)).mean(1)
    L -= L.max(-1, keepdims=True)
    return _norm(np.exp(L))


def majority_vote(P_oof, y, P_te, **_):
    C = P_te.shape[-1]
    votes = np.eye(C)[P_te.argmax(-1)].sum(1)
    # smooth so that log-loss stays finite; ties broken by mean prob
    return _norm(votes + 1e-3 * P_te.mean(1) + 0.5)


def greedy_ensemble_selection(P_oof, y, P_te, n_iter=50, **_):
    """Caruana et al. (2004): forward selection with replacement on OOF log-loss."""
    K = P_oof.shape[1]
    counts = np.zeros(K)
    cur = np.zeros_like(P_oof[:, 0])
    for it in range(n_iter):
        best, best_k = np.inf, 0
        for k in range(K):
            cand = (cur * it + P_oof[:, k]) / (it + 1)
            l = _logloss(_norm(cand), y)
            if l < best:
                best, best_k = l, k
        counts[best_k] += 1
        cur = (cur * it + P_oof[:, best_k]) / (it + 1)
    w = counts / counts.sum()
    return _norm(np.einsum("k,nkc->nc", w, P_te))


def _logfeat(P):
    L = np.log(_norm(P))
    return L.reshape(len(L), -1)


def stacking_lr(P_oof, y, P_te, **_):
    """Logistic regression on log-probabilities of all base models, C tuned by inner CV."""
    Xa, Xb = _logfeat(P_oof), _logfeat(P_te)
    C = P_te.shape[-1]
    cv = int(min(5, np.bincount(y, minlength=C)[np.bincount(y, minlength=C) > 0].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cv >= 2:
            m = LogisticRegressionCV(Cs=8, cv=cv, max_iter=2000, scoring="neg_log_loss")
        else:
            m = LogisticRegression(max_iter=2000)
        m.fit(Xa, y)
    out = np.full((len(Xb), C), EPS)
    out[:, m.classes_] = m.predict_proba(Xb)
    return _norm(out)


def knora_e_soft(P_oof, y, P_te, X_tr=None, X_te=None, k=7, **_):
    """KNORA-Eliminate (Ko et al. 2008) with soft voting of the selected models."""
    if X_tr is None:
        return arith_mean(P_oof, y, P_te)
    correct = P_oof.argmax(-1) == y[:, None]  # (n,K)
    nn = NearestNeighbors(n_neighbors=min(k, len(y))).fit(X_tr)
    idx = nn.kneighbors(X_te, return_distance=False)  # (m,k)
    out = np.zeros(P_te.shape[::2])
    for i in range(len(X_te)):
        for kk in range(idx.shape[1], 0, -1):
            sel = correct[idx[i, :kk]].all(0)
            if sel.any():
                break
        if not sel.any():
            sel = np.ones(P_te.shape[1], bool)
        out[i] = P_te[i, sel].mean(0)
    return _norm(out)


BASELINES = {
    "single_best": single_best,
    "arith_mean": arith_mean,
    "geo_mean": geo_mean,
    "majority_vote": majority_vote,
    "greedy_ES": greedy_ensemble_selection,
    "stack_LR": stacking_lr,
    "KNORA-E": knora_e_soft,
}


# --------------------------------------------------------------------------------------------
# Regularized Neural Ensemblers (Pineda Arango et al., AutoML 2025) -- official code, imported
# from external/RegularizedNeuralEnsemblers without installing (its pinned deps would downgrade
# torch / sklearn). Hyper-parameters = README defaults. Trained per dataset on the OOF rows.
# --------------------------------------------------------------------------------------------
def _neural_ensembler(P_oof, y, P_te, mode, epochs=1000, seed=42, **_):
    import contextlib
    import io
    import os
    import sys
    import tempfile

    import torch

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "external", "RegularizedNeuralEnsemblers")
    if root not in sys.path:
        sys.path.insert(0, root)
    from regularized_neural_ensemblers.model import NeuralEnsembler
    from regularized_neural_ensemblers.trainer import Trainer
    from regularized_neural_ensemblers.trainer_args import TrainerArgs

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xa = np.transpose(_norm(P_oof), (0, 2, 1)).astype(np.float32)  # (n, C, K)
    Xb = np.transpose(_norm(P_te), (0, 2, 1)).astype(np.float32)
    n, C, K = Xa.shape
    model = NeuralEnsembler(num_base_functions=K, num_classes=C, hidden_dim=32, num_layers=3,
                            dropout_rate=0.2, task_type="classification", mode=mode).to(dev)
    with tempfile.TemporaryDirectory() as td:
        args = TrainerArgs(batch_size=512, lr=0.001, epochs=epochs, device=dev,
                           checkpoint_name=os.path.join(td, "ne.pt"))
        with contextlib.redirect_stdout(io.StringIO()):
            Trainer(model=model, trainer_args=args).fit(Xa, y)
    model.eval()
    with torch.no_grad():
        P = model.predict(Xb)
    return _norm(np.asarray(P, dtype=np.float64))


def neural_ensembler_stack(P_oof, y, P_te, **kw):
    return _neural_ensembler(P_oof, y, P_te, mode="stacking")


def neural_ensembler_avg(P_oof, y, P_te, **kw):
    return _neural_ensembler(P_oof, y, P_te, mode="model_averaging")


BASELINES["NE_stack"] = neural_ensembler_stack
BASELINES["NE_avg"] = neural_ensembler_avg


# --------------------------------------------------------------------------------------------
# TabPFN v2 (Hollmann et al., Nature 2025) used as the meta-learner on OOF log-probabilities
# ("TabPFN as stacker"). tabpfn==2.0.9; supports <= 10 classes -> returns None otherwise.
# --------------------------------------------------------------------------------------------
_TABPFN = {}


def tabpfn_stacker(P_oof, y, P_te, seed=42, **_):
    C = P_te.shape[-1]
    if C > 10:
        return None
    from tabpfn import TabPFNClassifier

    Xa, Xb = _logfeat(P_oof), _logfeat(P_te)
    m = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xa, y)
        p = m.predict_proba(Xb)
    out = np.full((len(Xb), C), EPS)
    out[:, m.classes_] = p
    return _norm(out)


BASELINES["TabPFN_stack"] = tabpfn_stacker


# --------------------------------------------------------------------------------------------
# Fitted pooling rules. All are fitted on the OOF rows of the target
# data set and differ from FusionPFN only in that their parameters are global and fitted rather
# than instance-wise and inferred.
#   LogPool_fit    : log p_c ∝ sum_k w_k log p_kc + b_c   (K exponents + C intercepts, unconstrained;
#                    L2 towards the geometric mean, strength chosen by 3-fold CV)
#   LogPool_convex : Heskes-type logarithmic pool, w = tau * softmax(a) (convex weights + temperature)
#   TS_avg         : per-member temperature scaling, then arithmetic mean ("calibrate, then average")
#   LinPool_fit    : linear pool with simplex weights maximising the OOF log score
#                    (stacking of predictive distributions, Yao et al. 2018)
# --------------------------------------------------------------------------------------------
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402


def _logs(P):
    return np.log(_norm(P))  # (n, K, C), clipped at log(EPS)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _fit_logpool(L, y, lam, K, C):
    Y = np.eye(C)[y]
    w0 = np.full(K, 1.0 / K)

    def f(theta):
        w, b = theta[:K], theta[K:]
        S = _softmax(np.einsum("k,nkc->nc", w, L) + b)
        nll = -np.log(np.clip(S[np.arange(len(y)), y], 1e-15, None)).mean()
        G = (S - Y) / len(y)
        gw = np.einsum("nc,nkc->k", G, L) + 2 * lam * (w - w0)
        gb = G.sum(0) + 2 * lam * b
        return nll + lam * (np.sum((w - w0) ** 2) + np.sum(b ** 2)), np.concatenate([gw, gb])

    res = minimize(f, np.concatenate([w0, np.zeros(C)]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:K], res.x[K:]


def logpool_fit(P_oof, y, P_te, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=42, **_):
    n, K, C = P_oof.shape
    L, Lt = _logs(P_oof), _logs(P_te)
    counts = np.bincount(y, minlength=C)
    folds = int(min(3, counts[counts > 0].min()))
    best = lams[2]
    if folds >= 2:
        cv = []
        for lam in lams:
            ll = 0.0
            for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(L[:, 0], y):
                w, c = _fit_logpool(L[a], y[a], lam, K, C)
                S = _softmax(np.einsum("k,nkc->nc", w, L[b]) + c)
                ll += -np.log(np.clip(S[np.arange(len(b)), y[b]], 1e-15, None)).sum()
            cv.append(ll)
        best = lams[int(np.argmin(cv))]
    w, c = _fit_logpool(L, y, best, K, C)
    return _norm(_softmax(np.einsum("k,nkc->nc", w, Lt) + c))


def logpool_convex(P_oof, y, P_te, lam=1e-3, **_):
    n, K, C = P_oof.shape
    L, Lt = _logs(P_oof), _logs(P_te)
    Y = np.eye(C)[y]

    def f(theta):
        a, lt = theta[:K], theta[K]
        s = _softmax(a); tau = np.exp(lt); w = tau * s
        S = _softmax(np.einsum("k,nkc->nc", w, L))
        nll = -np.log(np.clip(S[np.arange(n), y], 1e-15, None)).mean()
        gw = np.einsum("nc,nkc->k", (S - Y) / n, L)
        ga = tau * (s * gw - s * (s @ gw)) + 2 * lam * a
        gt = w @ gw
        return nll + lam * np.sum(a ** 2), np.concatenate([ga, [gt]])

    res = minimize(f, np.zeros(K + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    w = np.exp(res.x[K]) * _softmax(res.x[:K])
    return _norm(_softmax(np.einsum("k,nkc->nc", w, Lt)))


def ts_average(P_oof, y, P_te, **_):
    n, K, C = P_oof.shape
    L, Lt = _logs(P_oof), _logs(P_te)
    out = np.zeros(P_te.shape[::2])
    for k in range(K):
        nll = lambda lb: -np.log(np.clip(_softmax(np.exp(lb) * L[:, k])[np.arange(n), y], 1e-15, None)).mean()
        lb = minimize_scalar(nll, bounds=(-4, 3), method="bounded").x
        out += _softmax(np.exp(lb) * Lt[:, k])
    return _norm(out / K)


def linpool_fit(P_oof, y, P_te, **_):
    n, K, C = P_oof.shape
    Py = _norm(P_oof)[np.arange(n), :, y]  # (n, K) probability of the true class per member

    def f(a):
        s = _softmax(a); p = Py @ s
        nll = -np.log(np.clip(p, 1e-15, None)).mean()
        gs = -(Py / np.clip(p, 1e-15, None)[:, None]).mean(0)
        return nll, s * gs - s * (s @ gs)

    res = minimize(f, np.zeros(K), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return _norm(np.einsum("k,nkc->nc", _softmax(res.x), _norm(P_te)))


BASELINES["LogPool_fit"] = logpool_fit
BASELINES["LogPool_convex"] = logpool_convex
BASELINES["TS_avg"] = ts_average
BASELINES["LinPool_fit"] = linpool_fit
