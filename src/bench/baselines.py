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
    import torch
    m = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed,
                         ignore_pretraining_limits=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xa, y)
        p = m.predict_proba(Xb)
    out = np.full((len(Xb), C), EPS)
    out[:, m.classes_] = p
    return _norm(out)


BASELINES["TabPFN_stack"] = tabpfn_stacker
