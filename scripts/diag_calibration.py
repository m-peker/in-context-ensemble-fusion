"""Diagnostic: how much does per-model NON-LINEAR recalibration help simple fusion rules?

For each base model a class-shared beta calibration (Kull et al., 2017),
    z_c = a * log p_c - b * log(1 - p_c) + c_0 * [c is the argmax]   ->  q = softmax(z),
is cross-fitted on its out-of-fold predictions; greedy ensemble selection, the geometric mean and the single
best model are then applied to the recalibrated committee.
Usage: python scripts/diag_calibration.py [committees subfolder]   (DEV split)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench.baselines import greedy_ensemble_selection, geo_mean, _norm  # noqa: E402
from evaluate import iter_folds, metrics  # noqa: E402


def _feats(P):
    P = _norm(P)
    am = np.eye(P.shape[-1])[P.argmax(-1)]
    return np.log(P), np.log1p(-np.clip(P, 0, 1 - 1e-6)), am


def _apply(theta, F):
    a, b, c = theta
    z = a * F[0] - b * F[1] + c * F[2]
    z -= z.max(-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(-1, keepdims=True)


def fit_beta(P, y):
    F = _feats(P)
    nll = lambda th: -np.log(np.clip(_apply(th, F)[np.arange(len(y)), y], 1e-12, None)).mean() + 1e-4 * np.sum((th - [1, 0, 0]) ** 2)
    return minimize(nll, x0=np.array([1.0, 0.0, 0.0]), method="L-BFGS-B").x


def calibrate_committee(P_oof, y, P_te, seed=0):
    n, K, C = P_oof.shape
    Q_oof, Q_te = np.zeros_like(P_oof), np.zeros_like(P_te)
    folds = int(min(5, np.bincount(y)[np.bincount(y) > 0].min()))
    skf = StratifiedKFold(max(folds, 2), shuffle=True, random_state=seed)
    for k in range(K):
        for a, b in skf.split(P_oof[:, k], y):
            Q_oof[b, k] = _apply(fit_beta(P_oof[a, k], y[a]), _feats(P_oof[b, k]))
        Q_te[:, k] = _apply(fit_beta(P_oof[:, k], y), _feats(P_te[:, k]))
    return Q_oof, Q_te


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("committees", nargs="?", default="committees", help="subfolder of runs/")
    committees = ap.parse_args().committees
    rows = []
    for meta, fold, path in iter_folds("dev", os.path.join(ROOT, "runs", committees)):
        z = np.load(path)
        P_oof, y, P_te, yt = z["P_oof"], z["y_tr"].astype(int), z["P_te"], z["y_te"].astype(int)
        Q_oof, Q_te = calibrate_committee(P_oof, y, P_te)
        for name, P in [("cal+greedy_ES", greedy_ensemble_selection(Q_oof, y, Q_te)),
                        ("cal+geo_mean", geo_mean(Q_oof, y, Q_te)),
                        ("cal+single_best", _norm(Q_te[:, np.argmin([-np.log(Q_oof[np.arange(len(y)), k, y]).mean() for k in range(Q_oof.shape[1])])]))]:
            rows.append({"dataset": meta["name"], "fold": fold, "method": name, "sec": 0, **metrics(P, yt)})
        print(meta["name"], fold, flush=True)
    out = os.path.join(ROOT, "results", f"dev_diagcal_{committees}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
