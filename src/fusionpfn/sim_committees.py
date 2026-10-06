"""Simulated-committee prior: REAL learning algorithms trained on SYNTHETIC datasets.

The parametric prior (prior.py) models committee errors as additive noise in logit space, where
logarithmic pooling is nearly Bayes-optimal. Real committees deviate (exact zeros from trees / kNN,
extreme Naive-Bayes posteriors, family-specific error structure). This module produces committee
tasks whose predictions come from actual scikit-learn models fitted on synthetic data, so the
pretraining distribution contains these idiosyncrasies. No benchmark data is used.

The family pool deliberately goes beyond the 8 benchmark base models (different hyper-parameters,
SVM, AdaBoost, LDA/QDA, decision trees, ...), so that the prior is not tailored to the benchmark.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def family_pool(rng: np.random.Generator):
    """Return a list of (name, factory) with randomized hyper-parameters."""
    s = int(rng.integers(1 << 30))
    pool = [
        ("lr", lambda: LogisticRegression(C=float(10 ** rng.uniform(-2, 2)), max_iter=1000)),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=int(rng.integers(3, 40)),
                                             weights=str(rng.choice(["uniform", "distance"])))),
        ("nb", lambda: GaussianNB()),
        ("rf", lambda: RandomForestClassifier(n_estimators=int(rng.integers(30, 200)),
                                              max_depth=[None, 4, 8][int(rng.integers(3))],
                                              random_state=s, n_jobs=1)),
        ("et", lambda: ExtraTreesClassifier(n_estimators=int(rng.integers(30, 200)), random_state=s, n_jobs=1)),
        ("hgb", lambda: HistGradientBoostingClassifier(max_iter=int(rng.integers(30, 200)),
                                                       learning_rate=float(10 ** rng.uniform(-1.5, -0.5)),
                                                       random_state=s)),
        ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(int(rng.integers(16, 128)),), max_iter=200,
                                      early_stopping=bool(rng.integers(2)), random_state=s)),
        ("dt", lambda: DecisionTreeClassifier(max_depth=[None, 3, 6, 10][int(rng.integers(4))], random_state=s)),
        ("svc", lambda: SVC(C=float(10 ** rng.uniform(-1, 1.5)), probability=True, random_state=s)),
        ("ada", lambda: AdaBoostClassifier(n_estimators=int(rng.integers(20, 100)), random_state=s)),
        ("lda", lambda: LinearDiscriminantAnalysis()),
        ("qda", lambda: QuadraticDiscriminantAnalysis(reg_param=float(rng.uniform(0.0, 0.5)))),
    ]
    return pool


def synthetic_dataset(rng: np.random.Generator):
    """Random tabular classification task: random MLP / SCM-like generator with nuisance features."""
    n = int(np.exp(rng.uniform(np.log(300), np.log(2500))))
    d = int(rng.integers(2, 25))
    C = int(rng.choice([2, 2, 2, 3, 3, 4, 5, 6, 8, 10]))
    X = rng.standard_normal((n, d))
    # correlated inputs
    if rng.random() < 0.5:
        X = X @ rng.standard_normal((d, d)) * 0.5 + X
    # random MLP producing class logits from an informative subset
    n_inf = max(1, int(rng.integers(1, d + 1)))
    inf = rng.choice(d, n_inf, replace=False)
    h = X[:, inf]
    for _ in range(int(rng.integers(0, 3))):
        W = rng.standard_normal((h.shape[1], int(rng.integers(4, 32)))) / np.sqrt(h.shape[1])
        act = [np.tanh, lambda z: np.maximum(z, 0), np.sin, lambda z: z][int(rng.integers(4))]
        h = act(h @ W + rng.standard_normal(W.shape[1]) * 0.5)
    logits = h @ rng.standard_normal((h.shape[1], C)) * float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
    logits += rng.uniform(-1, 1, C)  # imbalance
    if rng.random() < 0.5:  # noisy labels (sampled) vs deterministic (argmax)
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        y = np.array([rng.choice(C, p=pi) for pi in p])
    else:
        y = logits.argmax(1)
    # discretize some features (categorical-like / ordinal)
    for j in rng.choice(d, int(rng.integers(0, d // 2 + 1)), replace=False):
        X[:, j] = np.digitize(X[:, j], np.quantile(X[:, j], np.sort(rng.random(int(rng.integers(1, 6))))))
    # relabel to contiguous classes with >= 6 samples
    cls, cnt = np.unique(y, return_counts=True)
    keep = cls[cnt >= 6]
    if len(keep) < 2:
        return None
    m = np.isin(y, keep)
    X, y = X[m], np.searchsorted(keep, y[m])
    return X.astype(np.float32), y


def simulate_task(seed: int, max_models: int = 12):
    """One committee task. Returns dict like the benchmark fold files, or None if degenerate."""
    rng = np.random.default_rng(seed)
    ds = synthetic_dataset(rng)
    if ds is None:
        return None
    X, y = ds
    C = int(y.max()) + 1
    perm = rng.permutation(len(y))
    n_tr = int(len(y) * rng.uniform(0.6, 0.8))
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(np.unique(y[tr])) < C:
        return None
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
    ytr, yte = y[tr], y[te]
    pool = family_pool(rng)
    K = int(rng.integers(3, min(max_models, len(pool)) + 1))
    chosen = [pool[i] for i in rng.choice(len(pool), K, replace=False)]
    n_inner = int(max(2, min(5, np.bincount(ytr, minlength=C).min())))
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed % (1 << 31))
    P_oof = np.zeros((len(ytr), K, C), np.float32)
    P_te = np.zeros((len(yte), K, C), np.float32)

    def fit_pred(fac, Xa, ya, Xb):
        m = fac()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xa, ya)
            p = m.predict_proba(Xb)
        out = np.zeros((len(Xb), C), np.float64)
        out[:, m.classes_] = p
        out = np.nan_to_num(np.clip(out, 0, None))
        s = out.sum(1, keepdims=True)
        return np.where(s > 0, out / np.maximum(s, 1e-12), 1.0 / C)

    names = []
    for k, (name, fac) in enumerate(chosen):
        try:
            for a, b in skf.split(Xtr, ytr):
                P_oof[b, k] = fit_pred(fac, Xtr[a], ytr[a], Xtr[b])
            P_te[:, k] = fit_pred(fac, Xtr, ytr, Xte)
        except Exception:
            P_oof[:, k] = 1.0 / C
            P_te[:, k] = 1.0 / C
            name = name + "_failed"
        names.append(name)
    return {"P_oof": P_oof, "P_te": P_te, "y_tr": ytr.astype(np.int16), "y_te": yte.astype(np.int16),
            "X_tr": Xtr, "X_te": Xte, "models": np.array(names)}
